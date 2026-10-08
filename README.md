# hyper

A from-scratch LLM inference engine built for one specific machine: **3× RTX 3090** (24 GB each, no NVLink / P2P,
one card on a PCIe x8 link), a **Threadripper 3970X** and **256 GB DDR4-3200** (4 channels). It reads GGUF files
directly, runs on CUDA + C++, and ships an OpenAI-compatible server with a built-in web chat.

The main target is large mixture-of-experts models whose experts do not fit in VRAM: dense parts run tensor-parallel
on the three GPUs, the experts are split between the GPUs and the CPU, and the engine moves expert weights around
(prefill streaming, adaptive placement) so that the slow parts of the machine are used as little as possible.

This is a research / hobby project tuned for that hardware. It is not a general-purpose runtime: one request at a time,
three GPUs assumed, sm_86 only.

## Supported models

| Model | GGUF architecture | Tested quantization | Notes |
|---|---|---|---|
| Qwen3.8-27B | `qwen35` | Unsloth UD-Q8_K_XL | dense, Gated DeltaNet + gated attention, built-in MTP head |
| Qwen3.8-Flash-Next | `qwen4exp` | Unsloth UD-Q4_K_XL (4 shards) | MoE, hyper-connections, QSA sparse attention, PLE; optional separate NextN (MTP) GGUF |
| GLM-5.3-Flash | `glm5-next` | Unsloth UD-IQ4_XS, mainline tensor layout | MoE (288 experts), mHC residual streams, KDA linear attention, nope-MLA with a k-pool DSA indexer |

The engine is picked from the GGUF architecture.

## Performance

Measured on the machine above (October 2026), same weights for hyper and llama.cpp. Output was checked against llama.cpp
logits: the KL divergence is at the level llama.cpp shows against itself with a different batch size (the noise floor).

| Model | Metric | llama.cpp (mainline / ik_llama) | hyper |
|---|---|---|---|
| Qwen3.8-27B | decode, short context | 34.6 t/s (`-sm tensor`), 26.4 t/s (`-sm layer`) | **66 t/s**, **145 t/s** with MTP (greedy) |
| Qwen3.8-27B | prefill, 4096 tokens | 1672 t/s | **2113 t/s** |
| Qwen3.8-Flash-Next | decode, short context | 37.8 / 38.4 t/s | **74.5 t/s**, **124 t/s** with MTP (greedy), ~75–83 t/s at temperature 1 |
| Qwen3.8-Flash-Next | prefill, real text 8–21k | 461 t/s (pp512), 584 t/s (pp2048 @ 32k) | **1100–1350 t/s** |
| GLM-5.3-Flash | decode, short context | 15.7–17.0 / 16.8 t/s | **35–38 t/s** (prose) |
| GLM-5.3-Flash | decode, 16–65k context (coding agent) | ~15 t/s | **~26–28 t/s** |
| GLM-5.3-Flash | prefill, 16–32k | 280 t/s (mainline, 32k) / 164 t/s (ik) | **510–620 t/s** |

GLM decode speed depends on content: only about a third of the experts fit in VRAM, and the rest are computed by the CPU
at the speed of system RAM. See [ENGINE_LOG.md](ENGINE_LOG.md) (Czech) for the step-by-step development log with every
measurement.

## How it works

- **Tensor parallelism without P2P.** Attention heads, linear-attention heads and FFN hidden slices are split across the
  three GPUs. During decode, the per-layer allreduce runs through pinned, mapped host memory with an LL-style protocol
  (fp16 payload plus sequence flags in each packet). Prefill chunks use a copy-engine allreduce (device → pinned host →
  device), which overlaps with compute.
- **Weights in tensor-core fragment order.** Q8_0 weights are repacked at load time so that a single 16-byte load per lane
  forms an `mma.m16n8k16` operand. Decode uses a tensor-core GEMV (with split-K and a work-balanced variant for matrices
  that would otherwise leave a partial last wave). Prefill uses tiled GEMMs. Other quantization types are dequantized
  to fp16 at load, or re-quantized to Q8_0 (the GLM output layer).
- **CUDA graphs** for every decode shape (1–4 tokens). The decode critical path is mostly cross-GPU synchronization and
  memory bandwidth.
- **Mixture of experts across GPU + CPU.**
  - Experts live in their GGUF block formats (Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_XS, …), dequantized on the fly in
    GPU kernels. The most frequently routed experts (from routing statistics) are placed on the GPUs.
  - CPU-side experts sit in 2 MB huge pages and are computed with ggml's CPU dot products by a pinned thread pool.
  - GPU and CPU hand work to each other through mapped memory records, with no host round trip in the decode graph.
- **Prefill streaming.** For long prompts, the CPU-resident experts of each layer are streamed over PCIe into
  double-buffered GPU staging memory (shares weighted by link speed). Prefill therefore runs entirely on the GPUs.
  Short chunks stay on the CPU, where they are faster.
- **Adaptive expert placement (GLM).** Routing counts from the prompt and from generation are tracked with decay. The
  hottest CPU experts are swapped with the coldest GPU experts at runtime (one PCIe copy per swap, under a millisecond),
  so the GPU set follows what the conversation actually uses (prose, code, tool JSON, …).
- **Prompt cache for recurrent models.** Linear-attention layers (Gated DeltaNet, KDA) cannot be truncated like a KV
  cache, so the engine keeps snapshots of their state:
  - at message starts in the prompt,
  - every 1024 positions during generation,
  - at the end of each answer.

  A follow-up request in an agent loop therefore only recomputes the part after the last snapshot.
- **Speculative decoding with exact sampling.**
  - Draft sources: MTP heads (27B built-in, Flash-Next from a separate NextN GGUF) or prompt-lookup n-grams (GLM).
  - A draft is accepted only when it equals the token sampled at its row, so the output distribution is exactly
    plain sampling. With greedy decoding the output is token-for-token identical to non-speculative generation.
  - Rejected rows are rolled back from per-row state snapshots.
- **GLM-5.3 specifics.**
  - mHC: 4 residual streams mixed by per-token Sinkhorn-normalized matrices, fused with the Q8_0 mixing projection.
  - KDA: per-channel gated delta rule.
  - Nope-MLA attention over a 512-wide latent cache, split across GPUs by head.
  - k-pool DSA indexer: pooled keys with softmax gating, scores from 32 heads, top-512 pools selected with a one-pass
    histogram + bitmap selection.

## Building

Requirements:

- CUDA 12.x (tested 12.9) and an sm_86 GPU. Change `ARCH` in the `Makefile` for other GPUs; most kernels are tuned for
  the RTX 3090.
- g++ with OpenMP, AVX2 on the CPU.
- A **mainline llama.cpp** source tree, built with static libraries, at `$LLAMA` (default `~/llama.cpp-src`). hyper links
  against it for:
  - ggml's CPU dot products and dequantizers,
  - the tokenizer (vocab-only model load),
  - chat templates, reasoning and tool-call parsing (`libcommon`),
  - the reference tool used to validate results.

```sh
# llama.cpp (mainline), static libraries
cmake -S ~/llama.cpp-src -B ~/llama.cpp-src/build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON
cmake --build ~/llama.cpp-src/build -j

# hyper
make -j build/hyper-server build/hyper4 build/hyper build/ref
```

## Running the server

```sh
# GLM-5.3-Flash, 262k context, CPU experts on 30 threads, routing statistics persisted between runs
./build/hyper-server GLM-5.3-Flash-UD-IQ4_XS.gguf --ctx 262144 --cpu-threads 30 \
    --expert-stats glm_stats.bin --alias glm-5.3-flash --temp 1.0 --top-p 0.95

# Qwen3.8-Flash-Next with a NextN (MTP) head, 3 drafted tokens per step
./build/hyper-server Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --ctx 262144 \
    --mtp Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf --draft 3 --expert-stats fn_stats.bin

# Qwen3.8-27B with its built-in MTP head
./build/hyper-server Qwen3.8-27B-UD-Q8_K_XL.gguf --ctx 262144 --draft 3
```

Endpoints:

| Path | Purpose |
|---|---|
| `/` | web chat: conversations (stored in the browser), streaming with reasoning, markdown, code highlighting, math, per-message speed |
| `/live` | live statistics: generation and prompt-eval speed, context use, last request |
| `/v1/chat/completions` | OpenAI-compatible chat API, streaming and non-streaming, tool calls, `reasoning_content` |
| `/v1/models`, `/health`, `/stats`, `/props` | model list, health, statistics JSON, sampling defaults |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `0.0.0.0`, `8080` | listen address |
| `--ctx` | 262144 (Flash-Next 131072, GLM 65536 when not given) | maximum context; KV / latent caches are allocated for it, which takes VRAM from experts |
| `--alias` | file name | model name reported by the API |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | 0.6, 0.95, 20, 0 | sampling defaults (request fields override them) |
| `--draft` | 3 | speculative tokens per step (MTP or n-gram); `0` disables |
| `--mtp file` | — | Flash-Next: separate NextN GGUF |
| `--cpu-threads` | 30 | CPU expert threads (MoE models) |
| `--gpu-frac` | 1.0 | cap on the fraction of each layer's experts placed on the GPUs |
| `--expert-stats file` | — | routing statistics: read at start for placement, updated after every request |
| `--snapshots` | 48 | prompt-cache snapshots kept in pinned RAM |
| `--reasoning-effort` | template default | GLM: `low`, `high`, `max` or `none` (requests may override) |

Selected environment variables:

| Variable | Effect |
|---|---|
| `HYPER5_ADAPT=0` | GLM: disable adaptive expert placement (fully deterministic placement) |
| `HYPER4_STREAM_MIN`, `HYPER5_STREAM_MIN` | shortest prefill chunk that streams CPU experts to the GPUs (default 256 / 280) |
| `HYPER_DUMP_REQUEST=dir` | save every request body to `dir` (for replay benchmarks) |
| `HYPER_CPUPROF=1` | CPU expert timing and bandwidth in the log |
| `HYPER_SPIN_US` | how long CPU worker threads spin before sleeping (default 3000 µs) |

## Command-line tools

| Tool | Purpose |
|---|---|
| `ref` | runs mainline llama.cpp on a prompt and dumps the token ids and full logits (the reference). Supports experts on the CPU (`REF_CPU_MOE=1`) and only the last N rows (`REF_LAST=N`) |
| `hyper4` | MoE engines (Flash-Next, GLM): `check` / `checkpf` (KL vs the reference), `bench`, `tfbench` (fixed-content decode), `pfbench` (prefill), `ntbench`, `calib` (routing statistics), `mtpgen` (plain vs speculative, must match), `cachetest` |
| `hyper` | 27B engine: `check`, `checkn`, `gen`, `pfbench`, `cachetest`, `samptest` |
| `gemvbench`, `arbench`, `arbulk`, `iqkbench`, `bwtest`, `zctest` | microbenchmarks: Q8 GEMV per shape, allreduce, CPU expert dot products (mainline vs ik_llama), PCIe and zero-copy bandwidth |

Example correctness check:

```sh
REF_CPU_MOE=1 ./build/ref model.gguf bench/prompt3.txt ref.bin 999 512
./build/hyper4 check model.gguf ref.bin 1        # one token per forward
./build/hyper4 checkpf model.gguf ref.bin 300    # prefill 300 tokens, then decode the rest
```

## Hardware notes

- The PCIe split of streamed experts assumes GPU 1 sits on an x8 link (shares 2:1:2). Change it in
  `load_experts` for other topologies.
- GLM-5.3-Flash keeps a host copy of all experts in pinned huge pages (~140 GB of RAM).
- Running all three GPUs and the full CPU at the same time draws a lot of power. On the development machine the power
  supply shut down under sustained GLM load until the GPU clocks were capped (`nvidia-smi -lgc 210,1750`).

## Layout

```
src/        engines (engine*.cu), kernels (kernels*.cu), CPU expert pool (cpu_moe), GGUF reader, model configs
tools/      server (hyper_server.cpp + chat_page.h), CLIs, reference tool, microbenchmarks
bench/      prompts used for validation and benchmarks
ENGINE_LOG.md   development log with all measurements (Czech)
```

## Credits

Most of the code was written by Claude Opus 5.5 (Anthropic) in long agentic coding sessions, directed and tested by
Marek Malcovsky. It builds on ideas and file formats from [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), and links against llama.cpp for tokenization, chat templates
and CPU kernels.

## License

No license has been chosen yet.
