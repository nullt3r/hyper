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
| GLM-5.3-Flash | `glm5-next` | Unsloth UD-IQ4_XS, mainline tensor layout | MoE (288 experts), mHC residual streams, KDA linear attention, nope-MLA with a k-pool DSA indexer; optional NextN (MTP) block read from the original Unsloth split GGUF |

The engine is picked from the GGUF architecture.

## Performance

All numbers: this machine, October 2026, greedy decoding, the same GGUF files for hyper and llama.cpp.

### hyper (current version)

| Model (quantization) | Decode | Decode, speculative | Prefill |
|---|---|---|---|
| Qwen3.8-27B (UD-Q8_K_XL) | **59.6 t/s** at 11k context | **149 t/s** (built-in MTP, 3 drafts) | **1771 t/s** (11k-token prompt) |
| Qwen3.8-Flash-Next (UD-Q4_K_XL) | **97.5 t/s** at 11k context | **~140 t/s** (NextN MTP, up to 3 drafts) | **1774 t/s** (11k-token prompt) |
| Qwen3.8-Flash-Next Uncensored (Q4_K_M) | **93 t/s** at 11k context | **~130 t/s** (the base model's NextN MTP, up to 3 drafts) | **1806 t/s** (11k-token prompt) |
| GLM-5.3-Flash (UD-IQ4_XS) | **30.7 t/s** at 16k context | **~35 t/s** at 16k (NextN MTP, up to 3 drafts) | **998 t/s** (16k-token prompt) |

- Decode: `hyper4 tfbench` / `hyper gen` – a prefilled prompt of real text (source code and notes), then the reference
  tokens fed one per step, so the expert routing is that of real text.
- Speculative: `mtpgen` / `gen`, 9–11k prompt. The speed depends on how predictable the text is; with greedy decoding
  the output is token-for-token the plain output.
- GLM decode depends on the content: about a third of its experts fit in VRAM, the rest are computed by the CPU at
  the speed of system RAM.
- Drafts stop once the MTP head's own probability of the drafted run falls below a floor (GLM 0.85, Flash-Next 0.5):
  a verified row is not free, and a draft that is likely to be rejected costs more than it brings. The 27B verifies
  rows almost for free (dense, all in VRAM) and always drafts 3.
- The Uncensored model has no MTP head of its own; the base Flash-Next NextN GGUF drafts for it (exact as always,
  the main model verifies): +44 % at 9k context (90 → 130 t/s) vs +51 % for the base model.
- GLM MTP: every verified row costs 0.6–0.75 of a plain step (consecutive tokens rarely share CPU-held experts), so
  drafts are made only while the MTP block's own probability of the drafted run stays ≥ 0.85. Replay of 20 real Codex
  requests: decode +10 % greedy (28.3 vs 25.8 t/s), +11 % at temperature 1 (27.1 vs 24.4 t/s); prefill −3 % (the MTP
  block's weights and latent cache take VRAM from experts).

### Compared with llama.cpp

Best llama.cpp configuration found for each model (mainline d812350 or ik_llama, tuned flags, experts that do not fit
in VRAM on the CPU). The contexts are the closest ones that were measured on both sides.

| Model | Metric | llama.cpp | hyper |
|---|---|---|---|
| Qwen3.8-27B | decode | 34.4 t/s at 8k (mainline, `-sm tensor`) | 59.6 t/s at 11k |
| Qwen3.8-27B | prefill | 1396 t/s (pp512, mainline `-sm layer`) | 1771 t/s (11k prompt) |
| Qwen3.8-Flash-Next | decode | 36.1 t/s at 32k (mainline), 38.4 t/s short (ik_llama) | 97.5 t/s at 11k |
| Qwen3.8-Flash-Next | prefill | 584 t/s (pp2048 at 32k, mainline) | 1774 t/s (11k prompt) |
| GLM-5.3-Flash | decode | 15.2 t/s at 32k (mainline), 15.7 t/s at 32k (ik_llama, patched) | 30.7 t/s at 16k |
| GLM-5.3-Flash | prefill | 280 t/s (32k, mainline) | 998 t/s (16k prompt) |

### Correctness

Output is checked against llama.cpp logits (`ref` + `hyper4 check` / `hyper check`): the KL divergence is at the level
llama.cpp shows against itself with a different batch size (the noise floor). Speed changes are additionally required to
leave the output unchanged: the same KL / PPL on the reference text, the same chunked-prefill result and the same
speculative output hash before and after; kernel rewrites are compared bit for bit in microbenchmarks.
[ENGINE_LOG.md](ENGINE_LOG.md) (Czech) has the step-by-step log with every measurement.

## How it works

- **Tensor parallelism without P2P.** Attention heads, linear-attention heads and FFN hidden slices are split across the
  three GPUs. During decode, the per-layer allreduce runs through pinned, mapped host memory with an LL-style protocol
  (fp16 payload plus sequence flags in each packet). Prefill chunks use a copy-engine allreduce (device → pinned host →
  device), which overlaps with compute.
- **Weights in tensor-core fragment order.** Q8_0 weights are repacked at load time so that a single 16-byte load per lane
  forms an `mma.m16n8k16` operand. Decode uses a tensor-core GEMV (with split-K and a work-balanced variant for matrices
  that would otherwise leave a partial last wave). Prefill uses tiled GEMMs. K-quants (Q4_K, Q6_K) are repacked into
  the same fragment order in their own format (exact, no requantization); Q5_0 / Q4_0 become Q8_0 losslessly.
- **CUDA graphs** for every decode shape (1–4 tokens). The decode critical path is mostly cross-GPU synchronization and
  memory bandwidth.
- **Mixture of experts across GPU + CPU.**
  - Experts live in their GGUF block formats (Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_XS, …), dequantized on the fly in
    GPU kernels. The most frequently routed experts (from routing statistics) are placed on the GPUs.
  - CPU-side experts sit in 2 MB huge pages and are computed by a thread pool with ggml's CPU dot products, plus own
    AVX2 kernels where ggml is slow (IQ3_S dot product, Q8_K activation quantizer), bit-identical to ggml's.
  - GPU and CPU hand work to each other through mapped memory records, with no host round trip in the decode graph.
    On GPU 0 one kernel computes the routing and writes the CPU's record; the shared expert runs after it, while the
    CPU is already working.
- **Prefill streaming.** For long prompts, the CPU-resident experts of each layer are streamed over PCIe into
  double-buffered GPU staging memory (shares weighted by link speed). Prefill therefore runs entirely on the GPUs.
  Up to four 2048-token chunks go through each layer together, so a layer's experts cross PCIe once for all of them.
  Short chunks stay on the CPU, where they are faster.
- **Adaptive expert placement (GLM).** Routing counts from the prompt and from generation are tracked with decay. The
  hottest CPU experts are swapped with the coldest GPU experts at runtime (one PCIe copy per swap, under a millisecond),
  so the GPU set follows what the conversation actually uses (prose, code, tool JSON, …). The swaps are decided from
  complete counts only (the CPU queue is drained first), so the placement does not depend on timing.
- **Prompt cache for recurrent models.** Linear-attention layers (Gated DeltaNet, KDA) cannot be truncated like a KV
  cache, so the engine keeps snapshots of their state:
  - at message starts in the prompt,
  - every 1024 positions during generation,
  - at the end of each answer.

  A follow-up request in an agent loop therefore only recomputes the part after the last snapshot.
- **Speculative decoding with exact sampling.**
  - Draft sources: MTP heads (27B built-in, Flash-Next from a separate NextN GGUF, GLM from the original split GGUF)
    or prompt-lookup n-grams (GLM without MTP).
  - The NextN blocks (GLM, Flash-Next) are one attention + MoE layer. Prompt rows and kept verification rows only
    enter its cache (K/V or latents, indexer keys), so a prompt costs it a few small projections per token; only the
    row that drafts runs the whole block.
  - Drafts stop once the MTP head's probability of the drafted run drops below a floor.
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

# the same with its NextN (MTP) block from the original split GGUF (blk.45): up to 3 drafts per step
./build/hyper-server GLM-5.3-Flash-UD-IQ4_XS.gguf --ctx 262144 --cpu-threads 30 --expert-stats glm_stats.bin \
    --mtp GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf --draft 3 --alias glm-5.3-flash --temp 1.0 --top-p 0.95

# Qwen3.8-Flash-Next with a NextN (MTP) head, 3 drafted tokens per step
./build/hyper-server Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --ctx 262144 \
    --mtp Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf --draft 3 --expert-stats fn_stats.bin

# the Uncensored fine-tune drafts with the base model's NextN head
./build/hyper-server Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00001-of-00003.gguf --ctx 262144 \
    --mtp Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf --draft 3 --expert-stats fnu_stats.bin

# Qwen3.8-27B with its built-in MTP head
./build/hyper-server Qwen3.8-27B-UD-Q8_K_XL.gguf --ctx 262144 --draft 3
```

Endpoints:

| Path | Purpose |
|---|---|
| `/` | web chat: conversations (stored in the browser), streaming with reasoning, markdown, code highlighting, math, per-message speed |
| `/live` | live statistics: generation and prompt-eval speed, context use, last request |
| `/v1/chat/completions` | OpenAI-compatible chat API, streaming and non-streaming, tool calls, `reasoning_content` |
| `/v1/responses` | OpenAI Responses API (used by Codex): messages, reasoning, function / custom tool calls, streaming `response.*` events |
| `/v1/models`, `/health`, `/stats`, `/props` | model list, health, statistics JSON, sampling defaults |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `0.0.0.0`, `8080` | listen address |
| `--ctx` | 262144 (Flash-Next 131072, GLM 65536 when not given) | maximum context; KV / latent caches are allocated for it, which takes VRAM from experts |
| `--alias` | file name | model name reported by the API |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | 0.6, 0.95, 20, 0 | sampling defaults (request fields override them) |
| `--draft` | 3 | speculative tokens per step (MTP or n-gram); `0` disables |
| `--mtp file` | — | Flash-Next (and its fine-tunes): separate NextN GGUF; GLM: a GGUF with the NextN block `blk.<n_layer>` (the original split files) |
| `--cpu-threads` | 30 | CPU expert threads (MoE models) |
| `--gpu-frac` | 1.0 | cap on the fraction of each layer's experts placed on the GPUs |
| `--expert-stats file` | — | routing statistics: read at start for placement, updated after every request |
| `--snapshots` | 48 | prompt-cache snapshots kept in pinned RAM |
| `--reasoning-effort` | template default | GLM: `low`, `high`, `max` or `none` (requests may override) |

Selected environment variables:

| Variable | Effect |
|---|---|
| `HYPER5_ADAPT=0` | GLM: disable adaptive expert placement (fully deterministic placement) |
| `HYPER5_MTP_PMIN`, `HYPER4_MTP_PMIN`, `HYPER_MTP_PMIN` | MTP drafts while the drafted run's MTP probability is at least this: GLM (default 0.85), Flash-Next (0.5), 27B; 0: count from recent acceptance (Flash-Next, GLM) or always `--draft` (27B) |
| `HYPER4_STREAM_MIN`, `HYPER5_STREAM_MIN` | shortest prefill chunk that streams CPU experts to the GPUs (default 256 / 280) |
| `HYPER_DUMP_REQUEST=dir` | save every request body to `dir` (for replay benchmarks) |
| `HYPER_CPUPROF=1` | CPU expert timing and bandwidth in the log |
| `HYPER_SPIN_US` | how long CPU worker threads spin before sleeping (default 3000 µs) |

## Command-line tools

| Tool | Purpose |
|---|---|
| `ref` | runs mainline llama.cpp on a prompt and dumps the token ids and full logits (the reference). Supports experts on the CPU (`REF_CPU_MOE=1`) and only the last N rows (`REF_LAST=N`) |
| `hyper4` | MoE engines (Flash-Next, GLM): `check` / `checkpf` (KL vs the reference), `bench`, `tfbench` (fixed-content decode), `pfbench` (prefill), `ntbench`, `calib` (routing statistics), `mtpgen` (plain vs speculative, must match; GLM with `HYPER5_ADAPT=0`, `HYPER_SWEEP=k:pmin,...` compares draft policies in one run), `cachetest` |
| `hyper` | 27B engine: `check`, `checkn`, `gen`, `pfbench`, `cachetest`, `samptest` |
| `gemvbench`, `arbench`, `arbulk`, `iqkbench`, `bwtest`, `zctest` | microbenchmarks: Q8 GEMV per shape, allreduce, CPU expert dot products (mainline vs ik_llama), PCIe and zero-copy bandwidth |
| `cpumoebench`, `iq3bench` | the CPU expert decode job on real expert weights (time per job, bandwidth, output bit hash); IQ3_S / IQ4_XS kernels vs ggml, bit for bit |
| `moebench`, `mlabench`, `topkbench`, `f32bench` | GPU expert kernels for 1–4 local experts, GLM MLA decode attention, sampling candidates (top-64), fp32 router GEMV |

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

MIT, see [LICENSE](LICENSE). The IQ3_S / IQ4_NL lookup tables in `src/kernels4.cu` come from ggml (MIT).
