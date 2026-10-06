# hyper – deník vývoje

Cíl: vlastní inference engine pro 3× RTX 3090 (bez P2P, GPU1 PCIe x8) + TR 3970X (4× DDR4-3200, ~77 GB/s),
dlouhodobě pro velké MoE modely přes 3 GPU + CPU/RAM. Vývojový model: Qwen3.8-27B UD-Q8_K_XL (`qwen35`,
48× Gated DeltaNet + 16× gated attention, dense FFN, 31,5 GB). Kód: `~/Projects/ik-llama-claude/hyper` (git),
build/test na `~/hyper` (sync přes `./sync.sh`).

## Baseline – mainline llama.cpp d812350 (llama-bench, -fa on, -ub 2048)

| Konfigurace | pp512 | tg128 | tg @8k | tg @32k |
|---|---|---|---|---|
| A: 3 GPU, -sm layer | 1396 | **26,4** | 26,0 | 24,8 |
| B: 3 GPU, -sm tensor | 683 | **34,6** | 34,4 | 33,4 |
| C: 20× FFN na CPU, -sm layer | – | 9,2 | 9,1 | 9,0 |

Model A/C: čas = součet zařízení (střídají se) → 26,4 t/s = 93 % stropu sériového layer-splitu.

## Decode (hyper bench, gen 256 tokenů, mělký kontext)

| Krok | t/s | ms/tok | Poznámka |
|---|---|---|---|
| M1: layer split, fp32 aktivace, repack Q8 (qs/d zvlášť) | 25,7 | 38,9 | top-1 100 %, KL 2,2e-4 vs llama.cpp |
| M2: tensor parallel 3 GPU, allreduce přes mapovanou RAM (flag), CUDA graphs | 53,0 | 18,9 | |
| LL allreduce fp32 (8B paket {val,seq}) | 56,1 | 17,8 | allreduce 22 → 15,5 µs |
| LL allreduce fp16 payload | 58,6 | 17,1 | 8,8 µs; KL beze změny |
| head-aligned GDN partition (v-hlava h → k-hlava h%16) | 60,1 | 16,7 | bez replikace q/k |
| fúze matic se stejným vstupem (gate+up, GDN in, attn qkv), 4 řádky/warp | 61,5 | 16,3 | |
| paralelní gdn_step (72 bloků), RMSNorm do GEMV se statistikou z allreduce | 64,0 | 15,6 | gdn 0,91 → 0,23 ms |
| 2 řádky/warp | **66,3** | **15,1** | GEMV ~805–823 GB/s (~90 % praxe) |

Mainline: B 34,6 t/s → hyper **1,92×**; A 26,4 → **2,5×**.

## MTP spekulativní dekódování (prompt: chat, programovací otázka s thinkingem; 512 tokenů, greedy)

Výstup je ve všech variantách **token po tokenu identický** s obyčejným greedy generováním (ověřeno 512/512).

| Krok | t/s | tokenů/krok | ověření | MTP |
|---|---|---|---|---|
| obyčejné greedy (nt=1, přes API) | 62,4 | 1 | 15,9 ms | – |
| MTP K=1 (ověření 2 tokenů, rollback GDN ze snapshotu) | 96,4 | 1,94 | 18,3 ms | 1,5 ms |
| multi-token GEMV: dequant jednou na chunk, 4 řádky/warp | 98,2 | 1,94 | 18,3 ms | 1,5 ms |
| řetězené drafty K=2 (MTP ze svého výstupu) | 109,7 | 2,68 | 21,4 ms | 3,0 ms |
| Q8 kopie LM hlavy pro drafty, jen poslední řádek | **115,1** | 2,68 | 21,3 ms | 1,9 ms |
| (K=3 s Q8 hlavou) | 111,4 | 3,22 | 25,9 ms | 2,9 ms |

Úspěšnost draftu 1: 94 %, druhého (podmíněně) ~79 %. hyper K=2 vs mainline: **3,3× (TP), 4,4× (layer)**.

| tensor-core Q8 GEMM (mma m16n8k16, váhy ve fragmentovém pořadí, split-K) K=1/2/3 | 105 / 128 / 137 | | | |
| BF16 váhy (attn qkv, eh_proj, LM hlava) → fp16 fragmenty + mma, K=3 | **145,6** | 3,24 | | |

## Prefill (pfbench: prompt N tokenů, chunky po 512)

| Krok | 512 | 4096 | 32768 | Poznámka |
|---|---|---|---|---|
| token po tokenu (původně) | ~60 | | | |
| tensor-core GEMM 128×128 (cp.async + ldmatrix), LL allreduce | 821 | 764 | | allreduce 64 % času: LL přes mapovanou RAM jen ~1,6 GB/s |
| allreduce přes copy enginy (D2H → pinned → H2D, události mezi GPU), 2 mikro-dávky na 2 streamech (komunikace jedné překrývá výpočet druhé) | 2074 | 1676 | | host vlákno na GPU + bariéra u každého allreduce |
| split-K GQA attention (decode i prefill) | 2100 | 1878 | 970 | |
| tensor-core kauzální flash attention pro prefill | **2141** | **2113** | **1880** | |

Mainline (llama-bench, -fa 1, -ub 2048): pp4096 1672 (layer) / 695 (tensor); pp2048 @ d32768 1080 (layer) / 648 (tensor).

## Dekódování v hloubce

| | @0 | @4k | @32k |
|---|---|---|---|
| hyper, attention grid (hlavy × tokeny) = 8 bloků | 62 | 46 | |
| hyper, split-K GQA (blok = kv hlava × úsek pozic, sdílí K/V mezi q hlavami skupiny) | 64 | **63** | **55** |
| mainline tensor split | 34,6 | 34,4 | 32,9 |


## Zjištění
- Allreduce bez P2P: pevná latence ~3,5 µs (n=256), zbytek přenos (PCIe; GPU1 x8). LL protokol odstraní fence + flag.
- Hash výstupu není vhodný test; správnost = top-1 + KL proti llama.cpp referenci (`build/ref`).
- RMSNorm fúzovaný naivně (každý blok počítá normu) zpomalil GEMV o ~1 ms → statistiku počítá allreduce.
- Unroll/pipelining GEMV nepomáhá; limit je DRAM, ne počet požadavků v letu.
- PCIe: GPU0/2 ~25 GB/s, GPU1 (x8) ~13 GB/s každým směrem. Zero-copy čtení mapované paměti po malých kusech je pro velké
  zprávy katastrofální (1,5 GB/s); copy enginy dosáhnou linky. Allreduce 512×5120 fp16 přes DMA ≈ 1,2 ms (limit GPU1).
- KL proti mainline roste s délkou kontextu (64 tok.: 2,2e-4, 700 tok.: 6,5e-4) stejně pro decode i prefill cestu –
  pravděpodobně fp16 zaokrouhlení částečných součtů v allreduce; pořád pod rozdílem Q8_0 vs BF16.
- GEMM prefillu běží na ~65 TFLOPS (blízko fp16/fp32-acc stropu 3090) – další zisk jen fp16 akumulace (2× rychlost na
  GeForce, ale riziko přetečení u outlierů).

## Další kroky
1. ~~Tensor-core Q8 GEMM~~, ~~prefill GEMM~~, ~~flash attention prefill~~, ~~split-K decode attention~~ (hotovo).
2. Prefill: chunked Gated DeltaNet (gdn_step je sekvenční přes tokeny, ~12 % času prefillu), gdn_conv paralelně.
3. Heterogenní režim CPU+GPU (velké modely): CPU jako 4. „zařízení“ v TP, souběžně s GPU.
