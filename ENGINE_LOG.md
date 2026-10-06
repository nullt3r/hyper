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

## Zjištění
- Allreduce bez P2P: pevná latence ~3,5 µs (n=256), zbytek přenos (PCIe; GPU1 x8). LL protokol odstraní fence + flag.
- Hash výstupu není vhodný test; správnost = top-1 + KL proti llama.cpp referenci (`build/ref`).
- RMSNorm fúzovaný naivně (každý blok počítá normu) zpomalil GEMV o ~1 ms → statistiku počítá allreduce.
- Unroll/pipelining GEMV nepomáhá; limit je DRAM, ne počet požadavků v letu.

## Další kroky
1. **MTP spekulativní dekódování** (model má NextN hlavu): ověření 2–3 tokenů ≈ cena 1 (bandwidth-bound) → odhad 1,5–1,8×.
   Vyžaduje multi-token GEMV (malé N), GDN se snapshoty stavu pro rollback, attention pro více dotazů.
2. **Prefill**: tensor-core GEMM pro Q8, chunked Gated DeltaNet, flash attention prefill (dnes jen token po tokenu).
3. Heterogenní režim CPU+GPU (velké modely): CPU jako 4. „zařízení“ v TP, souběžně s GPU.
