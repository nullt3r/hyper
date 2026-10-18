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

## Server (hyper-server, OpenAI API)

- Šablony + parsování reasoning/tool calls z mainline libcommon, tokenizer z libllama (vocab only), cpp-httplib.
- **Prompt cache**: KV platí pro společný prefix s předchozí sekvencí; rekurentní stav DeltaNetu se obnoví ze snapshotu
  (~50 MB/GPU, pinned RAM) na pozici s ≤ L−1 (MTP záznam s−1 použil token s). Snapshoty: poslední 2 začátky zpráv
  (`<|im_start|>`), starší jen ≥ 256 tokenů od předchozího, každých 4096 tokenů; max 48 (LRU-ish).
  Test `cachetest`: výstup s cache = výstup od nuly (96/96) pro stejný / rozbíhající se / prodloužený prompt.
- **Sampling se spekulací, přesný**: draft se přijme, když se rovná tokenu navzorkovanému na jeho řádku (sample-and-compare,
  pro deterministický draft je to přesně rozdělení samplingu). Kandidáti: top-64 na GPU (radix select), host dělá
  temperature/top-k/min-p/top-p. `samptest` (400 běhů, T=1, top-k 40): TV pozic 1–3 0,05–0,095 vs šum (pozice 0) 0,09.
- Malé chunky: GEMM dlaždice 32/64/128 tokenů podle velikosti, ≤ 8 tokenů GEMV, < 48 tokenů LL allreduce místo DMA
  (DMA má fixní ~0,4 ms/volání). Chunk 5 tokenů 69 → 21 ms, 128 tokenů 109 → 78 ms.
- Multi-turn (1,9k tokenů systém): 2. kolo cached 1858/1986, prompt 0,44 s místo 1,34 s; generace T=0,6: 94–126 t/s.

# Qwen3.8-Flash-Next (qwen4exp, UD-Q4_K_XL 104 GiB) – hyper v2

Záloha v1: git tag `v1-qwen27b`, `../backups/`, na hostu `~/hyper-v1-qwen27b/` + `~/run-hyper-v1.sh`.

## Baseline (llama-bench, rozložení qwen-run2: experty blk 2–10 / 23–31 / 40–47 na GPU, zbytek CPU, -fa 1, -ub 1024)

| | pp512 | tg128 | pp2048 @32k | tg64 @32k |
|---|---|---|---|---|
| ik_llama-latest (-rtr -mqkv -muge) | 530–610 | **38,4** | – (ik bench nemá -d) | – |
| mainline d812350 | 461 | **37,8** | 584 | 36,1 |

Historicky (paměť): ik qwen-run2 TG@155k 25,1 t/s.

## Architektura (z GGUF + mainline src/models/qwen4exp.cpp)
- reziduál 4 proudy × 2560 (hyper-connections, low-rank 320): před každým blokem mixer (rms per proud → down → silu(·/4)
  → up → mean_s xn·σ(gate)), po bloku res_s += out · 2σ(inject_s/4); finální mixer = output norm
- 36× Gated DeltaNet (jako 27B, výstupní gate sigmoid), 12× gated attention (24 q / 2 kv × 256, rot 64) + QSA indexer
  (4 hlavy × 128, bloky po 4 buňkách, top 2048 → pod ~2k tokenů je hustá)
- MoE 512 expertů top-10 softmax + renormalizace, sdílený expert se sigmoid gate; gate/up Q4_K (1 vrstva Q5_K),
  down Q5_1 (5 vrstev Q8_0)
- PLE ve vrstvě 1: n-gram hash (n=2,3; 16 hlav × 160) do 27 GB IQ4_NL tabulky, gate přes key/query, kauzální konv K=4 dil=3

## hyper4 – dekódování (bench: ref2 prompt 64 tok., 256 greedy tokenů; kalibrace rozmístění na jiném textu ref4)

Správnost: KL vs mainline ref (256 tok.) 0,042–0,047, top-1 88–91 %. Šumová podlaha mainline sám proti sobě
(ubatch 256 vs 1 vs 16): KL 0,030–0,038, top-1 89–91 % → MoE routing je chaotický, hyper je na podlaze.

| Krok | t/s | jen GPU (CPU experty ignorovány, jen čas) |
|---|---|---|
| v0: naivní MoE kernely, 60 % expertů na GPU podle indexu | 40,0 | |
| CPU experty zkopírované do 2 MB stránek (mmap → anon + MADV_HUGEPAGE) | 47,0 | |
| 75 % expertů na GPU | 46,5 (ref2) / 53,2 (ref4) | |
| rozmístění podle četnosti (kalibrace `hyper4 calib`, round-robin mezi GPU) | 51,3 | 59,9 |
| MoE: řádek na warp + vektorové loady, router přes rank, f32 GEMV blok na řádek | 69,4 | 81,5 |
| split-K Q8 GEMV pro matice s málo řádky (HC down 320 ř., shexp, router), deterministický součet | **74,5** | 97,6 |

Baseline ik 38,4 / mainline 37,8 → **1,94×**. Bugy cestou: CPU aktivace kvantizovány typem vah místo vec_dot_type
(NaN); race v poolu CPU vláken (vlákno z předchozí generace „spotřebovalo“ index úlohy → zaseknutí) → claim přes CAS
na (generace << 32 | index).

## hyper4 – prefill (pfbench: 4096 tokenů, chunky po 512)

Správnost: `checkpf` (prefill 128 tok. → dekódování zbytku po tokenu) KL 0,033–0,037 = šumová podlaha; výsledek je bitově
stejný pro chunky 128 i 50+50+28 (každý token se počítá stejně nezávisle na rozdělení).

| Krok | prefill t/s |
|---|---|
| v0: GEMM + DMA allreduce + flash attn, MoE po párech (GEMV), CPU experty seskupené podle experta | 455 |
| skupinový tensor-core GEMM pro experty (dekvantizace do fp16 v shared memory, váha 1× za chunk), cuBLAS sgemm pro fp32 router | 531 |
| 83 % expertů na GPU | 667 |
| 30 CPU vláken (16: 545, 48: 727 ale horší dekódování) | **708** |

Baseline: mainline pp512 461, pp2048@32k 584; ik pp512 530–610. Profil: GPU0 čeká v moe_reduce na CPU experty
(~40 % času) → další krok: překrytí CPU a GPU (2 mikro-dávky), rychlejší CPU kernel (víc tokenů na řádek vah).
Pozor: QSA (sparse attention nad ~2k tokenů) zatím chybí → výsledky jsou přesné jen do ~2k kontextu.

## hyper4 – QSA + server

- QSA (sparse attention): indexer replikovaný na GPU (q/k projekce BF16→fp16 frag., raw klíče fp16, bloky po 4: průměr →
  RMS norm → RoPE na první pozici), skóre Σ_h relu(q·k)/√128, top-512 radix select (8bit číslice, shody → nižší index),
  attention přes seznam buněk (split-K kernel). Dekódování vždy přes seznam (do ~2k = 0..p), prefill hustý flash dokud
  celý chunk ≤ 2048 bloků. **Výběr bloků se shoduje s mainline 511/512** (vrstva 3, token 3199; jediný rozdíl = těsná
  shoda skóre). KL na 12k je vyšší (0,21 vs mainline-vs-mainline 0,05) kvůli chaotickým přepnutím výběru/routingu
  na citlivých pozicích – stejné pozice skáčou i u mainline proti sobě.
- 32k: prefill 513 t/s (mainline pp2048@32k 584), dekódování 51,6 t/s (mainline 36,1).
- Server: společné rozhraní `LLM` (llm.h), server vybere engine podle arch (qwen35 → v1, qwen4exp → Engine4).
  Engine4: prompt cache (snapshoty DeltaNet + PLE konvoluce), sampling top-64 z GPU. `cachetest`: stejný i rozbíhající
  se prompt 64/64 identických; prodloužený se rozejde po 16 tokenech (jiné dělení chunků → GEMV vs GEMM numerika).
- `~/run-hyper-fn.sh` (tmux hyper, :8080): ctx 131072, 65 % expertů na GPU → dekódování ~49 t/s (T=1,0).
  TODO: rozdělení attention po kv hlavách (GPU1 drží obě kv hlavy → méně místa na experty), MTP, překrytí CPU/GPU.

## hyper4 – plný kontext 262k, prefill na reálném textu

Pozor na benchmark: `pfbench` s opakovanými tokeny (ref2 × N) soustředí routing na pár (horkých) expertů → nereálně
rychlé. Reálný text (`HYPER4_PFREAL=1`, ref6 = 12k tokenů zdrojáků/deníku) zasáhne skoro všechny studené experty v RAM.
Server na promptu z opencode ukázal ~190 t/s (131k kontext, 65 % expertů na GPU, kalibrace z jiného textu).

| Změna | prefill t/s (reálný text 8k) |
|---|---|
| CPU experty, chunk 512 | 790–820 |
| streamování studených expertů na GPU (pinned, podíl podle PCIe 2:1:2, 2 staging buffery, upload l+2 během l), chunk 512 | 643–662 (každý chunk přenese ~22 GB) |
| chunk 2048, CPU experty | 1144–1210 |
| chunk 2048, streamování | **1271–1357** |
| server, 21k tokenů reálného textu | **1122** (předtím 528) |

- Attention po kv hlavách (GPU0 bez attention → víc expertů; GPU1/2 jedna kv hlava = 3,2 GB KV při 262k)
- Experty podle volné VRAM každé GPU (dvě fáze načítání) → 262k kontext: 132/103/100 expertů na vrstvu (65 %)
- Server ukládá statistiky routingu po každém požadavku (`--expert-stats`, kumulativně) → při dalším startu
  rozmístění podle skutečného používání.

## hyper4 – MTP (NextN z `~/models/mtp/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf`)

- Obecné husté váhy: Q8_0 (rychlá cesta) nebo fp16 dekvantizované při načtení přes ggml `to_float` (bezztrátové);
  malé tenzory / embeddingy z libovolného typu. (Nutné pro MTP soubor: Q4_K/Q5_0/Q6_K.)
- MTP blok = vrstva 48: hustá attention (compress_ratio 0), HC mixery, vlastních 512 expertů (v rozpočtu VRAM jako 49.
  vrstva, CPU slot 48), vstup [rms(e)·enorm | rms(h_s)·hnorm_s] → eh_proj po proudech, výstupní mixer = output_hc_*
  z MTP souboru (předictor-only layout jako v ik). Rollback: snapshoty DeltaNet conv/state + PLE konvoluce po řádcích.
- `mtpgen` (greedy, ref2 prompt, 256 tok.): **plain 77,6 → MTP 124,1 t/s**, výstup identický 256/256,
  1,91 přijatých draftů/krok (K=3); K=2: 124,9 (1,51), K=1: 110,7. Norma skrytého stavu po proudech 1,91 vs přes
  všechny proudy (ik) 1,88 → ponechána po proudech.
- Server (T=1,0, 262k ctx, 61 % expertů na GPU): krátký prompt 83,4 t/s (předtím ~49), 21k kontext 55 t/s.

## 2026-10-09/10 – vícechunkový prefill a optimalizace kernelů a přenosů

Všechny změny mají **bit-identický výstup** (stejné KL/PPL na referenci, stejný výsledek `checkpf`, stejný hash
spekulativního/MTP výstupu; přepsané kernely porovnané bit po bitu v mikrobenchmarcích). Varianty, které měnily
numeriku, zůstaly vypnuté (env).

Měřicí podmínky (stejné před i po): dekódování `tfbench` (prefill reálného textu, pak referenční tokeny po jednom),
Flash-Next 11k kontext, GLM 16k; prefill `pfbench` s reálným textem (`HYPER4_PFREAL=1`); 27B `gen` / `pfbench` 11k.

| Model | Metrika | před | po |
|---|---|---|---|
| GLM-5.3-Flash | dekódování @16k | 27,4–27,6 t/s | **30,7 t/s** (+11 %) |
| GLM-5.3-Flash | prefill 16k | 653–656 t/s | **998 t/s** (+52 %) |
| Qwen3.8-Flash-Next | dekódování @11k | 91,7–92,3 t/s | **97,5 t/s** (+6 %) |
| Qwen3.8-Flash-Next | MTP (prompt 9k, greedy) | 116 t/s | **138–142 t/s** (+20 %) |
| Qwen3.8-Flash-Next | prefill 11k | 1486–1491 t/s | **1774 t/s** (+19 %) |
| Flash-Next Uncensored | dekódování @11k / prefill 11k | 89 / 1491 t/s | **93 / 1806 t/s** |
| Qwen3.8-27B | dekódování / MTP / prefill 11k | 58,4 / 126 / 1774 t/s | **59,6 / 130 / 1771 t/s** |

Co pomohlo:
- **Prefill po vrstvách pro až 4 chunky** (GLM i Flash): streamované CPU experty vrstvy přejdou přes PCIe jednou
  pro všechny chunky; MTP drafty po chuncích.
- **GLM: routing publikovaný CPU před sdíleným expertem** (CPU začne o ~50 µs/vrstvu dřív) a **routing + zápis
  záznamu pro CPU v jednom kernelu** (oba MoE enginy).
- **GLM MLA dekódovací attention 85 → 48 µs/vrstvu** (q v registrech po warpech, buňky po 4, váhy softmaxu jako float4,
  dávkované globální loady); **stav KDA uložený po sloupcích** (souvislé loady místo kroku 512 B, 15 → 3 µs).
- **CPU experty:** přesný AVX2 kvantizátor Q8_K (ggml má na x86 jen skalární referenci), IQ3_S dot s lookupy
  ve skalárních registrech (+17 % na jádro), vážený součet rozdělený mezi vlákna, MADV_COLLAPSE na hostovské kopie.
- **Výběr kandidátů pro sampling (top-64) dvoustupňově:** 117 → 17 µs na token (všechny enginy).
- **Flash-Next:** Q5_1/Q5_0 dekvantizace bez větvení podle lane, indexer zařazený před q/k/v projekci, prep + pool
  indexeru v jednom kernelu, předčítání buněk v attention (i 27B).
- **Oprava:** adaptivní rozmístění expertů GLM záviselo na časování (rebalance četl počty routingu dřív, než je
  CPU vlákno dopočítalo) → nejdřív se vyprázdní fronta CPU, rozmístění je deterministické.

Slepé uličky (změřeno): zero-copy čtení části CPU expertů přes PCIe, překrytí down projekce posledního experta
(čtení částí řádků půlí propustnost RAM), vyvažování počtu úloh, vícetokenové CPU doty (úlohy jsou omezené přenosem
různých expertů), GPU MoE kernely s loady předem, fúze mHC mix + pre, MLA na tensor cores s fp16 (není přesné),
dvakrát `#pragma unroll`, který změnil kontrakci FMA (vráceno).

Nástroje: `cpumoebench`, `iq3bench`, `mlabench`, `topkbench`, `f32bench`.

## 2026-10-10 – GLM: MTP (NextN) hlava

NextN blok `blk.45` je jen v původních Unsloth GGUF (5 shardů; `to_mainline.py` ho vynechal), data jsou bajtově
stejná → `--mtp <shard 1>` / `HYPER4_MTP` otevře původní soubor a vezme z něj jen blk.45 (vrstva s indexem n_layer
ve stejných polích: MLA + MoE bez mHC, experti Q3_K/Q4_K → nová dekvantizace Q3_K na GPU, bit po bitu = ggml).

Sémantika podle ik_llama (`build_glm5next_mtp` + `common/speculative.cpp`): řádek MTP na pozici q = (token q,
skrytý stav hlavního modelu na q−1 = vstup LM hlavy, tj. output_norm průměru 4 mHC proudů); token na pozici 0
vynulovaný; x = eh_proj([enorm(emb) | hnorm(h)]), pre-norm MLA s indexerem, MoE + sdílený expert, shared_head_norm →
LM hlava; řetězené drafty berou vlastní výstup po shared_head_norm.

| 16k kontext, greedy, 300 tokenů | t/s | vs obyčejné | ověřený krok |
|---|---|---|---|
| obyčejné dekódování | 30,3–31,2 | | 32–33 ms |
| 1 draft | 33,3–35,5 | +10–14 % | 52 ms (1,6×) |
| 2 drafty / 3 drafty (pevně) | 32,4 / 31,7 | +2 / +4 % | 74 / 93 ms (2,35× / 2,9×) |
| **do 3 draftů, dokud p(MTP) běhu ≥ 0,85** | **34,7–36,4** | **+11–17 %** | 61 ms, 1,37 draftu/krok, 96 % přijato |

Ověřovací řádek stojí 0,6–0,75 kroku: po sobě jdoucí tokeny skoro nesdílejí experty na CPU, takže každý další řádek
čte z RAM nové experty (proto ik_llama s MTP na GLM zpomalil). Vyplatí se jen drafty s velkou šancí → pravděpodobnost
draftu z MTP (max + součet exp po GPU, `max_sumexp`) a řetěz končí, když součin klesne pod 0,85 (`HYPER5_MTP_PMIN`).

Reálný provoz – replay 20 požadavků Codexu (`tools/replay.py`, 300 tokenů, server s `--mtp --draft 3`):

| | dekódování | prefill | experti na GPU / vrstvu |
|---|---|---|---|
| bez MTP, greedy | 25,8 t/s | 752 t/s | 31/32/31 |
| MTP, greedy | **28,3 t/s (+9,9 %)** | 733 t/s (−2,6 %) | 29/31/29 |
| bez MTP, teplota 1 | 24,4 t/s | 760 t/s | |
| MTP, teplota 1 | **27,1 t/s (+11,4 %)**, 92 % draftů přijato | 735 t/s (−3,2 %) | |

Prefill ztrácí kvůli VRAM (váhy bloku, latentní cache MTP přes celý kontext, snapshoty KDA pro rollback).
MTP vrstva v prefillu i u ponechaných ověřovacích řádků jen **zapisuje do své cache** (latenty, klíče indexeru závisí
jen na vstupu bloku): bez attention, MoE a hlavy → prefill MTP téměř zdarma (dřív −4,3 %), MTP za krok 4,9 → 3,6 ms.
Celou vrstvou jde jen poslední řádek (ten, co dává draft). Drafty jsou s tím stejné (`HYPER5_MTP_FULL` = stará cesta,
A/B s `HYPER5_ADAPT=0`: shodné počty draftů i hash výstupu).

Výstup: s pevným rozmístěním (`HYPER5_ADAPT=0`) je spekulativní greedy výstup **token po tokenu shodný** s obyčejným
(300/300, stejný hash pro všechny politiky draftů); hlavní cesta beze změny (CHECK4 KL 0,027086, PPL 7,6363).
Nástroje: `mtpgen` s `HYPER_SWEEP=k:pmin,...` (víc politik v jednom procesu), `deqtest` s Q3_K.

## 2026-10-10 – MTP i pro ostatní modely: Uncensored, Flash-Next, 27B

- **Uncensored (orcarouter Q4_K_M) s NextN hlavou základního Flash-Next** (`--mtp Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf`,
  beze změny kódu; výstup přesný, ověřuje hlavní model): prompt 9k **90 → 124 t/s**, s prahem 0,5 **130 t/s (+44 %)**;
  1,5k 85 → 116 t/s (+36 %). Přijato 1,46 draftu/krok (základní model 1,73) – fine-tune skryté stavy mění jen málo.
- **Flash-Next: MTP vrstva jako u GLM** – řádky promptu jen do cache bloku (K/V, klíče indexeru; bez attention, MoE
  se streamováním expertů a hlavy), z ponechaných ověřovacích řádků jde celou vrstvou jen poslední (FFN v decode
  režimu pro 1 řádek). MTP za krok 2,59 → 2,35 ms, prefill s MTP 5,44 → 5,37 s (9k). `HYPER4_MTP_FULL` = stará cesta.
- **Práh pravděpodobnosti draftů i pro Flash-Next** (`HYPER4_MTP_PMIN`, výchozí 0,5): 9k 138 → 142,5 t/s,
  1,5k 114 → 118 t/s; Uncensored 124 → 130 (9k), 111 → 116 (1,5k). Nižší práh (0,2) je horší než žádný.
- **27B: 3 drafty místo 2** (výchozí): 11k 130 → **149 t/s**, 1,5k 119 → 126 t/s. Ověřovací řádky jsou u hustého
  modelu ve VRAM skoro zadarmo, práh nepomáhá (±1–2 %, `HYPER_MTP_PMIN` zůstává 0).
- **Oprava přesnosti: split-K attention dělila cache podle počtu řádků** (`256 / (n_kv · nt)`), takže 3- a 4-řádkové
  ověření zaokrouhlovalo jinak než krok po jednom tokenu (27B se 3 drafty: shodný prefix jen 206/256, KL nt=4 0,000222
  vs 0,000221). Teď dekódovací dávky (≤ 4 řádky) dělí jako jeden řádek: KL nt=1..4 shodné (27B 0,000221 / max
  0,00134; Flash 0,048582 / 4,35564), výstup se 3 drafty 256/256, rychlost beze změny.
- `mtpgen` / `hyper gen`: `HYPER_SWEEP=k:pmin,...` pro všechny enginy (víc politik v jednom procesu, hash výstupu, prefill).

## 2026-10-10 – audit přesnosti

- **Sampling byl ořezaný.** GPU posílají 64 kandidátů na řez slovníku a `top_k 0` znamenalo „64 nejlepších“;
  navíc výchozí `top_k 20` serveru (doporučení Qwenu) platilo i pro GLM, kde je doporučeno jen T 1,0 / top_p 0,95.
  Na referenčních logitech GLM: top_k 20 ořízlo nucleus na 34 % pozic (průměrná TV vzdálenost 0,089 na token),
  limit 64 na 26 % (0,056). Nově `src/sampling.h` pro všechny enginy: když nabraná množina (top_k / min_p / top_p)
  nemůže sahat za kandidáty, rozhodnou oni s přesným normalizátorem z GPU (`max_sumexp` s teplotou); jinak se načte
  celý řádek logitů a množina se najde přesně (koše log-váhy, třídí se jen hraniční koš). `samplertest`: 90 nastavení
  × řádky referencí, množina i pravděpodobnosti = úplné seřazení (rozdíl ≤ 3e-13). Při T 1 / top_p 0,95 se celý řádek
  čte na ~28 % pozic. Server: GLM má výchozí T 1,0 / top_p 0,95 / top_k 0 (vypnuto), `/props` hlásí skutečné hodnoty.
- **fp16 částečné součty mezi GPU** (LL i prefill) – hypotéza z dřívějška (růst KL s kontextem) **vyvrácena**:
  varianta s přesnými fp32 částmi (`HYPER_AR32=1`, dvojnásobné sloty) nemění KL na žádném modelu (27B 0,000221 /
  0,000653 → 0,000222 / 0,000651; Flash, Uncensored, GLM v šumu). Zůstává fp16.
- **KL vs šum llama.cpp na stejném textu** (reference lišící se jen ubatch / CPU experty): GLM 0,038–0,040 vs 0,043
  (v šumu); Flash 0,046–0,062 vs 0,029–0,038; Uncensored 0,104–0,111 vs 0,069. Trasování po vrstvách
  (`hyper4 trace` + `ref` s `REF_DUMPLAST`/`REF_TOKENS`) na 256. tokenu: hyper se od llama.cpp liší už za první
  vrstvou o 1,7 % (llama.cpp sama proti sobě 0,3–0,5 %), konkrétně v hc mixeru vrstvy 0 ze stejného vstupu.
  Přesný výpočet mixeru v double z vah souboru: **hyper 3,3e-5 relativně, llama.cpp 1,7 %** – llama.cpp kvantizuje
  aktivace na 8 bitů (q8_1 bloky po 32) a vstup mixeru má odlehlé hodnoty (max 12,9 při RMS 1,13); její varianty
  sdílejí stejné zaokrouhlení, proto se shodují mezi sebou. Rozdíl je chyba reference, ne hyperu.
- Rozdíly podle umístění expertů (CPU: 8bitové aktivace jako llama.cpp, GPU: fp16/fp32) zůstávají: jsou vlastní
  dělení CPU/GPU (llama.cpp GPU vs CPU experty: KL 0,069) a na kvalitu nemají vliv.

## 2026-10-11 – nápady ze Strata (Niko1221/Strata: Flash-Next na jedné herní GPU, 2–3bitové kvanty, int8/4bit KV)

Srovnání: Strata na 2× RTX 3090 s IQ3_S (3 bity) a int8 KV 137–145 t/s s MTP; hyper na 3× 3090 s UD-Q4_K_XL (4 bity)
~140 t/s s MTP. Rychlost Strata jde hlavně z nižší přesnosti (kvanty, KV), to nepřebíráme.

- **Sampling s Gumbelovým šumem pevným pro (seed, pozice, token):** tah = argmax(log p + G) – přesné vzorkování;
  pozice jsou pozice vzorkovaného tokenu, takže **spekulativní běh dá stejný text jako obyčejný i při samplingu**
  (Flash, 27B, GLM s pevným rozmístěním: 256/256 resp. 300/300). `samplertest`: TV Gumbelova tahu = TV multinomického
  (0,0036 vs 0,0040 při 200k tazích).
- **Drafty svázané se samplingem** (myšlenka ze Strata): MTP hlava vrací top-64 kandidátů na řez, host z nich táhne
  draft se stejným ořezem (top_k / min_p / top_p) a stejným šumem. Šum přes celý slovník bez ořezu škodil (koncové
  tokeny mimo nucleus: 27B −5 %, GLM přijetí 1,17 → 0,97). S ořezem při T 0,6: **Flash 128 → 132 t/s** (přijato
  1,64 → 1,73 / krok), 27B 142 → 138 (jeho greedy drafty se trefují lépe → vypnuto), GLM T 1 s pevným rozmístěním:
  přijetí nikdy nižší, jeden draft bez prahu 0,79 → 0,88, s prahem ±2 % (šum). `HYPER_GUMBEL_DRAFT=0/1`.
- **Přesouvání expertů GLM bez zastavení dekódování** (odcházející expert hned na CPU, příchozí se kopíruje na pozadí):
  **pomalejší** (29,4–29,7 vs 30,3–30,6 t/s) – během kopírování počítá CPU oba a kopie se perou o PCIe s allreduce;
  zastavení přitom stojí jen ~0,4 % (1–7 přesunů za 1–3 ms na 16 tokenů). Vráceno.
- **Jeden rozpočet VRAM přes všechny vrstvy místo stejného počtu na vrstvu:** podle reálných statistik routingu by
  přidal 0,04 b. (GLM) / 0,6 b. (Flash) podílu routingu na GPU – nedělá se.
- Adaptivní rozmístění pro Flash-Next: CPU experty stojí ~1 ms z 10,2 ms na token (HYPER4_NOCPU 109 vs 98 t/s),
  přínos nejvýš 2–3 % – odloženo.

## 2026-10-11 – adaptivní rozmístění expertů i pro Flash-Next

Uživatel na serveru (262k kontext, T 1,0) viděl 145 t/s u kódu, ale 50–60 t/s v delším českém chatu: krok byl dražší
(~29 ms místo ~23 ms) při menším počtu ověřovaných řádků → experty, které chat používá, ležely na CPU (rozmístění ze
statistik převážně z kódu; při 262k kontextu drží GPU1/GPU2 KV cache, na GPU je jen 59–63 % expertů).
Port mechanismu z Engine5: kopie všech expertů v hostu (2 MB stránky, pinned), skóre z routingu promptu (váha 0,25) a
z dekódování (decay 0,92), výměna nejteplejších CPU expertů za nejstudenější GPU (stejný slot) po promptu a každých
16 kroků (max 64), streamování prefillu ze seznamů expertů (běhy po sobě jdoucích). `HYPER4_ADAPT=0` = staré chování
(KL bitově shodné: 0,048582).

| 262k kontext | bez adaptace | s adaptací |
|---|---|---|
| tfbench 11k (kód/poznámky) | 87,1 / 88,9 t/s | **92,9 / 93,1 t/s (+5,5 %)** |
| replay 20 požadavků Codexu, T 1, MTP – dekódování | 132,9 t/s | **138,7 t/s (+4,4 %)** |
| – prefill | 1248 t/s | **1352 t/s (+8,3 %)** |

Výměny za celý replay 1,3 s (14k výměn). Na obsahu vzdálenějším od statistik (jiný jazyk, chat) čekám víc.


## 2026-10-11 – offline experimenty: n-gram drafty, CPU experty, rezerva KV cache

Bez spuštění modelu (server běžel dál): simulace na skutečných datech + mikrobenchmarky CPU expertů.

**N-gram drafty (prompt lookup) k MTP – simulace na 35 sezeních z Codexu** (`tools/spec_render.py`, `tokdump`,
`tools/spec_sim.py`): 521k tokenů, z toho 183k vygenerovaných (tool cally 103k, reasoning 75k). Draft se počítá jen
když se shoduje s tím, co model skutečně vygeneroval. Nákladový model z replaye Flash-Next (262k, T 1): krok s R
ověřovanými řádky ≈ 5,9 + 4,4·R ms (+1,2 ms řetěz MTP), MTP 1,88 přijatých / krok. Nejlepší hybrid (n-gram draft 3–5
tokenů, jen při shodě ≥ 12 tokenů, jinak MTP): **+5–6 %**; delší drafty ztrácejí (každý řádek navíc stojí 4,4 ms =
43 % tokenu – MoE: každý řádek čte jiné experty). Neimplementováno, malý zisk.

**CPU experty (cpumoebench, reálné váhy vrstvy 10, 30 vláken, výstup bitově stejný – hash):**

| úloha | RG 16 / RD 32 (dosud) | RG 32 / RD 128, předstih gate/up 2 |
|---|---|---|
| Flash 3 experti × 1 token | ~61–64 GB/s | 63–67 GB/s |
| Flash 3 × 4 (MTP ověřování) | ~63 GB/s | **68–69,5 GB/s** |
| Flash 6 × 4 | ~62 GB/s | **68–69 GB/s** |
| GLM 5 × 1 | 68,6 GB/s | **70,1 GB/s** |
| GLM 5 × 2 | 62,0 GB/s | 64,2 GB/s |

Delší souvislé čtení na úlohu; profil 3 × 4: příprava 10 µs, ocas down posledního experta 45 → 30 µs. Strop RAM
(čisté čtení) 77–80 GB/s → zbývá ~12 %. Vlákna: 24–48 stejné, 62 (SMT) polovina.

**Kolik routingu jde na CPU (statistiky serveru `fn_stats.bin`, 1,6M routingů / vrstvu):** dnes 303 expertů na vrstvu
na GPU (262k KV rezervováno) → 9,4 % routovaných expertů na CPU; 340 → 6,3 %; 380 → 3,7 %; 400 → 2,7 %.
MTP krok se 4 řádky: ~3,7 CPU expertů / vrstvu ≈ 167 µs proti ~60 µs GPU části MoE → CPU je kritická cesta MoE.
Nápad: **pružná KV cache** – nevyužitou část rezervy 262k (GPU1/2 po 3,2 GB) půjčit expertům a vracet je, když kontext
roste (hostovská kopie expertů už existuje, vrácení = jen změna tabulky). Kapacita 262k zůstává.

Ověření na modelu (replay 20 požadavků Codexu, greedy, 262k, MTP 3, rozmístění ze zmrazených statistik; `~/rp_ab.sh`):
staré RG/RD 148,4 / 151,9 t/s → nové **157,5 / 156,4 t/s (+4,5 %)**, prefill beze změny (~1400 t/s), výstupy 20/20
identické (porovnání bez náhodných id položek odpovědi – hash v `replay.py` je zahrnuje). `HYPER4_NOCPU` (CPU experty
vynechány, jen čas): 184,8 t/s → CPU experty stojí ~15 % dekódování. Server s `--ctx 65536` (68 % expertů na GPU místo
59 %, horní mez pro pružnou KV cache): 180,6 t/s, na krok +9 % (jiný výstup → jiná míra přijetí), prefill +4,5 %.

**Pružná KV cache (Flash-Next, `HYPER4_ELASTIC`, výchozí zapnuto)** – `src/vmm.h`: K / V a klíče indexeru jako virtuální
rozsahy (CUDA VMM, granule 2 MB), mapované jen pozice v užívání (kroky 8192 pozic). Zbytek plné velikosti 262k (GPU1/2 po
~4,1 GB) = sloty navíc pro nejteplejší CPU experty (pole expertů rezervovaná pro všech 512 slotů, mapuje se jen
používaný začátek). Invariant: cache + sloty navíc ≤ plná rezerva cache → 262k se vždy vejde.
- Doplňování na pozadí: vlákno mapuje sloty a kopíruje na vlastním streamu (CPU ty experty mezitím počítá dál);
  převzetí v **pevném kroku dekódování** (16 kroků na 256 expertů), zrušená dávka se zahodí celá → deterministické
  (s převzetím „až dokopíruje“ bylo jen 6/20 shodných výstupů mezi dvěma stejnými běhy).
- Růst kontextu: nejstudenější expert ve slotu navíc se stane CPU expertem (bez kopie, host má kopii všech), paměť jde
  cache. Kratší konverzace vrátí paměť expertům až po 32 krocích dekódování (vedlejší požadavky agenta nic nemění).
- Mapování 75 µs / granule, odmapování 90 µs, H2D 13 GB/s na x8 (GPU1).

Měření (vše v jedné relaci, GPU bez capu, rozmístění ze zmrazených statistik):

| | vypnuto | zapnuto |
|---|---|---|
| replay 20 požadavků Codexu, 262k, MTP, greedy – dekódování | 155,2 t/s | **176,8 / 176,2 t/s (+14 %)**; na krok +6 % |
| – prefill | 1399 t/s | 1416–1424 t/s |
| tfbench 10k, 2000 tokenů po jednom | 103,7 t/s | **107,6 t/s (+3,8 %)** |
| KL ref4 (256) | 0,047464 | 0,045267–0,047469 |
| checkpf ref7_5k (prefill 3801, 1199 dekódovaných přes hranici vracení; krok 256, bez rezervy) | 0,040484 | 0,041279 |

Experty ve slotech navíc: ~2600 (~55 na vrstvu) při krátkém kontextu. Vypnuto = bitově stejné jako dřív (20/20).
Simulace n-gram draftů s čerstvým nákladovým modelem (krok 9,6 ms + ~4,0 ms / řádek, s pružnou cache ~3,1 ms / řádek):
nejlépe +6,5 % / +7,5 % (draft 5, shoda ≥ 24 tokenů).

**Pružná KV cache i pro GLM** (Engine5): latentní cache MLA (512 × fp16 na pozici) a pooled klíče indexeru jsou na
každé GPU → půjčuje se na všech třech (~2,9 GB na GPU při krátkém kontextu). Experty GLM ~11,7 MB → ~17 navíc na MoE
vrstvu (29 % → ~35 %). Převzetí dávky 16 kroků na 64 expertů. Oprava (obě enginy): zrušená dávka (růst kontextu během
doplňování) teď odmapuje své rezervované sloty – předtím v nich zůstala paměť a GLM v replayi dostal jen 143 expertů
navíc místo ~700.

| GLM, 262k, vše v jedné relaci | vypnuto | zapnuto |
|---|---|---|
| tfbench 16k (glm_ref5L, 512 tokenů) | 26,4 / 25,8 t/s | **28,7 / 28,7 t/s (+10 %)** |
| replay 20 požadavků Codexu, greedy, bez MTP | 24,4 t/s | **27,0 t/s (+11 %)** |
| KL glm_ref5 (2048) | 0,026341 | 0,027519 (šum) |
| replay, dva běhy | | 20/20 shodných |

Pozn.: dřívější GLM „30,7 t/s @16k“ bylo s `HYPER_MAXPOS=24576` (rezerva KV jen na 24k → víc expertů na GPU), ne s 262k.

Flash-Next znovu, čistě (zahřáté, vypnuto–zapnuto–vypnuto): 156,0 / **177,8** / 154,7 t/s, prefill 1386 / 1436 / 1391.
První běh po vypnutí produkčního serveru je zkreslený (jádro uvolňuje ~100 GB a vytlačuje page cache: prefill prvního
požadavku 69 s místo 16 s) → při A/B vždy jeden zahřívací běh.


## 2026-10-13 – obrázky pro Qwen3.8-Flash-Next (`--mmproj`)

Vision encoder: unsloth `mmproj-F16.gguf` (904 MB, projektor `qwen3vl_merger`, 27 bloků, výstup 2560 = `n_embd`). Kóduje ho
mainline libmtmd (statická `libmtmd.a`) na GPU 0. Načítá se před enginem, takže rozmístění expertů s jeho VRAM (~1,1 GiB)
počítá: GPU 0 drží 114 místo 122 expertů na vrstvu.

Model používá prokládané M-RoPE (IMROPE, sekce 11/11/10). Obrazový token má pozici (t, h, w) podle buňky mřížky a text
za obrázkem pokračuje od n_past + max(nx, ny). Engine proto má na každé GPU tabulku pozic [buňky][3]:
- Kernely `attn_prep`, `idx_prep`, `idx_prep_pool` a `idx_pool` z ní čtou složku podle páru frekvencí (`mrope_pos`).
  Poolovaný klíč se točí podle pozice svého prvního členu (jako v mainline).
- Pozice promptu se zapíšou jednou za požadavek, budoucí buňky dostanou „buňka + posun“. Dekódování tak nic nepřidává.
- Text má všechny tři složky rovné indexu, výpočet je tedy stejný jako dřív.
- Obrazové pozice dostávají řádky encoderu místo embeddingu tokenu, a to i MTP (řádek t nese token pos + t + 1).
- PLE na nich hashuje `ple.image_token_id` (248056), stejně jako mainline.
- Prompt cache porovnává klíče: obrazové pozice nesou záporný hash obrázku. Tabulka pozic se odkládá s konverzací.

Ověření:

| Test | Výsledek |
|---|---|
| replay 20 požadavků Codexu, greedy, tabulka vs. `HYPER4_NO_MROPE=1` | 20/20 shodných, 187,0 / 186,5 t/s, prefill 1294 / 1291 |
| syntetický obrázek (text, tvary, barvy, čísla) | vše správně |
| titulní strana NYT 1969 | titulek, podtitulek, noviny i datum správně |
| navazující otázka se stejným obrázkem | obrázek z cache (cached 328), odpověď správná |
| jiný obrázek na stejném místě | cache se rozejde přesně na první obrazové pozici |
| Responses `input_image`, obrázek ve výstupu nástroje | správně |
| greedy vs. mainline (4 obrazové testy) | rozchod po 67–350 znacích, text-only 24–330 (stejný šum) |

Úskalí: `mtmd_input_text` má pole `text_len`. `llama_model_n_embd_inp()` u modelu jen se slovníkem vrací 0, proto se šířka
řádku bere z `clip.vision.projection_dim`.

Cena encoderu (replay 20 požadavků Codexu, greedy, 262k):
- Encoder na GPU 0: 114 místo 122 expertů na vrstvu na GPU 0, prefill 1283 vs. 1291–1299 t/s (~−1 %).
- Výstupy se změnou rozmístění liší, a to je šum rozmístění. Se stejným rozmístěním (bez encoderu, 900 MiB obsazených
  jiným procesem) jsou výstupy 20/20 shodné a rychlost stejná: 176,8 / 176,8 t/s.
- Encoder na CPU (`HYPER_MMPROJ_CPU=1`) má rozmístění 122, 20/20 shodné a 187,3 vs. 186,8 t/s.
- První běh s encoderem (168,3 t/s) byl zkreslený, šlo o první běh po zastavení jiného serveru.

Qwen3.8-27B (engine.cu) má totéž: unsloth `mmproj-F16.gguf` pro 27B (projektor `qwen3vl_merger`, výstup 5120), tabulku
pozic na všech třech GPU (tensor parallel, `attn_prep`), obrazové řádky i pro MTP, klíče v prompt cache a park span.
Kernel na plnění tabulky (`rpos_fill`) je teď sdílený v kernels.cu.

| 27B | Výsledek |
|---|---|
| replay 20 požadavků Codexu, greedy, tabulka vs. `HYPER_NO_MROPE=1` | 20/20 shodných, 125,1 / 125,4 t/s, prefill 1118 / 1126 |
| 4 obrazové testy vs. mainline (greedy) | 3× text úplně shodný, 1× rozchod po 68 znacích |

Produkční skripty: `VISION=0 ~/run-hyper-fn.sh` (`~/run-hyper.sh`) spustí server bez encoderu, takže GPU 0 zase drží
všechny experty.

## 2026-10-14 – hloubka kontextu: profil reálných přírůstků a dva bitově přesné kernely

Metoda místo dlouhých replayů na živém modelu:
- **Hloubkový scénář** (`~/deep_fn.sh TAG [ENV=…]`): nejhlubší zachycená Codex session (01a12cab…, 9 požadavků). První požadavek
  naplní kontext (154 868 tokenů, studený prefill 112,6 s při 1376 t/s), pak 8 navazujících požadavků (979–3698 nových tokenů v hloubce
  155–170k, 300 greedy tokenů). Zmrazené statistiky expertů, výstupy jako hash na požadavek. Jeden běh ~4,5 min.
- **Profil** stejného scénáře pod nsys (`~/prof_fn.sh`, `nsys launch` + `start/stop` jen kolem přírůstků, `--cuda-event-trace=false`).
  Rozpad času podle kernelů *a podle launch gridu* (gridY/gridZ = počet řádků), takže je vidět zvlášť dávka prefillu, ověření
  1–4 řádků a hustá attention MTP vrstvy (`bench2/kgrid.py`, `nsys_phase.py`, `kdist.py`).
- **Mikrobenchmark** `tools/qsabench.cu`: QSA cesta Flash-Next (indexer + výběr + sparse attention, i hustá attention MTP vrstvy)
  na syntetické cache v libovolné hloubce, reálné wrappery, čas na volání (i se studenou L2) a hashe skóre, výběru a výstupu.
  Změna kernelu se ověří bit po bitu za pár sekund.

Profil (GPU1, ~160k, staré kernely): dávka prefillu indexeru po 64 řádcích **1,7 ms** (`k_qidx_score` čte všech ~40k bloků
klíčů znovu pro každý řádek; ~0,65 s na přírůstek 2,5k tokenů), sparse attention prefillu 6,5–10,7 ms na vrstvu, hustá attention
MTP vrstvy ~0,55 ms na volání. GPU0 (bez attention) mezitím čeká ve spinu LL allreduce.

| Kernel (qsabench, 160k) | starý | nový |
|---|---|---|
| indexer + výběr, 64 řádků | 1739 µs | **407 µs** |
| indexer + výběr, 4 řádky | 127 µs | **85 µs** |
| hustá attention MTP, 1 / 4 řádky | 639 / 1900 µs | **325 / 1041 µs** |
| sparse attention, 4 / 64 řádků | 57,6 / 467 µs | **47,7 / 297 µs** |

- `k_qidx_score_t`: 8 řádků na blok, vlákno načte svůj blok klíčů jednou do registrů a skóruje proti němu všechny řádky (q ze sdílené
  paměti). Stejný výraz ve stejném pořadí → skóre bitově shodná. `HYPER_QIDX_OLD=1` = starý kernel.
- `k_attn_split` (sdílený i s 27B): smyčka přes hlavy bez `break` u `ng` – větvení bránilo kompilátoru prokládat řetězce shuffle
  a exp šesti hlav. Nepoužité hlavy mají q = 0 a nezapisují se. `HYPER_ATTN_OLD=1` = stará varianta.
- Zkoušeno a nepomohlo: předčítání buněk o 3 dopředu (pomalejší), pořadí bloků „split nejpomaleji“ (beze změny), celá dávka 64 řádků
  v jednom bloku indexeru (málo bloků, pomalejší).

| Hloubkový scénář (Flash-Next, 262k, MTP 3) | staré kernely | nové |
|---|---|---|
| prefill přírůstků | 860 t/s (2584 ms / požadavek) | **1145 t/s (1940 ms)** |
| dekódování v hloubce 155–170k | 134,6 t/s | **145,3 t/s** |
| výstupy 8 požadavků | | **8/8 shodných hashů** |

Z produkčního logu: ani jednou (0 z 786 navazujících požadavků) se nevyužila cache až do konce předchozí odpovědi – předchozí odpověď
se přepočítává (Flash průměrně 211 tokenů). Snapshot na konci generování nejde použít (historie končí před posledním tokenem,
při spekulaci se navíc nebere). Na řadě.

### Prefill: upload expertů už neblokuje allreduce (FIFO copy enginu)

Časová osa GPU1 v prefillu přírůstku ukázala, že po MoE každé vrstvy GPU ~7,5 ms stojí. Upload studených expertů vrstvy l+2
(~75 kopií, ~8 ms na x8) se zařadil do H2D fronty hned po MoE vrstvy l. H2D kopie následujícího allreduce (kritická cesta) pak
čekaly ve FIFO copy enginu, až celý upload doběhne. Ve vrstvě tak stály kernely i linka zároveň (PCIe aktivní jen ~37 % prefillu).

Oprava: upload vrstvy se dělí na části (`HYPER4_UP_PARTS`, výchozí 2). Každá část jde do fronty až za H2D kopie nejbližšího
allreduce (událost na compute streamu, `cudaStreamWaitEvent` na copy streamu). Před MoE vrstvy se případný zbytek pošle hned.
Pořadí výpočtu se nemění.

| Hloubkový scénář (prefill přírůstků) | t/s | ms / požadavek |
|---|---|---|
| `HYPER4_UP_PARTS=1` (dosavadní) | 1153 | 1928 |
| **2 části** | **1460** | **1522** |
| 4 části | 1433 | 1551 |

Výstupy 9/9 shodné ve všech třech během. Od začátku dne (staré kernely + celý upload): 860 → 1460 t/s (+70 %) a 2584 → 1522 ms
na požadavek. GLM streamuje stejně, ale jeho prefill je čistě omezený PCIe (upload ~35 ms na vrstvu), takže pořadí ve frontě
tam pomůže méně.

### GDN krok v prefillu: škály tokenu jednou, ne v každém warpu

`k_gdn_step_w` (warp na sloupec stavu, 8 warpů na blok, 128 sloupců na hlavu) počítal pro každý token v každém warpu znovu normy
q/k (dva warp_sumy), softplus/exp decay a sigmoid beta. Ty jsou pro všechny warpy hlavy stejné. XOR motýlek dává ve všech lanech
bitově stejný součet (každý pár sčítá stejné dva operandy), takže je jde předpočítat. `k_gdn_prep` je spočítá jednou na token
a hlavu stejnými výrazy (scratch na zařízení, `gdn_pre_scratch`), krok je jen načte. `HYPER_GDN_NOPRE=1` = staré.

- `tools/gdnbench`: 2048 tokenů, 5–6 k-hlav / 16 v-hlav (podíl jedné GPU): **3,04 → 1,80 ms**, hash výstupu i stavu shodný
  (i pro 300 a 512 tokenů). Explicitní předčítání dalšího tokenu bylo pomalejší (1,91 ms).
- Flash-Next hloubkový scénář: prefill 1462 → **1490 t/s**, 9/9 shodných výstupů. 27B replay: prefill 1130 → **1154 t/s**,
  20/20 shodných (i s předchozím během).
- Allreduce prefillu se stahováním dílů peerů na vlastním streamu (souběžně s vlastním D2H): 1461 → 1469 t/s, v šumu. Allreduce
  čeká hlavně na nejpomalejší GPU, ne na pořadí kopií. Nezařazeno.

### Režie požadavku mimo engine: tokenizace

Rozdíl mezi časem klienta a časem enginu na požadavek v hloubce ~160k byl **610–690 ms** (~16 % požadavku). Měření fází
v `prepare` (řádek `prompt:` v logu): parse 8, převod Responses→chat 19, šablona ~150, tokenizace ~430 ms.
- `tokenize` volal `llama_tokenize` dvakrát (nejdřív kvůli počtu tokenů): jeden průchod do bufferu horní meze → ~215 ms.
- Inkrementálně (`tokenize_prompt`): llama.cpp dělí text na speciálních tokenech a kusy mezi nimi tokenizuje nezávisle,
  takže tokeny předchozího promptu platí až po poslední značku začátku zprávy (`<|im_start|>`, u GLM `<|user|>` …)
  ležící uvnitř společného prefixu textu. Tokenizuje se jen zbytek (~10 KB). `HYPER_TOKCACHE_CHECK=1` porovná s plnou
  tokenizací: 8/8 shodných. `HYPER_TOKCACHE=0` = vždy celý text.

| Hloubkový scénář | mimo engine / požadavek | tokenizace |
|---|---|---|
| původně | 610–690 ms | ~430 ms |
| jeden průchod | 400–420 ms | ~230 ms |
| **inkrementálně** | **180–195 ms** | **2–8 ms** |

Výstupy 9/9 shodné. Zbývá šablona (~150 ms, minja v libcommon).

### Chyba v předvýpočtu GDN (opraveno) a poučení

Scratch `k_gdn_prep` byl jeden na zařízení. 27B ale v prefillu počítá dvě mikrodávky na dvou streamech: mikrodávka A mohla
v další vrstvě přepsat předpočítané škály, zatímco krok mikrodávky B v předchozí vrstvě je ještě četl. Na hlubokém promptu
(155k, chunkování prompt cache jako v serveru) z toho byl poškozený rekurentní stav a výstup „!!!!“. Projevilo se to jen
při určitém časování: replay 27B (mělký) prošel 20/20 a výstup rozbilo teprve jiné pořadí bloků attention nebo LM hlava
přes všechny řádky. Oprava: scratch na stream.

Poučení pro metodu: změny sdílené s 27B (kernels.cu) ověřovat i hloubkovým scénářem 27B (`~/deep_27.sh`), nejen mělkým replayem.
Pro bisekci na úrovni enginu: `HYPER_DUMP_TOKENS` uloží prompt serveru jako referenční soubor (jen tokeny) a `hyper gen` ho
s `HYPER_PCACHE=1` předpočítá se stejným chunkováním jako server (`HYPER_GEN_PRINT=1` vypíše tokeny).

### Hustá dekódovací attention: řádky nejrychleji v gridu (27B, MTP vrstva Flash)

Při ověření 2–4 řádků hustá attention (27B ve všech vrstvách attention, Flash v MTP vrstvě) spouštěla bloky v pořadí
(hlava, split, řádek). Řádky 2–4 tak běžely v další vlně a stejné úseky KV cache četly znovu z DRAM. Pořadí „řádky
nejrychleji“ (`nt_x`) pustí bloky všech řádků jednoho úseku současně a cache se čte jednou. Práce bloku je stejná, výsledek
bitově shodný (qsabench, tvar 27B, shodné hashe):

| 27B, 4 řádky | dosud | řádky nejrychleji |
|---|---|---|
| 60k | 455 µs | 401 µs |
| 120k | 988 µs | 774 µs |
| 200k | 1794 µs | 1290 µs |

`hyper gen` na hlubokém promptu 27B (155k, chunkování jako server): spekulativní dekódování 85,2 → 98,5 t/s, výstup shodný
s obyčejným. Výchozí pro hustou attention s 2–4 řádky (`HYPER_ATTN_ROWS=0/1` vynutí jedno pořadí pro všechny případy).
(Je v commitu 4b38e92 spolu s opravou scratche GDN.)

### Vzorkování přes celý řádek

Když top_k nestačí (vypnutý top_k a široký nucleus, např. výchozí nastavení GLM), host projde celý řádek logitů. Biny
podle log váhy byly vektory vektorů. Teď je to stabilní counting sort podle binu (stejné pořadí uvnitř binu): 2,53 → 1,71 ms
na řádek o 248k. `tools/samptest_eq` (144 kombinací parametrů proti předchozímu kódu): tokeny, bity pravděpodobností
i Gumbelovy tahy shodné.

### Incident 2026-10-14 01:11 UTC: soft lockup hostu (restart nutný)

Kernel log předchozího bootu:
1. 01:10:18 skončil poslední požadavek běhu `e27c_base` (27B). Skript poslal SIGINT, ale **proces spuštěný na pozadí (`&`)
   v neinteraktivním skriptu má SIGINT ignorovaný**, takže se nic nestalo. Po 60 s (01:11:18) dostal SIGKILL uprostřed
   klidového stavu a jádro ho uklízelo samo.
2. Další běh čekal jen na volnou VRAM, ne na konec procesu, a začal hned načítat **stejný GGUF** (27B).
3. 01:11:20 `BUG: Bad page state in process hyper-server ... aops:ext4_da_aops ino:564e87 (GGUF 27B) ... page dumped because:
   non-NULL mapping`, ve starém procesu na cestě `do_exit → … → uvm_release [nvidia_uvm] → mmput → exit_mmap → release_pages`.
   Uvolnila se stránka, která ještě byla v page cache (refcount klesl na 0 o jednu referenci dřív).
4. 01:12:02 `watchdog: BUG: soft lockup - CPU#0 stuck` v novém procesu: `filemap_fault → filemap_get_entry → xas_load` na té stránce,
   donekonečna. Za ním systemd a `cgroup_release` ve stavu D, SSH relace se otevíraly minuty. Pomohl jen restart.

Uživatelský kód „Bad page state“ sám způsobit nemůže. Jde o chybu jádra nebo ovladače (6.8.0-137, nvidia-open 580.173.02, HMM
zapnuté: `uvm_disable_hmm=N`), kterou spustil souběh dvou věcí: úklidu zabitého procesu (SIGKILL, 2 s před hlášením) a načítání
stejného souboru druhým. Shodu v issue trackeru nvidia jsem nenašel. Opatření:
- benchmarkové skripty (`~/bench2/guard.sh`): ukončení přes SIGTERM a čekání na úplný konec procesu (`wait`, `pgrep`), žádný SIGKILL;
  před startem a po každém běhu kontrola `dmesg` na „Bad page state / soft lockup“, řetěz běhů se pak zastaví;
- `hyper-server` zachytí SIGINT/SIGTERM vláknem se `sigwait` (blokované ve všech vláknech, funguje i pro úlohu na pozadí),
  zastaví HTTP server a engine se uklidí v destruktorech (GPU paměť, pinned buffery, mapování GGUF); druhý signál = okamžitý konec.

### MTP jen pro poslední řádek, LM hlava 27B, flash attention s double-bufferingem, upload GLM

Hloubkový scénář (155–170k), každý pár ve stejné relaci; výstupy (hash na požadavek) ve všech párech shodné, u MTP změn
i míra přijetí draftů na každý požadavek (drafty bitově stejné):

| Změna | před | po |
|---|---|---|
| Flash: MTP vrstva při ověření i u ocasu promptu počítá attention, wo a allreduce jen pro poslední řádek (ostatní jen K/V) | 146,3 t/s dekódování, 1400 ms/požadavek | **148,3 t/s, 1393 ms** |
| 27B: MTP při ověření jen poslední řádek; v prefillu chunk, který není poslední, jen K/V; LM hlava prefillu jen pro poslední řádek | 642 t/s prefill, 68,5 t/s | 649 / 69,2 (bez FA) |
| 27B: prefill flash attention s double-bufferingem K/V (`HYPER_FA_SINGLE=1` = staré), dohromady s předchozím řádkem | **639 t/s, 68,2 t/s** | **745 t/s (+17 %), 69,0 t/s** |
| GLM: upload expertů po polovinách až za stahováním allreduce (`HYPER5_UP_PARTS=0` = staré) | 502 t/s (4053 ms/požadavek) | **608 t/s (+21 %, 3346 ms)** |

- qsabench (tvar 27B, 64 řádků): flash attention 2522 → 2067 µs na 20k, 17,1 → 14,0 ms na 155k, hash shodný.
- V hloubce flash attention prefillu běží jen 32–64 bloků (řádkové dlaždice × 8 hlav) na 82 SM. Split přes klíče
  (`HYPER_FA_SPLIT=1`, sloučení přes `k_attn_combine`) zkrátí 64 řádků na 155k z 13,9 na **2,07 ms**. Není bitově shodný
  (P se ve fp16 zaokrouhluje vůči maximu svého úseku: rozdíl max 4,7e-6, relativně 3,6e-4), proto zatím opt-in, dokud
  neprojde KL proti referenci.
- Bariéra nahrávacích vláken se dá přerušit (výjimka v jednom vlákně uvolní ostatní místo věčného čekání).

### Split flash attention přes klíče: přesnost proti fp64, výchozí zapnutý

KL proti llama.cpp (27B, hluboký prompt 155k, posledních 256 pozic; `hyper checkn`, chunky 512) se splitem mírně roste:
0,004025 → 0,004162. llama.cpp ale není přesná reference. Jeho vlastní šum (ub512 proti ub256) je na 155k 0,00045, kdežto
hyper je od něj v hloubce 9× dál (na 12k jen 0,00094 proti šumu 0,00069). Proto přímé měření proti přesné attention:
`qsabench` s `QSA_REF=1` počítá stejné řádky v fp64 (q·scale nezaokrouhlené, K/V z fp16 cache), 64 řádků, tvar 27B:

| případ | jeden průchod: rel L2 / max | split: rel L2 / max |
|---|---|---|
| 20k klíčů | 2,90e-4 / 1,34e-5 | **2,84e-4 / 1,22e-5** |
| 155k klíčů | 3,80e-4 / 5,71e-6 | **2,87e-4 / 4,29e-6** |
| 20k, ostřejší attention (`QSA_QMUL=3`) | 4,85e-4 / 8,2e-4 | **4,80e-4 / 7,8e-4** |
| 155k, ostřejší | 5,40e-4 / 4,4e-4 | **5,15e-4 / 4,7e-4** |

Na skutečných datech je rozdíl větší. `HYPER_FA_DUMP` vypíše vstupy volání `attn_prefill` (27B, GPU 1, 256 řádků na
150k, první / prostřední / poslední attention vrstva) a `qsabench QSA_LOAD=` je přepočítá:

| vrstva | jeden průchod: rel L2 / max | split: rel L2 / max |
|---|---|---|
| první | 4,73e-4 / 7,0e-4 | **1,56e-4 / 1,8e-4** |
| prostřední | 2,46e-4 / 3,5e-3 | **2,06e-4 / 3,5e-3** |
| poslední | 2,20e-4 / 1,5e-2 | **8,3e-5 / 4,9e-3** |

Skutečná attention má sinky s velkým skóre. V jednom průchodu je běžící maximum od první dlaždice skóre sinku, takže P
ostatních klíčů padá ve fp16 do subnormálů a ztrácí relativní přesnost. Úseky splitu bez sinku mají vlastní maximum a do
fp32 se škálují až při sloučení. Split je tedy 2–3× přesnější a 6,7–7,2× rychlejší (155k, 64 řádků: 13,9 → 2,07 ms).
Je teď výchozí, `HYPER_FA_SPLIT=0` vrátí jeden průchod. Flash-Next ho nepoužívá (flash attention má jen pro mělké chunky
pod sparse režimem), týká se 27B.

Hloubkový scénář 27B (155–170k, 8 přírůstků po 300 tokenech), oba běhy čerstvé: prefill **754 → 957 t/s (+27 %)**, 2946 →
2322 ms na požadavek, dekódování 68,5 / 68,4 t/s. Hashe výstupů: 8 z 9 shodných, jeden požadavek se liší (greedy na
téměř shodě).

Proti fp32 referenci (llama.cpp `REF_KV=f32 REF_FA=0 REF_UBATCH=128`, 155k, posledních 256 pozic):

| | KL průměr / max | top-1 |
|---|---|---|
| llama.cpp výchozí (FA, f16 KV, ub512) | 0,004239 / 0,547 | 98,44 % |
| llama.cpp ub256 | 0,003306 / 0,288 | 97,66 % |
| hyper, split | **0,000534** / 0,026 | 99,22 % |
| hyper, jeden průchod | **0,000498** / 0,018 | 99,22 % |

Hyper je v hloubce 8× blíž přesnému výsledku než llama.cpp. Rozdíl „hyper vs llama.cpp 0,004“ byla chyba llama.cpp.
Rozdíl mezi splitem a jedním průchodem proti této referenci (0,000534 vs 0,000498) je šum: chybu tu určuje zaokrouhlení
K/V do f16, které mají obě varianty stejné, a samotnou attention rozhoduje fp64 test výše.

Než se ukázalo, že chyba je na straně llama.cpp, padlo podezření na RoPE: hyper počítá úhel a sin/cos v double, llama.cpp
ve fp32 s `-use_fast_math` (úhel 155k rad má ve fp32 krok 0,016). S aritmetikou llama.cpp v hyperu (dočasný přepínač) byl
ale KL proti llama.cpp na 155k 0,004090 místo 0,004025 a na 12k 0,000946 místo 0,000942. RoPE to tedy není.

Zkoušeno bez zisku: Q fragmenty ve flash attention načtené jednou do registrů a přeskočené přeškálování O při nezměněném
maximu. Výsledek je bitově shodný, rychlost stejná (255 registrů bez spillu). Na SM to běží asi na 80 % tensor peaku
(m16n8k16 s fp32 akumulací), rezerva je jen v paralelismu, a ten řeší split.

### Snapshot na konci odpovědi (převzetí předchozí odpovědi)

Produkční log (911 požadavků Flash-Next, 819 navazujících dvojic): každý tah začínal prefill na startu poslední zprávy
předchozího promptu (`cached = P - 5`). Předchozí odpověď se tak počítala znovu a tvořila **20,7 % všech znovu
prefillovaných tokenů** (166k z 802k). Znovu vykreslený prompt se s vygenerovanými tokeny shoduje v celé odpovědi u ~78 %
dvojic. Uvnitř odpovědi se rozchází u 61 dvojic (jiná tokenizace, vypuštěné uvažování), u 117 jde o jinou konverzaci.

Proč se snapshot na konci odpovědi nikdy nepoužil: při spekulativním dekódování přijde stop (EOS, max_tokens, odpojení
klienta) až po forwardu, který už obsahuje i drafty za koncem výstupu. Stav tedy nebyl použitelný (`state_ok = false`)
a historie končila před posledním tokenem.

Nově, ve všech třech enginech:
- token na pozici p se odešle před svým forwardem, takže stop na něm nechá stav na p (a ušetří zbytečný forward);
- stop uvnitř přijatých draftů vrátí stav na řádek před posledním tokenem výstupu (stejný restore jako při odmítnutém
  draftu) a dopíše jejich řádky do MTP cache;
- snapshot na pozici posledního tokenu, historie ho obsahuje (MTP řádek p − 1 byl spočítán s ním). Další požadavek, který
  odpověď zopakuje, pokračuje od jejího konce.

Ověření (`hyper4 cachekl`, Flash-Next, pevné rozmístění expertů `HYPER4_ADAPT=0`, odpovědi 40–44 tokenů, aby stop padl na
různé řádky): převzetí po odpovědi vygenerované s drafty je **bitově shodné** s převzetím po odpovědi bez draftů. Agentní
smyčka na serveru (6 tahů s tool cally): každý tah pokračuje přesně na P + délka odpovědi, dosud na P − 5.

Numerika: K/V a stav odpovědi zůstanou takové, jak je spočítalo dekódování, místo nového prefillu. U Flash-Next a GLM běží
CPU experti při dekódování s aktivacemi kvantovanými na Q8_K, kdežto prefill je streamuje na GPU (fp16). Proti „vše
prefillované“ je proto KL dalších tokenů ~1,2e-2. Stejný řád ale dává i samotné jiné rozdělení na chunky (malý chunk
počítá CPU experty na CPU): 0,9–2,2e-2. Podle pravidla o přesnosti je převzetí odpovědi u MoE modelů vypnuté a zapíná se
`HYPER_ANSWER_REUSE=1`.

U 27B (bez CPU expertů) jsou obě cesty stejně přesné. Proti fp32 referenci na 12k (posledních 1024 pozic) vychází
prefill po 512 KL 0,000785, dekódování po jednom tokenu 0,000785 a ověřovací řádky po dvou 0,000785. Převzetí je tu proto
zapnuté. Agentní smyčka na 27B: každý tah navazuje na konec odpovědi, i když byla uříznutá na max_tokens (600 tokenů,
další požadavek prefilloval jen 30 tokenů).

Bez převzetí (výchozí u Flash/GLM) je generování stejné jako dřív. Agentní smyčka Flash s pevným rozmístěním expertů
(`HYPER4_ADAPT=0`), starý a nový build: všech 6 tahů bitově shodných. S adaptivním rozmístěním se výstupy od tahu 1 liší.
Nový kód vynechá zbytečný forward za EOS, takže statistiky routování (a s nimi rozmístění expertů mezi CPU a GPU) se posunou
o jeden krok.

Měřeno, nepokračuji: vykreslení chat šablony stojí v hloubce většinu režie mimo engine (155k: parse 7, převod 17,
šablona 138, tokenizace 1–6 ms; 18k: 7 ms, 3k: 2 ms). Je to jinja engine llama.cpp nad ~600 KB textu, mírně nadlineárně.

### Dekódovací attention na tensor cores (27B v hloubce)

Profil dekódování 27B na 155–170k (nsys, `prof_27.sh`): krok ~42 ms (přijetí 1,65–2,1 draftu, tedy ověřování 4 řádků),
z toho **hustá attention 4 řádků ~17 ms** (`k_attn_split`, 1,06 ms na vrstvu × 16). Jeden řádek (MTP) trvá 0,40 ms, což je
šířka pásma DRAM pro 327 MB K/V na vrstvu a GPU. Čtyři řádky stojí 2,6× víc. Nejde o paměť: každá trojice (řádek, hlava,
pozice) stojí na CUDA jádrech ~40 warp instrukcí (8 FMA, redukce přes 5 shuffle a 5 add, 2 exp, 16 operací v P·V).

Zkoušeno a zahozeno (bitově shodné, ale pomalé):
- všechny řádky v jednom bloku (warpy řádků čtou tytéž pozice, L1 je sdílí): GPU1 1041 → 820 µs, ale GPU0/2 (6+2 hlav na KV
  hlavu) 1025 → 1222 µs, protože bloky po 4 hlavách čtou KV0 dvakrát;
- totéž s dlaždicemi K/V ve sdílené paměti (jedno čtení na blok): pomalejší než přes L1, protože limitují instrukce.

`k_attn_dec_tc` (výchozí, `HYPER_ATTN_TC=0` = fp32 kernel): MMA řádky jsou dvojice (token, q hlava) jedné KV hlavy (16, nebo 32 v 8warpovém bloku).
q·scale a p se dělí na fp16 hi + lo, k a v jsou ve fp16 přesně, takže součiny jsou fp32-přesné. Tensor cores ale akumulují
fp32 s ořezáním, proto mají řetězce krátké: skóre ve čtyřech akumulátorech po 64 dimenzích plus lo zvlášť, P·V každé
dlaždice do vlastního akumulátoru, a teprve pak se sčítá ve fp32 (bez toho byla chyba proti fp64 8,8e-6, s tím 5,6e-7).
Dlaždice 32 klíčů leží na absolutních pozicích a každý řádek má svůj úsek. Výsledek řádku tak nezávisí na ostatních řádcích
bloku: řádek samotný = řádek ve 4řádkovém volání, bitově (qsabench `QSA_ROWID`), takže spekulativní dekódování dál přesně
odpovídá obyčejnému. Dvojnásobek úseků proti fp32 kernelu; 16řádkový blok má 2 bloky na SM (Q fragmenty v registrech, jeden
buffer), 32řádkový 1 blok s double bufferem.

qsabench, 160k klíčů (fp32 kernel → tensor cores), chyba proti fp64 (rel L2):

| tvar | 1 řádek | 4 řádky | chyba fp32 / tc |
|---|---|---|---|
| 27B GPU1 (4+4 hlav) | 405 → 433 µs | **1041 → 455 µs** | 5,0e-7 / 5,6e-7 |
| 27B GPU0/2 (6+2) | 404 → 434 µs | **1031 → 535 µs** | 5,0e-7 / 5,6e-7 |
| Flash-Next MTP (12 hlav) | **321 → 235 µs** | 1035 → 643 µs | – / 5,6e-7 |
| 27B, 20k | 69 → 94 µs | 151 → 119 µs | 2,9e-7 / 5,3e-7 |

End to end:
- 27B proti fp32 referenci (12k, 1024 pozic): dekódování KL 0,000785 → **0,000784** (max 0,2074 → 0,2008), ověřovací řádky
  po dvou stejně;
- `hyper gen` na 155k: spekulativní **57,5 → 74,0 t/s**, hlavní forward 36,4 → 27,3 ms na krok, obyčejné 45,1 → 44,5 t/s;
  obě sekvence shodné s obyčejným dekódováním (200/200);
- hloubkový scénář 27B (server, 155–170k, oba běhy čerstvé): dekódování **67,9 → 89,0 t/s (+31 %)**, prefill 945 / 954 t/s.
  Hashe 8 z 9 shodné, míra přijetí na požadavek shodná až na ten jeden (týž požadavek, který se lišil i u splitu: greedy na
  téměř shodě).
- Flash-Next (attention na tensor cores jen v MTP vrstvě), hloubkový scénář, oba běhy čerstvé: dekódování 146,4 → 148,7 t/s,
  prefill 1635 / 1629 t/s. Všech 9 hashů shodných, míra přijetí draftů na každý požadavek shodná.

## 2026-10-17 – Flash-Next: router v bf16

- Router + gate sdíleného experta (`ffn_gate_inp`, `ffn_gate_inp_shexp`) jsou v GGUF F32, ale všechny hodnoty jsou
  bf16-exaktní (48 vrstev i MTP). Ukládají se jako bf16 (kontrola při načtení po vrstvách, jinak fp32 a hláška),
  dekódování: GEMV se stejnými součiny a pořadím součtu jako fp32, prefill: rozšíření do fp32 + stejný sgemm.
  `HYPER4_ROUTER32=1` = původní cesta.
- Kernel proti fp32 na skutečných vahách: 12,6 M výstupů, 0 bitových rozdílů. Deep A/B (155k + 8 přírůstků):
  se stejným rozmístěním expertů (bf16 v alokaci velikosti fp32) 9/9 hashů identických. Bez toho uvolní ~125 MB
  na GPU → 122/93/90 místo 122/92/89 expertů na vrstvu, výstup se pak liší jen tím rozmístěním.
- Rychlost v šumu (decode 134,9 → 135,0–135,9 t/s, prefill beze změny); odhad byl ~1 % (router ~0,3 ms z ~15 ms kroku).
- GLM-5.3-Flash stejně (engine5, `HYPER5_ROUTER32=1` = fp32): router 42 MoE vrstev bf16-exaktní, kernel 6,05 M výstupů
  bez rozdílu, deep A/B (155k + 8 přírůstků): základ / bf16 ve fp32 alokaci / bf16 9/9 hashů identických; rozmístění
  beze změny (28 29 28 na vrstvu, ~100 MB nestačí na expert), rychlost v šumu (decode 21,8 / 21,7 / 21,6 t/s).

## 2026-10-18 – MiMo-V2.6-Flash (mimo2): engine6, první verze

Model: 309 B / 15 B aktivních, 48 vrstev (39 SWA 128 s attention sinky, 9 globálních), GQA (q/k 192, v 128, RoPE 64),
vrstva 0 hustá, 47 MoE (256 expertů, top 8, sigmoid + bias, bez sdíleného), experti MXFP4 (`ggml-org/MiMo-V2.6-Flash-MOPD-GGUF`).

- MXFP4 v GGUF i v GPU MoE kernelech (deq8, prefill GEMM přes generickou cestu); `deqtest` bit-identický s ggml.
- `kernels6`: q/k/v RoPE (double) + zápis do cache (globální vrstvy plné, SWA kruh 4096 řádků), GQA flash-decoding
  (fp32 skóre a softmax, sinky, okno, split přes klíče + combine). Attention rozdělená po kv hlavách, u globálních vrstev
  rotace dvojitého podílu mezi GPU.
- Zbytek převzat z engine5 (CPU / GPU experti, adaptivní rozmístění, elastické sloty, streaming při prefillu).
- Přesnost (ref z mainline llama.cpp, experti na CPU): 2779 tokenů KL 0,0167 top1 93,9 %, PPL 4,952 vs 4,945;
  šumová podlaha llama.cpp proti sobě (ubatch 512) KL 0,0163 top1 94,6 %. Prefill 1024 + decode KL 0,0149.
- Rychlost: decode 25 t/s (studené rozmístění) / 36 t/s po 8k prefillu; prefill 8k 1020–1120 t/s (opakovaný text).
  llama.cpp (15 vrstev expertů na GPU): pp2048 120 t/s, tg 16,5 t/s.
- Chybí: MTP (3 NextN vrstvy v sidecaru), vision, TC prefill attention, snapshoty / parkování (prompt cache jen navazuje
  na poslední konverzaci, dokud SWA kruh drží okno).
