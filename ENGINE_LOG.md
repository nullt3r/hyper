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

