# To-do:

- Licensing?
- Adjust embedding flow to be more accurate. Maybe using grammar (Kronk); Maybe one embed for expression (2-5 words)?
- Update the performance tests for the embedding;
- Update agents and readme with new flow and kronk usage

- Check "per-request allocation" error with vulkan (and maybe other processors) -> T-019dfff7-3572-72cc-bede-689295bad366
  - last kronk version before the bug is v1.24.8, with Llama.cpp pinned to version b9247 for compatibility.
  - Possible fix committed in local branch "vulkan_bug" in both kronk AND locallens repos.
  - Until the bug is fixed, will move on with processor=CPU only.
  - Now running with kronk v1.28.3 and llama.cpp v9750, LocalLens managed to describe 5 images before breaking with the vulkan error;

- Look for suggested model config values at the model provider sites, or test other models > https://chatgpt.com/c/6a04c226-7d9c-83e9-86ae-7f8743cea1ec ;

- Search for "TODO" along the codebase;

- Review and correct indexing/searching recursiveness code; Add recursiveness test too;
- Add search by name to the folders browser;

# Future features

- Option to the user describe a common context of a set of images before the description -not the search- to ask the vision model to use more contextualized vocabulary and to pay more attention in what makes that image unique among the others in the same context. e.g.: search for a specific moment of the celebration among many wedding photos;
- Adapt prompt and search to different idioms;
- Face recognition;
- Configurable detail vs speed;
- Run the tool inside local explorer browsers; As a plugin?
- Run the tool inside photographers apps (adobe,etc); As a plugin?
- Run it in mobile? -> iPhone Photos;
- Configuration adaptable to the user hardware constraints;
- LocalEars to audio files;
- LocalMovs to video files;
- Concurrent improvements;


# Issues

- How to display best scored images? A list, a link, a preview screen?
- Similar tools[https://www.google.com/search?q=search+my+photos+offline+photographer&sca_esv=6cda7a6ffd36fe53&rlz=1C1CHZN_pt-brBR949BR949&sxsrf=ANbL-n6odB5tPvr0NwCsWpLzPjT9f_4tAg%3A1769308487234&ei=R4F1afmGDoXc5OUP9p3W-AY&oq=search+my+photos+offline+photographer&gs_lp=Egxnd3Mtd2l6LXNlcnAiJHNlYXJjaCBteSBwaG90b3Mgb2ZmbGluZSBwaG90b2dyYXBlcioCCAAyBxAhGKABGAoyBxAhGKABGApIpJcBUMgLWLyEAXAGeACQAQCYAeABoAHxGKoBBjAuMTcuMrgBA8gBAPgBAZgCGaAC-xnCAggQABiwAxjvBcICBRAAGO8FwgIFECEYoAHCAgUQIRifBZgDAIgGAZAGBJIHBjYuMTcuMqAH6FeyBwYwLjE3LjK4B-EZwgcHMC4xMC4xNcgHToAIAA&sclient=gws-wiz-serp]

# Performance and configs

- Goals:      | accurate image description | fast description process | low hardware demand |
              |-----------------------------------------------------------------------------|
    Needs     | outTok > 200               | small maxSize            | small model and cfgs|
  Constrains  | min viable maxSize         | outToks > 200            | KV cache = Q8_0     |

### Considerations:

- Many outToks(≈250) are needed to get good descriptions(IT IS A MUST), but there is no need for more than maxTok=300;
- But actually verbose descriptions alone cannot guarantee good vectors. Need to explore more what kind of descriptions can generate good vectors, and then try to enforce them through prompt or using grammar. 
- I found before that format restrictive prompts were preventing the model to "talk", but I need to ensure that the cause was not bad configuration.

- Temperature is set to 0.1 because we want: "same description to same image" every time;
- Can we have more predictable descriptions using a system prompt? YES, THEY SEEM TO BE MORE CONCISE. Can be re-evaluated after search tests.
- Can we achieve more detailed descriptions sending separated prompts for each category(people characteristics, actions, etc)? Need to see if message caching works for images in the newer Kronk versions.

- Recent tests show similar inference times for maxSide= 128, 256 and 384px : ~16 seconds/image. 512px increased this time a little ~20sec
- maxSide=512px provide more precise descriptions (text, context, small details). 384px it is the minimum from now! I also set Quality image enconding to 100%.
- A test with this prompt - "Did you load an image? Just say yes or no." - shows the time to "load" the image: 62px - 2,2s | 384px - 11,3s - Maybe there is a metric for that.
- Now that is a metric: ttft - time to first token - but now that is not the bottleneck of inference time anymore. The description appropriate for a good embedding will be the most important thing.

- Big image maxSize(>384px) and restrictive/demanding prompts can cause model to fail responses when running with limited hardware, as mine. These two factors showed up to play a higher role in performance than model configs(considering viable values, at least) as cxtWindow, Nbatch and NUbatch.
- Thanks to changes made to allow track yzma mtmd.HelperEvalChunks return codes, it is possible to see that big image maxSize(768, sometimes even 512) can cause a prefill error, KV_cache_full, totally corrupting the response;
- The restrictive/demanding prompts don't cause KV_cache_full error, but increase a lot the time/tokens and cause truncated(few tokens) response.
- The hallucinated responses (randomly repeated words forever) could not be precisely associated with some test case yet. More often when KVcache Q4_0 than Q8_0; >> Solved using dry(dont repeat yourself)_multiplier=2.5, a per-request sampling parameter
- The model size itself and its quantization value, as the projection file attached, impacts the hardware demand. Currently using small ones, with good behavior of descriptions achieved. Can try even smaller configs later.
- I assumed KV cache precision should track model weight precision — it doesn't. They're independent. The K cache holds attention keys that get multiplied against every query head in the group; with Qwen2-VL's 7:1 GQA ratio, Q4_0 quantization noise in the K cache gets amplified across all 7 sharing heads, destroying attention patterns. Q8_0 is the minimum safe level for K cache, especially on small Qwen models. V cache is more tolerant (could even go Q4_0), but Q8_0/Q8_0 is the pragmatic safe default.

- Kronk now has auto-detect GPU. If the machine has a GPU, kronk will use it automatically. BUT, if a machine has only an older iGPU and low memory (< 16GB), to use the auto-selected Vulkan backend is worse than use CPU-only as processor. So in this case is better set the env KRONK_PROCESSOR=cpu; Maybe we can set some threshold values to choose Vulkan instead of CPU.

- Tokens/sec seems not to be related with config or image size. It remains approx. 14-15 tok/s. Maybe it is related just with hardware? No, it varies across same hardware runs, maybe related with memory demand on the hardware at inference time.

- inToks is always the same, despite model config and image maxSize? Maybe it is related with the prompt? YES, with the prompt AND image size.

- Until now, we reached avg description times of 24sec/img(CPU only, 8GB RAM) and 7sec/img(6GB VRAM GPU, 32GB RAM) with good descriptions, but bad search similarity;



### Some performance test outputs:
more recent at top

```
====================================================================================================
HARDWARE
====================================================================================================
GPU:         none
System RAM:  3316 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [384 512]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | RAM Use%
----------------------------------------------------------------------------------------------------
app        |   8192 |   2048 |    2048 |     Q8_0 |     Q8_0 |      940.4 |      112.0 |    28.4%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
app      @384: avgTime  18103ms | ttft   3603ms | avgTimeVar  30% | inTok  197 | outTok 184 | Tok/s 13.5
app      @512: avgTime  21274ms | ttft   5847ms | avgTimeVar  36% | inTok  285 | outTok 198 | Tok/s 13.6
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
app      |  384 | forest.jpg      |       19625 |      3269 |     16356 |      24% |    181 |    210 |  13.6 |  100% |   0 runs
app      |  384 | graduate.jpg    |       12544 |      3565 |      8978 |      34% |    195 |    110 |  13.6 |  100% |   0 runs
app      |  384 | parrot.jpg      |       13029 |      3415 |      9614 |      31% |    195 |    120 |  13.8 |  100% |   0 runs
app      |  384 | search_images.gif |       23438 |      3972 |     19466 |      42% |    209 |    252 |  13.6 |  100% |   0 runs
app      |  384 | setup_panel-0.jpeg |       24948 |      3815 |     21133 |      10% |    209 |    272 |  13.4 |  100% |   0 runs
app      |  384 | wedding.jpg     |       15036 |      3580 |     11456 |      38% |    195 |    140 |  13.3 |  100% |   0 runs
app      |  512 | forest.jpg      |       16294 |      4814 |     11480 |      29% |    249 |    146 |  13.7 |  100% |   0 runs
app      |  512 | graduate.jpg    |       16822 |      5889 |     10934 |      47% |    285 |    138 |  13.7 |  100% |   0 runs
app      |  512 | parrot.jpg      |       21518 |      5985 |     15533 |      52% |    285 |    199 |  13.6 |  100% |   0 runs
app      |  512 | search_images.gif |       29236 |      6280 |     22956 |      28% |    303 |    300 |  13.6 |  100% |   0 runs
app      |  512 | setup_panel-0.jpeg |       26133 |      6291 |     19842 |      32% |    303 |    255 |  13.4 |  100% |   0 runs
app      |  512 | wedding.jpg     |       17639 |      5825 |     11814 |      25% |    285 |    150 |  13.7 |  100% |   0 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           24
Runs with pressure:   0 (0.0%)
  - Slow token:       0
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    1644 MB
Max page faults:      195101
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260610_202506.csv
Individual results saved to: results\vision\performVis_ind_20260610_202506.csv
--- PASS: TestVisionPerformance (544.34s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  544.568s



====================================================================================================
HARDWARE
====================================================================================================
GPU:         none
System RAM:  2451 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [128 256 384]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | RAM Use%
----------------------------------------------------------------------------------------------------
app        |   8192 |   2048 |    2048 |     Q8_0 |     Q8_0 |      940.4 |      112.0 |    38.4%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
app      @128: avgTime  16339ms | ttft    964ms | avgTimeVar  17% | inTok   84 | outTok 200 | Tok/s 14.0
app      @256: avgTime  15573ms | ttft   1738ms | avgTimeVar  20% | inTok  122 | outTok 181 | Tok/s 13.9
app      @384: avgTime  16520ms | ttft   3457ms | avgTimeVar  37% | inTok  197 | outTok 169 | Tok/s 13.9
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
app      |  128 | forest.jpg      |       20988 |      1017 |     19970 |      10% |     84 |    248 |  13.9 |  100% |   0 runs
app      |  128 | graduate.jpg    |       12278 |       959 |     11319 |       1% |     84 |    148 |  14.1 |  100% |   0 runs
app      |  128 | parrot.jpg      |       15482 |       963 |     14519 |      40% |     84 |    192 |  14.0 |  100% |   0 runs
app      |  128 | search_images.gif |       21782 |       930 |     20852 |      13% |     84 |    279 |  13.9 |  100% |   0 runs
app      |  128 | setup_panel-0.jpeg |       12141 |       972 |     11169 |       9% |     84 |    142 |  13.8 |  100% |   0 runs
app      |  128 | wedding.jpg     |       15362 |       945 |     14417 |      30% |     84 |    190 |  14.0 |  100% |   0 runs
app      |  256 | forest.jpg      |       11066 |      1581 |      9486 |       6% |    114 |    116 |  13.5 |  100% |   0 runs
app      |  256 | graduate.jpg    |       13434 |      1784 |     11650 |      28% |    123 |    150 |  13.9 |  100% |   0 runs
app      |  256 | parrot.jpg      |       11841 |      1813 |     10028 |       9% |    123 |    128 |  13.9 |  100% |   0 runs
app      |  256 | search_images.gif |       20476 |      1743 |     18734 |      33% |    123 |    250 |  14.0 |  100% |   0 runs
app      |  256 | setup_panel-0.jpeg |       19742 |      1762 |     17980 |      23% |    123 |    239 |  13.9 |  100% |   0 runs
app      |  256 | wedding.jpg     |       16878 |      1748 |     15130 |      20% |    123 |    200 |  14.0 |  100% |   0 runs
app      |  384 | forest.jpg      |       13013 |      3058 |      9955 |      26% |    181 |    127 |  13.9 |  100% |   0 runs
app      |  384 | graduate.jpg    |       11850 |      3427 |      8423 |      44% |    195 |    105 |  13.9 |  100% |   0 runs
app      |  384 | parrot.jpg      |       13598 |      3403 |     10195 |      21% |    195 |    130 |  13.9 |  100% |   0 runs
app      |  384 | search_images.gif |       22646 |      3736 |     18910 |      37% |    209 |    249 |  13.8 |  100% |   0 runs
app      |  384 | setup_panel-0.jpeg |       22524 |      3720 |     18804 |      44% |    209 |    248 |  13.8 |  100% |   0 runs
app      |  384 | wedding.jpg     |       15487 |      3401 |     12086 |      50% |    195 |    156 |  13.9 |  100% |   0 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           36
Runs with pressure:   0 (0.0%)
  - Slow token:       0
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    1612 MB
Max page faults:      187783
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260610_181047.csv
Individual results saved to: results\vision\performVis_ind_20260610_181047.csv
--- PASS: TestVisionPerformance (691.97s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  692.194s



====================================================================================================
HARDWARE
====================================================================================================
GPU:         none
System RAM:  4174 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64 256]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | RAM Use%
----------------------------------------------------------------------------------------------------
app        |   8192 |   2048 |    1024 |     Q8_0 |     Q8_0 |      934.7 |      112.0 |    22.4%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
app      @ 64: avgTime  26000ms | ttft   1388ms | avgTimeVar  27% | inTok   70 | outTok 238 | Tok/s 10.3
app      @256: avgTime  20186ms | ttft   3176ms | avgTimeVar  32% | inTok  110 | outTok 156 | Tok/s  9.6
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
app      |   64 | forest.jpg      |       34072 |      1597 |     32475 |       0% |     70 |    292 |   9.2 |  100% |   1 runs
app      |   64 | graduate.jpg    |       23173 |      1495 |     21678 |       0% |     70 |    203 |   9.7 |  100% |   1 runs
app      |   64 | parrot.jpg      |       28945 |      1229 |     27716 |       0% |     70 |    253 |   9.4 |  100% |   1 runs
app      |   64 | wedding.jpg     |       17809 |      1230 |     16579 |       0% |     70 |    205 |  13.0 |  100% |   0 runs
app      |  256 | forest.jpg      |       15726 |      2749 |     12977 |       0% |    103 |    118 |   9.7 |  100% |   1 runs
app      |  256 | graduate.jpg    |       15153 |      3043 |     12110 |       0% |    112 |    108 |   9.5 |  100% |   1 runs
app      |  256 | parrot.jpg      |       20749 |      3762 |     16987 |       0% |    112 |    151 |   9.3 |  100% |   1 runs
app      |  256 | wedding.jpg     |       29117 |      3150 |     25967 |       0% |    112 |    247 |   9.8 |  100% |   1 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           8
Runs with pressure:   7 (87.5%)
  - Slow token:       7
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    2166 MB
Max page faults:      180652
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260401_225032.csv
Individual results saved to: results\vision\performVis_ind_20260401_225032.csv
--- PASS: TestVisionPerformance (195.87s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  196.082s




====================================================================================================
HARDWARE
====================================================================================================
GPU:         Vulkan0 (gpu_vulkan) | 4061 MB total | 3655 MB free
System RAM:  4188 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64 256]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | GPU Use%
----------------------------------------------------------------------------------------------------
app        |   8192 |   2048 |    1024 |     Q8_0 |     Q8_0 |      934.7 |      112.0 |    23.0%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
app      @ 64: avgTime  39498ms | ttft   7129ms | avgTimeVar  32% | inTok   70 | outTok 208 | Tok/s  7.0
app      @256: avgTime  36827ms | ttft  10002ms | avgTimeVar  28% | inTok  110 | outTok 193 | Tok/s  7.4
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
app      |   64 | forest.jpg      |       56629 |      7185 |     49444 |       0% |     70 |    296 |   6.8 |  100% |   1 runs
app      |   64 | graduate.jpg    |       32588 |      7150 |     25438 |       0% |     70 |    171 |   7.1 |  100% |   1 runs
app      |   64 | parrot.jpg      |       27391 |      7092 |     20299 |       0% |     70 |    133 |   6.9 |  100% |   1 runs
app      |   64 | wedding.jpg     |       41386 |      7088 |     34298 |       0% |     70 |    232 |   7.0 |  100% |   1 runs
app      |  256 | forest.jpg      |       39957 |      9250 |     30707 |       0% |    103 |    260 |   8.8 |  100% |   1 runs
app      |  256 | graduate.jpg    |       27409 |     10256 |     17153 |       0% |    112 |    113 |   7.0 |  100% |   1 runs
app      |  256 | parrot.jpg      |       49799 |     10239 |     39560 |       0% |    112 |    270 |   7.0 |  100% |   1 runs
app      |  256 | wedding.jpg     |       30142 |     10265 |     19877 |       0% |    112 |    129 |   6.9 |  100% |   1 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           8
Runs with pressure:   8 (100.0%)
  - Slow token:       8
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    1369 MB
Max page faults:      255894
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260401_223309.csv
Individual results saved to: results\vision\performVis_ind_20260401_223309.csv
--- PASS: TestVisionPerformance (332.05s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  332.380s





====================================================================================================
HARDWARE
====================================================================================================
GPU:         none
System RAM:  2220 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64 256]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | RAM Use%
----------------------------------------------------------------------------------------------------
app        |   8192 |   2048 |    1024 |     Q8_0 |     Q8_0 |      934.7 |        0.0 |    42.1%
small      |   2048 |   1024 |     512 |     Q8_0 |     Q8_0 |      934.7 |        0.0 |    42.1%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
app      @ 64: avgTime  29887ms | ttft   1684ms | avgTimeVar   0% | inTok   70 | outTok 202 | Tok/s 10.4
app      @256: avgTime  23747ms | ttft   3333ms | avgTimeVar   0% | inTok  110 | outTok 180 | Tok/s  9.8
small    @ 64: avgTime  24481ms | ttft   1276ms | avgTimeVar   0% | inTok   70 | outTok 225 | Tok/s 10.3
small    @256: avgTime  24557ms | ttft   3156ms | avgTimeVar   0% | inTok  110 | outTok 183 | Tok/s  9.1
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
app      |   64 | .locallens.index |           0 |         0 |         0 |       0% |      0 |      0 |   0.0 |    0% |   0 runs
app      |   64 | forest.jpg      |       35873 |      4690 |     31183 |       0% |     70 |    233 |   9.8 |  100% |   1 runs
app      |   64 | graduate.jpg    |       23476 |      1197 |     22279 |       0% |     70 |    114 |   9.9 |  100% |   1 runs
app      |   64 | lighthouse.jpg  |       28282 |      1250 |     27032 |       0% |     70 |    162 |   9.9 |  100% |   1 runs
app      |   64 | marvel.jpg      |       42323 |      1348 |     40975 |       0% |     68 |    299 |   9.9 |  100% |   1 runs
app      |   64 | night.jpg       |       27869 |      1238 |     26631 |       0% |     70 |    172 |   9.8 |  100% |   1 runs
app      |   64 | parrot.jpg      |       21455 |      1274 |     20181 |       0% |     70 |    126 |  10.0 |  100% |   1 runs
app      |   64 | vietnam.jpg     |       22325 |      1190 |     21135 |       0% |     70 |    213 |  13.7 |  100% |   0 runs
app      |   64 | wedding.jpg     |       37490 |      1283 |     36207 |       0% |     70 |    300 |   9.9 |  100% |   1 runs
app      |  256 | .locallens.index |           0 |         0 |         0 |       0% |      0 |      0 |   0.0 |    0% |   0 runs
app      |  256 | forest.jpg      |       27795 |      2600 |     25195 |       0% |    103 |    197 |   9.9 |  100% |   1 runs
app      |  256 | graduate.jpg    |       19687 |      2983 |     16704 |       0% |    112 |    127 |   9.9 |  100% |   1 runs
app      |  256 | lighthouse.jpg  |       18578 |      4307 |     14271 |       0% |    121 |    121 |   9.9 |  100% |   1 runs
app      |  256 | marvel.jpg      |       33312 |      2713 |     30599 |       0% |     94 |    286 |   9.9 |  100% |   1 runs
app      |  256 | night.jpg       |       20982 |      3258 |     17724 |       0% |    112 |    156 |   9.8 |  100% |   1 runs
app      |  256 | parrot.jpg      |       16974 |      2978 |     13996 |       0% |    112 |    131 |   9.9 |  100% |   1 runs
app      |  256 | vietnam.jpg     |       32599 |      3728 |     28871 |       0% |    112 |    272 |   9.7 |  100% |   1 runs
app      |  256 | wedding.jpg     |       20052 |      4098 |     15954 |       0% |    112 |    150 |   9.9 |  100% |   1 runs
small    |   64 | .locallens.index |           0 |         0 |         0 |       0% |      0 |      0 |   0.0 |    0% |   0 runs
small    |   64 | forest.jpg      |       20274 |      1439 |     18835 |       0% |     70 |    179 |   9.9 |  100% |   1 runs
small    |   64 | graduate.jpg    |       15213 |      1267 |     13946 |       0% |     70 |    131 |  10.0 |  100% |   1 runs
small    |   64 | lighthouse.jpg  |       32219 |      1288 |     30931 |       0% |     70 |    300 |   9.9 |  100% |   1 runs
small    |   64 | marvel.jpg      |       32319 |      1188 |     31131 |       0% |     68 |    300 |   9.9 |  100% |   1 runs
small    |   64 | night.jpg       |       13819 |      1290 |     12529 |       0% |     70 |    159 |  13.6 |  100% |   0 runs
small    |   64 | parrot.jpg      |       22049 |      1289 |     20760 |       0% |     70 |    189 |   9.5 |  100% |   1 runs
small    |   64 | vietnam.jpg     |       29445 |      1241 |     28204 |       0% |     70 |    271 |   9.9 |  100% |   1 runs
small    |   64 | wedding.jpg     |       30508 |      1207 |     29301 |       0% |     70 |    274 |   9.6 |  100% |   1 runs
small    |  256 | .locallens.index |           0 |         0 |         0 |       0% |      0 |      0 |   0.0 |    0% |   0 runs
small    |  256 | forest.jpg      |       19379 |      3190 |     16189 |       0% |    103 |    127 |   8.2 |  100% |   1 runs
small    |  256 | graduate.jpg    |       14807 |      2954 |     11853 |       0% |    112 |    106 |   9.6 |  100% |   1 runs
small    |  256 | lighthouse.jpg  |       17076 |      3536 |     13540 |       0% |    121 |    119 |   9.4 |  100% |   1 runs
small    |  256 | marvel.jpg      |       35042 |      2199 |     32843 |       0% |     94 |    300 |   9.4 |  100% |   1 runs
small    |  256 | night.jpg       |       34242 |      3124 |     31118 |       0% |    112 |    252 |   9.1 |  100% |   1 runs
small    |  256 | parrot.jpg      |       20819 |      3423 |     17396 |       0% |    112 |    141 |   9.0 |  100% |   1 runs
small    |  256 | vietnam.jpg     |       35268 |      3242 |     32026 |       0% |    112 |    280 |   9.0 |  100% |   1 runs
small    |  256 | wedding.jpg     |       19821 |      3583 |     16238 |       0% |    112 |    142 |   9.2 |  100% |   1 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           36
Runs with pressure:   30 (83.3%)
  - Slow token:       30
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    954 MB
Max page faults:      265864
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260325_143938.csv
Individual results saved to: results\vision\performVis_ind_20260325_143938.csv
--- PASS: TestVisionPerformance (845.98s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  846.196s



====================================================================================================
HARDWARE
====================================================================================================
GPU:         CUDA0 (gpu_cuda) | 6143 MB total | 5105 MB free
System RAM:  24432 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64 256]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | GPU Use%
----------------------------------------------------------------------------------------------------
app        |   8192 |   2048 |    1024 |     Q8_0 |     Q8_0 |      934.7 |        0.0 |    15.2%
small      |   2048 |   1024 |     512 |     Q8_0 |     Q8_0 |      934.7 |        0.0 |    15.2%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
app      @ 64: avgTime   7574ms | ttft    182ms | avgTimeVar   0% | inTok   70 | outTok 252 | Tok/s 56.0
app      @256: avgTime   7231ms | ttft    223ms | avgTimeVar   0% | inTok  110 | outTok 176 | Tok/s 36.6
small    @ 64: avgTime   6194ms | ttft    137ms | avgTimeVar   0% | inTok   70 | outTok 196 | Tok/s 56.1
small    @256: avgTime   7934ms | ttft    235ms | avgTimeVar   0% | inTok  110 | outTok 205 | Tok/s 36.2
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
app      |   64 | forest.jpg      |       10481 |       470 |     10011 |       0% |     70 |    246 |  26.6 |  100% |   0 runs
app      |   64 | graduate.jpg    |        9446 |       195 |      9251 |       0% |     70 |    214 |  25.2 |  100% |   0 runs
app      |   64 | lighthouse.jpg  |       12637 |       214 |     12423 |       0% |     70 |    295 |  25.3 |  100% |   0 runs
app      |   64 | marvel.jpg      |        3854 |       188 |      3666 |       0% |     68 |    300 | 104.0 |  100% |   0 runs
app      |   64 | night.jpg       |        3453 |        66 |      3387 |       0% |     70 |    275 | 106.7 |  100% |   0 runs
app      |   64 | parrot.jpg      |        6288 |        81 |      6207 |       0% |     70 |    152 |  28.1 |  100% |   0 runs
app      |   64 | vietnam.jpg     |        3488 |       154 |      3334 |       0% |     70 |    267 | 105.6 |  100% |   0 runs
app      |   64 | wedding.jpg     |       10942 |        91 |     10851 |       0% |     70 |    267 |  26.6 |  100% |   0 runs
app      |  256 | forest.jpg      |        6483 |       236 |      6247 |       0% |    103 |    145 |  26.6 |  100% |   0 runs
app      |  256 | graduate.jpg    |        5814 |       266 |      5548 |       0% |    112 |    128 |  26.8 |  100% |   0 runs
app      |  256 | lighthouse.jpg  |        6364 |       237 |      6127 |       0% |    121 |    145 |  27.1 |  100% |   0 runs
app      |  256 | marvel.jpg      |       12795 |       212 |     12583 |       0% |     94 |    300 |  25.4 |  100% |   0 runs
app      |  256 | night.jpg       |        6857 |       253 |      6604 |       0% |    112 |    155 |  26.7 |  100% |   0 runs
app      |  256 | parrot.jpg      |        2196 |       214 |      1982 |       0% |    112 |    127 | 107.0 |  100% |   0 runs
app      |  256 | vietnam.jpg     |       11108 |       116 |     10992 |       0% |    112 |    273 |  26.7 |  100% |   0 runs
app      |  256 | wedding.jpg     |        6233 |       248 |      5985 |       0% |    112 |    138 |  26.6 |  100% |   0 runs
small    |   64 | forest.jpg      |        2846 |       116 |      2730 |       0% |     70 |    206 | 106.7 |  100% |   0 runs
small    |   64 | graduate.jpg    |        5337 |        89 |      5248 |       0% |     70 |    128 |  28.5 |  100% |   0 runs
small    |   64 | lighthouse.jpg  |        2245 |       149 |      2096 |       0% |     70 |    132 | 102.4 |  100% |   0 runs
small    |   64 | marvel.jpg      |       12119 |        92 |     12027 |       0% |     68 |    300 |  26.7 |  100% |   0 runs
small    |   64 | night.jpg       |        5883 |       211 |      5672 |       0% |     70 |    128 |  26.4 |  100% |   0 runs
small    |   64 | parrot.jpg      |        6296 |       188 |      6108 |       0% |     70 |    141 |  26.5 |  100% |   0 runs
small    |   64 | vietnam.jpg     |        3269 |       157 |      3112 |       0% |     70 |    243 | 104.9 |  100% |   0 runs
small    |   64 | wedding.jpg     |       11558 |        90 |     11468 |       0% |     70 |    288 |  27.0 |  100% |   0 runs
small    |  256 | forest.jpg      |        6446 |       261 |      6185 |       0% |    103 |    146 |  27.0 |  100% |   0 runs
small    |  256 | graduate.jpg    |        4613 |       253 |      4360 |       0% |    112 |     99 |  27.7 |  100% |   0 runs
small    |  256 | lighthouse.jpg  |        6746 |       259 |      6487 |       0% |    121 |    155 |  27.1 |  100% |   0 runs
small    |  256 | marvel.jpg      |       12453 |       211 |     12242 |       0% |     94 |    293 |  25.5 |  100% |   0 runs
small    |  256 | night.jpg       |       12934 |       267 |     12667 |       0% |    112 |    300 |  25.3 |  100% |   0 runs
small    |  256 | parrot.jpg      |        3487 |       234 |      3253 |       0% |    112 |    253 | 103.8 |  100% |   0 runs
small    |  256 | vietnam.jpg     |        9980 |       126 |      9854 |       0% |    112 |    246 |  27.1 |  100% |   0 runs
small    |  256 | wedding.jpg     |        6811 |       267 |      6544 |       0% |    112 |    151 |  26.2 |  100% |   0 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           32
Runs with pressure:   0 (0.0%)
  - Slow token:       0
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    23087 MB
Max page faults:      81398
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260323_222422.csv
Individual results saved to: results\vision\performVis_ind_20260323_222422.csv
--- PASS: TestVisionPerformance (245.81s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  246.088s



====================================================================================================
HARDWARE
====================================================================================================
GPU:         CUDA0 (gpu_cuda) | 6143 MB total | 5105 MB free
System RAM:  24955 MB
GPU Offload: true

====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [128 256]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include:
                        objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV |   VRAM(MB) | KVSlot(MB) | GPU Use%
----------------------------------------------------------------------------------------------------
small      |   8192 |   2048 |    1024 |     Q8_0 |     Q8_0 |      934.7 |        0.0 |    15.2%

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
small    @128: avgTime   6426ms | ttft    170ms | avgTimeVar   0% | inTok   75 | outTok 225 | Tok/s 61.1
small    @256: avgTime   6904ms | ttft    190ms | avgTimeVar   0% | inTok  112 | outTok 208 | Tok/s 51.5
====================================================================================================

==================================================================================================================================
GROUPED RESULTS
==================================================================================================================================
Config   |  Max | Image           | AvgTime(ms) |  TTFT(ms) |   GenTime | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
----------------------------------------------------------------------------------------------------------------------------------
small    |  128 | forest.jpg      |        4382 |       460 |      3922 |       0% |     75 |    300 |  95.4 |  100% |   0 runs
small    |  128 | graduate.jpg    |        3003 |        67 |      2936 |       0% |     75 |    212 |  98.1 |  100% |   0 runs
small    |  128 | lighthouse.jpg  |        4965 |        92 |      4873 |       0% |     80 |    118 |  28.6 |  100% |   0 runs
small    |  128 | marvel.jpg      |       10907 |       143 |     10764 |       0% |     70 |    250 |  25.0 |  100% |   0 runs
small    |  128 | night.jpg       |        4089 |       167 |      3922 |       0% |     75 |    300 |  95.4 |  100% |   0 runs
small    |  128 | parrot.jpg      |        2434 |        68 |      2366 |       0% |     75 |    153 |  97.9 |  100% |   0 runs
small    |  128 | vietnam.jpg     |       11882 |        89 |     11793 |       0% |     75 |    272 |  24.6 |  100% |   0 runs
small    |  128 | wedding.jpg     |        9744 |       272 |      9472 |       0% |     75 |    197 |  23.5 |  100% |   0 runs
small    |  256 | forest.jpg      |        4509 |       196 |      4313 |       0% |    105 |    274 |  85.3 |  100% |   0 runs
small    |  256 | graduate.jpg    |        5238 |       116 |      5122 |       0% |    114 |    129 |  29.5 |  100% |   0 runs
small    |  256 | lighthouse.jpg  |        2401 |       200 |      2201 |       0% |    123 |    134 |  95.9 |  100% |   0 runs
small    |  256 | marvel.jpg      |        9050 |       109 |      8941 |       0% |     96 |    229 |  28.1 |  100% |   0 runs
small    |  256 | night.jpg       |        8972 |       215 |      8757 |       0% |    114 |    213 |  26.8 |  100% |   0 runs
small    |  256 | parrot.jpg      |       11074 |       222 |     10852 |       0% |    114 |    263 |  26.2 |  100% |   0 runs
small    |  256 | vietnam.jpg     |       11414 |       233 |     11181 |       0% |    114 |    273 |  26.3 |  100% |   0 runs
small    |  256 | wedding.jpg     |        2576 |       225 |      2351 |       0% |    114 |    145 |  94.3 |  100% |   0 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           16
Runs with pressure:   0 (0.0%)
  - Slow token:       0
  - High page faults: 0
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    23644 MB
Max page faults:      79571
====================================================================================================

Grouped results saved to: results\vision\performVis_grp_20260321_105733.csv
Individual results saved to: results\vision\performVis_ind_20260321_105733.csv
--- PASS: TestVisionPerformance (113.90s)
PASS
ok      github.com/ramon-reichert/locallens/internal/service/tests/performance  114.187s



====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include:
                        objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV
----------------------------------------------------------------------
small      |   1024 |   1024 |    1024 |     Q8_0 |     Q8_0

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
small    @ 64: avgTime   2844ms | avgTimeVar   0% | inTok   72 | outTok 206 | Tok/s 108.4
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
--------------------------------------------------------------------------------------------------------------
small    |   64 | .locallens.index |           0 |       0% |      0 |      0 |   0.0 |    0% |   0 runs
small    |   64 | forest.jpg      |        3408 |       0% |     72 |    230 | 103.5 |  100% |   1 runs
small    |   64 | graduate.jpg    |        1978 |       0% |     72 |    123 | 112.0 |  100% |   1 runs
small    |   64 | lighthouse.jpg  |        2255 |       0% |     72 |    154 | 111.5 |  100% |   1 runs
small    |   64 | marvel.jpg      |        3576 |       0% |     70 |    300 | 111.3 |  100% |   1 runs
small    |   64 | night.jpg       |        2482 |       0% |     72 |    167 | 107.9 |  100% |   1 runs
small    |   64 | parrot.jpg      |        2267 |       0% |     72 |    144 | 106.3 |  100% |   1 runs
small    |   64 | vietnam.jpg     |        3326 |       0% |     72 |    255 | 107.1 |  100% |   1 runs
small    |   64 | wedding.jpg     |        3461 |       0% |     72 |    277 | 107.7 |  100% |   1 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           9
Runs with pressure:   8 (88.9%)
  - Slow token:       0
  - High page faults: 8
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    23389 MB
Max page faults:      80061
====================================================================================================
failed to save grouped CSV: open results/vision/performVis_grp_20260304_180408.csv: O sistema não pode encontrar o caminho especificado.
failed to save individual CSV: open results/vision/performVis_ind_20260304_180408.csv: O sistema não pode encontrar o caminho especificado.



====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64 384]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include:
                        objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV
----------------------------------------------------------------------
small      |   1024 |   1024 |    1024 |     Q8_0 |     Q8_0
large      |   4096 |   2048 |    2048 |     Q8_0 |     Q8_0

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
large    @ 64: avgTime  20572ms | avgTimeVar   0% | inTok   72 | outTok 181 | Tok/s 10.6
large    @384: avgTime  36990ms | avgTimeVar   0% | inTok  182 | outTok 234 | Tok/s 10.2
small    @ 64: avgTime  21949ms | avgTimeVar   0% | inTok   72 | outTok 202 | Tok/s 10.9
small    @384: avgTime  68175ms | avgTimeVar   0% | inTok  182 | outTok 188 | Tok/s 10.6
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
--------------------------------------------------------------------------------------------------------------
small    |   64 | forest.jpg      |       24670 |       0% |     72 |    228 |  10.8 |  100% |   1 runs
small    |   64 | graduate.jpg    |       14298 |       0% |     72 |    118 |  10.9 |  100% |   1 runs
small    |   64 | lighthouse.jpg  |       15570 |       0% |     72 |    132 |  11.0 |  100% |   1 runs
small    |   64 | marvel.jpg      |       31155 |       0% |     70 |    300 |  10.8 |  100% |   1 runs
small    |   64 | night.jpg       |       16062 |       0% |     72 |    139 |  10.9 |  100% |   1 runs
small    |   64 | parrot.jpg      |       15274 |       0% |     72 |    137 |  11.4 |  100% |   1 runs
small    |   64 | vietnam.jpg     |       27895 |       0% |     72 |    262 |  10.8 |  100% |   1 runs
small    |   64 | wedding.jpg     |       30668 |       0% |     72 |    297 |  10.9 |  100% |   1 runs
small    |  384 | forest.jpg      |       25884 |       0% |    172 |    157 |  11.0 |  100% |   1 runs
small    |  384 | graduate.jpg    |       22808 |       0% |    186 |     92 |  10.4 |  100% |   1 runs
small    |  384 | lighthouse.jpg  |      337686 |       0% |    214 |    275 |   9.8 |  100% |   1 runs
small    |  384 | marvel.jpg      |       41202 |       0% |    144 |    300 |   9.9 |  100% |   1 runs
small    |  384 | night.jpg       |       24219 |       0% |    186 |    122 |  11.1 |  100% |   1 runs
small    |  384 | parrot.jpg      |       25063 |       0% |    186 |    132 |  11.0 |  100% |   1 runs
small    |  384 | vietnam.jpg     |       40003 |       0% |    186 |    263 |  10.7 |  100% |   1 runs
small    |  384 | wedding.jpg     |       28534 |       0% |    186 |    166 |  10.7 |  100% |   1 runs
large    |   64 | forest.jpg      |       19581 |       0% |     72 |    184 |  11.3 |  100% |   1 runs
large    |   64 | graduate.jpg    |       14545 |       0% |     72 |    122 |  11.1 |  100% |   1 runs
large    |   64 | lighthouse.jpg  |       14866 |       0% |     72 |    129 |  11.1 |  100% |   1 runs
large    |   64 | marvel.jpg      |       30068 |       0% |     70 |    300 |  11.2 |  100% |   1 runs
large    |   64 | night.jpg       |       17340 |       0% |     72 |    137 |  10.1 |  100% |   1 runs
large    |   64 | parrot.jpg      |       19756 |       0% |     72 |    161 |  10.3 |  100% |   1 runs
large    |   64 | vietnam.jpg     |       28718 |       0% |     72 |    265 |  10.4 |  100% |   1 runs
large    |   64 | wedding.jpg     |       19704 |       0% |     72 |    150 |   9.6 |  100% |   1 runs
large    |  384 | forest.jpg      |       39365 |       0% |    172 |    262 |  10.5 |  100% |   1 runs
large    |  384 | graduate.jpg    |       28545 |       0% |    186 |    153 |  10.4 |  100% |   1 runs
large    |  384 | lighthouse.jpg  |       46003 |       0% |    214 |    300 |  10.0 |  100% |   1 runs
large    |  384 | marvel.jpg      |       39979 |       0% |    144 |    300 |  10.3 |  100% |   1 runs
large    |  384 | night.jpg       |       30392 |       0% |    186 |    156 |   9.5 |  100% |   1 runs
large    |  384 | parrot.jpg      |       28511 |       0% |    186 |    148 |  10.6 |  100% |   1 runs
large    |  384 | vietnam.jpg     |       44215 |       0% |    186 |    300 |  10.6 |  100% |   1 runs
large    |  384 | wedding.jpg     |       38906 |       0% |    186 |    253 |  10.2 |  100% |   1 runs

====================================================================================================
Total runs:           32
Runs with pressure:   32 (100.0%)
  - Slow token:       0
  - High page faults: 32
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    24574 MB
Max page faults:      266480
====================================================================================================
failed to save grouped CSV: open results/vision/performVis_grp_20260226_224716.csv: O sistema não pode encontrar o caminho especificado.
failed to save individual CSV: open results/vision/performVis_ind_20260226_224716.csv: O sistema não pode encontrar o caminho especificado.



====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [64 384]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include:
                        objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV
----------------------------------------------------------------------
small      |   1024 |   1024 |    1024 |     Q8_0 |     Q8_0
large      |   4096 |   2048 |    2048 |     Q8_0 |     Q8_0

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
large    @ 64: avgTime  31352ms | avgTimeVar   0% | inTok   72 | outTok 206 | Tok/s  9.3
large    @384: avgTime  25880ms | avgTimeVar   0% | inTok  182 | outTok 178 | Tok/s 10.6
small    @ 64: avgTime  23020ms | avgTimeVar   0% | inTok   72 | outTok 175 | Tok/s 10.7
small    @384: avgTime  26682ms | avgTimeVar   0% | inTok  182 | outTok 182 | Tok/s  9.9
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ | Pressure
--------------------------------------------------------------------------------------------------------------
small    |   64 | forest.jpg      |       20289 |       0% |     72 |    134 |   9.9 |  100% |   1 runs
small    |   64 | graduate.jpg    |       20992 |       0% |     72 |    113 |  10.0 |  100% |   1 runs
small    |   64 | lighthouse.jpg  |       26875 |       0% |     72 |    191 |   9.5 |  100% |   1 runs
small    |   64 | marvel.jpg      |       28916 |       0% |     70 |    300 |  13.4 |  100% |   1 runs
small    |   64 | night.jpg       |       17600 |       0% |     72 |    147 |  13.3 |  100% |   1 runs
small    |   64 | parrot.jpg      |       19903 |       0% |     72 |    129 |   9.8 |  100% |   1 runs
small    |   64 | vietnam.jpg     |       28366 |       0% |     72 |    218 |   9.7 |  100% |   1 runs
small    |   64 | wedding.jpg     |       21215 |       0% |     72 |    169 |   9.8 |  100% |   1 runs
small    |  384 | forest.jpg      |       26042 |       0% |    172 |    146 |   8.2 |  100% |   1 runs
small    |  384 | graduate.jpg    |       18329 |       0% |    186 |    120 |  13.4 |  100% |   1 runs
small    |  384 | lighthouse.jpg  |       27238 |       0% |    214 |    155 |   9.8 |  100% |   1 runs
small    |  384 | marvel.jpg      |       27414 |       0% |    144 |    218 |   9.8 |  100% |   1 runs
small    |  384 | night.jpg       |       19170 |       0% |    186 |    115 |   9.9 |  100% |   1 runs
small    |  384 | parrot.jpg      |       31407 |       0% |    186 |    237 |   9.8 |  100% |   1 runs
small    |  384 | vietnam.jpg     |       37620 |       0% |    186 |    300 |   9.8 |  100% |   1 runs
small    |  384 | wedding.jpg     |       26233 |       0% |    186 |    165 |   8.7 |  100% |   1 runs
large    |   64 | forest.jpg      |       37810 |       0% |     72 |    300 |   9.1 |  100% |   1 runs
large    |   64 | graduate.jpg    |       31156 |       0% |     72 |    144 |   7.6 |  100% |   1 runs
large    |   64 | lighthouse.jpg  |       45667 |       0% |     72 |    130 |   6.4 |  100% |   1 runs
large    |   64 | marvel.jpg      |       44200 |       0% |     70 |    300 |   9.6 |  100% |   1 runs
large    |   64 | night.jpg       |       32462 |       0% |     72 |    267 |   9.5 |  100% |   1 runs
large    |   64 | parrot.jpg      |       15049 |       0% |     72 |    148 |  12.8 |  100% |   1 runs
large    |   64 | vietnam.jpg     |       25711 |       0% |     72 |    215 |   9.3 |  100% |   1 runs
large    |   64 | wedding.jpg     |       18758 |       0% |     72 |    148 |   9.8 |  100% |   1 runs
large    |  384 | forest.jpg      |       23048 |       0% |    172 |    144 |   9.6 |  100% |   1 runs
large    |  384 | graduate.jpg    |       21564 |       0% |    186 |    119 |   9.5 |  100% |   1 runs
large    |  384 | lighthouse.jpg  |       24655 |       0% |    214 |    175 |  13.0 |  100% |   1 runs
large    |  384 | marvel.jpg      |       29293 |       0% |    144 |    300 |  13.3 |  100% |   1 runs
large    |  384 | night.jpg       |       23616 |       0% |    186 |    123 |   9.1 |  100% |   1 runs
large    |  384 | parrot.jpg      |       23563 |       0% |    186 |    139 |   9.1 |  100% |   1 runs
large    |  384 | vietnam.jpg     |       41464 |       0% |    186 |    300 |   8.9 |  100% |   1 runs
large    |  384 | wedding.jpg     |       19836 |       0% |    186 |    120 |  12.3 |  100% |   1 runs

====================================================================================================
MEMORY PRESSURE SUMMARY
====================================================================================================
Total runs:           32
Runs with pressure:   32 (100.0%)
  - Slow token:       0
  - High page faults: 32
  - Low RAM:          0
  - Truncated output: 0
Min available RAM:    1136 MB
Max page faults:      267220
====================================================================================================

Grouped results saved to: results/vision/performVis_grp_20260225_151932.csv
Individual results saved to: results/vision/performVis_ind_20260225_151932.csv



====================================================================================================
CONFIGS
====================================================================================================
Model:       Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:      mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MaxSizes:    [ 384 ]
MaxTokens:   300
Temperature: 0.1

Prompt: You extract image keywords for semantic search.

Describe this image in detail. Include:
objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.


Name       | CtxWin | NBatch | NUBatch |   CacheK |   CacheV
----------------------------------------------------------------------
small      |   1024 |      8 |       8 |     Q8_0 |     Q8_0

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
small    @384: avgTime  31459ms | avgTimeVar   5% | inTok   61 | outTok 264 | Tok/s 14.8
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ
----------------------------------------------------------------------------------------------------
small    |  384 | complex.jpg     |       32980 |       2% |     61 |    301 |  14.7 |  100%
small    |  384 | simple.jpg      |       29938 |       7% |     61 |    226 |  14.8 |  100%
Grouped results saved to: results/performVis_grp_20260126_011916.csv
Individual results saved to: results/performVis_ind_20260126_011916.csv




====================================================================================================
CONFIGS
====================================================================================================
Model:      Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:     mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
CacheTypeK: Q8_0
CacheTypeV: Q8_0
MaxSizes:   [128 64]

Prompt: You extract image keywords for semantic search.
Output format: single comma-separated list of short phrases.
Include both general and specific terms. Don't repeat meaning.
No articles (a, the), no filler words.

Analyze this image and describe clear details into this categories:
- people characteristics
- actions
- objects
- location, environment
- visible text
- lighting, colors
- backgound and atmosphere

Name     | CtxWin | NBatch | NUBatch | MaxTok |  Temp
------------------------------------------------------------
small    |   2048 |      8 |       8 |    200 | 0.1
large    |   8192 |   2048 |    2048 |    200 | 0.1

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
large    @ 64: avgTime   7214ms | avgTimeVar  14% | inTok  107 | outTok  58 | Tok/s 14.0
large    @128: avgTime   9190ms | avgTimeVar  22% | inTok  107 | outTok  70 | Tok/s 14.8
small    @ 64: avgTime   7217ms | avgTimeVar   5% | inTok  107 | outTok  54 | Tok/s 13.9
small    @128: avgTime  10159ms | avgTimeVar  16% | inTok  107 | outTok  75 | Tok/s 13.7
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ
----------------------------------------------------------------------------------------------------
small    |  128 | complex.jpg     |       10159 |      16% |    107 |     75 |  13.7 |  100%
small    |   64 | complex.jpg     |        7217 |       5% |    107 |     54 |  13.9 |  100%
large    |  128 | complex.jpg     |        9190 |      22% |    107 |     70 |  14.8 |   67%
large    |   64 | complex.jpg     |        7214 |      14% |    107 |     58 |  14.0 |  100%
Grouped results saved to: performVis_grp_20260124_191028.csv
Individual results saved to: performVis_ind_20260124_191028.csv




====================================================================================================
CONFIGS
====================================================================================================
Model:      Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:     mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
CacheTypeK: Q8_0
CacheTypeV: Q8_0
MaxSizes:   [128 512]

Prompt: Analyze this image and provide a dense list of keywords and short phrases
optimized for semantic search embedding.
Describe clear details into this categories:
- people characteristics;
- actions;
- objects;
- location, environment;
- visible text;
- lighting, colors;
- backgound and atmosphere.
Include both general and specific terms. Don't repeat meaning.
Output format: single comma-separated list of short phrases.
No articles (a, the), no filler words.

Name     | CtxWin | NBatch | NUBatch | MaxTok |  Temp
------------------------------------------------------------
small    |   2048 |      8 |       8 |    200 | 0.1
smal+tmp |   2048 |      8 |       8 |    200 | 0.4
large    |   8192 |   2048 |    2048 |    200 | 0.1
larg+tmp |   8192 |   2048 |    2048 |    200 | 0.4

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
large    @128: avgTime  10305ms | avgTimeVar  20% | inTok  122 | outTok  96 | Tok/s 15.1
large    @512: avgTime  24470ms | avgTimeVar   5% | inTok  122 | outTok  23 | Tok/s 14.7
larg+tmp @128: avgTime   9095ms | avgTimeVar  17% | inTok  122 | outTok  78 | Tok/s 15.0
larg+tmp @512: avgTime  29649ms | avgTimeVar  10% | inTok  122 | outTok  96 | Tok/s 14.7
small    @128: avgTime  12035ms | avgTimeVar  13% | inTok  122 | outTok 112 | Tok/s 15.0
small    @512: avgTime  30068ms | avgTimeVar  15% | inTok  122 | outTok  75 | Tok/s 14.4
smal+tmp @128: avgTime  11968ms | avgTimeVar  36% | inTok  122 | outTok 115 | Tok/s 14.9
smal+tmp @512: avgTime  29904ms | avgTimeVar  22% | inTok  122 | outTok  84 | Tok/s 14.4
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ
----------------------------------------------------------------------------------------------------
small    |  128 | complex.jpg     |       12035 |      13% |    122 |    112 |  15.0 |  100%
small    |  512 | complex.jpg     |       30068 |      15% |    122 |     75 |  14.4 |  100%
smal+tmp |  128 | complex.jpg     |       11968 |      36% |    122 |    115 |  14.9 |  100%
smal+tmp |  512 | complex.jpg     |       29904 |      22% |    122 |     84 |  14.4 |  100%
large    |  128 | complex.jpg     |       10305 |      20% |    122 |     96 |  15.1 |  100%
large    |  512 | complex.jpg     |       24470 |       5% |    122 |     23 |  14.7 |   67%
larg+tmp |  128 | complex.jpg     |        9095 |      17% |    122 |     78 |  15.0 |  100%
larg+tmp |  512 | complex.jpg     |       29649 |      10% |    122 |     96 |  14.7 |  100%
Grouped results saved to: performVis_grp_20260124_154902.csv
Individual results saved to: performVis_ind_20260124_154902.csv



====================================================================================================
CONFIGS
====================================================================================================
Model:      Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:     mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
CacheTypeK: Q8_0
CacheTypeV: Q8_0
MaxSizes:   [128 256 512]

Prompt: Analyze this image and provide a dense list of keywords and short phrases
optimized for semantic search embedding.
Describe people characteristics, actions, objects, location, visible text, lighting, colors and atmosphere.
Include both general and specific terms.
Output format: single comma-separated list of short phrases.
No articles (a, the), no filler words.

Name     | CtxWin | NBatch | NUBatch | MaxTok |  Temp
------------------------------------------------------------
small    |   2048 |      8 |       8 |    150 | 0.1
+temp    |   2048 |      8 |       8 |    150 | 0.4
large    |   8192 |   2048 |    2048 |    150 | 0.1

====================================================================================================
SUMMARY BY CONFIG + MAXSIZE
====================================================================================================
+temp    @128: avgTime   5446ms | avgTimeVar  16% | inTok   99 | outTok  27 | Tok/s 14.9
+temp    @256: avgTime  10942ms | avgTimeVar  11% | inTok   99 | outTok  73 | Tok/s 14.8
+temp    @512: avgTime  31782ms | avgTimeVar   6% | inTok   99 | outTok  42 | Tok/s 14.3
large    @128: avgTime   7365ms | avgTimeVar  36% | inTok   99 | outTok  60 | Tok/s 15.0
large    @256: avgTime  12758ms | avgTimeVar   8% | inTok   99 | outTok 103 | Tok/s 15.0
large    @512: avgTime  30101ms | avgTimeVar   7% | inTok   99 | outTok  40 | Tok/s 14.7
small    @128: avgTime   8396ms | avgTimeVar  42% | inTok   99 | outTok  63 | Tok/s 14.9
small    @256: avgTime  13112ms | avgTimeVar   8% | inTok   99 | outTok  99 | Tok/s 15.0
small    @512: avgTime  30034ms | avgTimeVar   4% | inTok   99 | outTok  26 | Tok/s 14.4
====================================================================================================

====================================================================================================
GROUPED RESULTS
====================================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |  Succ
----------------------------------------------------------------------------------------------------
small    |  128 | complex.jpg     |        9807 |      70% |     99 |     77 |  15.1 |  100%
small    |  128 | simple.jpg      |        6984 |      14% |     99 |     49 |  14.6 |  100%
small    |  256 | complex.jpg     |       16286 |       2% |     99 |    151 |  14.8 |  100%
small    |  256 | simple.jpg      |        9938 |      14% |     99 |     47 |  15.2 |  100%
small    |  512 | complex.jpg     |       25596 |       5% |     99 |     23 |  14.3 |  100%
small    |  512 | simple.jpg      |       34472 |       3% |     99 |     29 |  14.5 |  100%
+temp    |  128 | complex.jpg     |        5512 |      20% |     99 |     24 |  15.1 |  100%
+temp    |  128 | simple.jpg      |        5380 |      13% |     99 |     30 |  14.7 |  100%
+temp    |  256 | complex.jpg     |       12866 |      19% |     99 |    106 |  14.6 |  100%
+temp    |  256 | simple.jpg      |        9018 |       3% |     99 |     40 |  14.9 |  100%
+temp    |  512 | complex.jpg     |       28345 |       5% |     99 |     49 |  14.5 |  100%
+temp    |  512 | simple.jpg      |       35219 |       7% |     99 |     34 |  14.2 |  100%
large    |  128 | complex.jpg     |        6242 |      25% |     99 |     39 |  14.9 |  100%
large    |  128 | simple.jpg      |        8489 |      47% |     99 |     81 |  15.1 |  100%
large    |  256 | complex.jpg     |       14961 |       8% |     99 |    143 |  14.9 |  100%
large    |  256 | simple.jpg      |       10554 |       7% |     99 |     63 |  15.1 |  100%
large    |  512 | complex.jpg     |       25517 |      12% |     99 |     36 |  14.7 |  100%
large    |  512 | simple.jpg      |       34685 |       1% |     99 |     45 |  14.7 |  100%
Grouped results saved to: performVis_grp_20260124_144514.csv
Individual results saved to: performVis_ind_20260124_144514.csv




======================================================================================
CONFIGS
======================================================================================
Model:      Qwen2-VL-2B-Instruct-Q4_K_M.gguf
MMProj:     mmproj-Qwen2-VL-2B-Instruct-Q4_K_M.gguf
CacheTypeK: Q8_0
CacheTypeV: Q8_0
MaxSizes:   [256 384 512]

Prompt: Analyze this image and provide a dense list of comma-separated keywords and sh
optimized for semantic search embedding. Focus on objects, location, visible text, act
lighting, and atmosphere.
Include both general and specific terms (e.g., "red Honda scooter" not just "vehicle")
Output format: single comma-separated list of short phrases.
No sentences, no articles (a, the), no filler words.

Name     | CtxWin | NBatch | NUBatch | MaxTok |  Temp
------------------------------------------------------------
small    |   2048 |      8 |       8 |    120 | 0.1
+toks    |   2048 |      8 |       8 |    150 | 0.1
large    |   8192 |   2048 |    2048 |    150 | 0.2

======================================================================================
SUMMARY BY CONFIG + MAXSIZE
======================================================================================
+toks    @256: avgTime   8739ms | avgTimeVar  10% | inTok  115 | outTok  26 | Tok/s 14
+toks    @384: avgTime  16234ms | avgTimeVar   8% | inTok  115 | outTok  37 | Tok/s 14
+toks    @512: avgTime  27836ms | avgTimeVar   2% | inTok  115 | outTok  23 | Tok/s 14
large    @256: avgTime   8188ms | avgTimeVar   8% | inTok  115 | outTok  34 | Tok/s 15
large    @384: avgTime  14164ms | avgTimeVar   6% | inTok  115 | outTok  24 | Tok/s 15
large    @512: avgTime  28656ms | avgTimeVar   6% | inTok  115 | outTok  24 | Tok/s 13
small    @256: avgTime   9247ms | avgTimeVar  15% | inTok  115 | outTok  38 | Tok/s 15
small    @384: avgTime  14936ms | avgTimeVar   4% | inTok  115 | outTok  29 | Tok/s 15
small    @512: avgTime  28100ms | avgTimeVar   3% | inTok  115 | outTok  26 | Tok/s 14
======================================================================================

======================================================================================
GROUPED RESULTS
======================================================================================
Config   |  Max | Image           | AvgTime(ms) | TimeVar% |  InTok | OutTok | Tok/s |
--------------------------------------------------------------------------------------
small    |  256 | complex.jpg     |       10890 |      26% |    115 |     54 |  15.0 |
small    |  256 | simple.jpg      |        7605 |       4% |    115 |     22 |  15.1 |
small    |  384 | complex.jpg     |       14022 |       7% |    115 |     36 |  15.0 |
small    |  384 | simple.jpg      |       15851 |       1% |    115 |     21 |  15.0 |
small    |  512 | complex.jpg     |       24508 |       4% |    115 |     30 |  14.8 |
small    |  512 | simple.jpg      |       31692 |       2% |    115 |     21 |  14.5 |
+toks    |  256 | complex.jpg     |        8643 |       9% |    115 |     28 |  14.2 |
+toks    |  256 | simple.jpg      |        8836 |      11% |    115 |     24 |  14.4 |
+toks    |  384 | complex.jpg     |       15908 |       8% |    115 |     54 |  14.7 |
+toks    |  384 | simple.jpg      |       16561 |       8% |    115 |     20 |  14.7 |
+toks    |  512 | complex.jpg     |       24235 |       3% |    115 |     24 |  14.6 |
+toks    |  512 | simple.jpg      |       31438 |       0% |    115 |     22 |  14.6 |
large    |  256 | complex.jpg     |        8408 |       4% |    115 |     41 |  15.1 |
large    |  256 | simple.jpg      |        7967 |      13% |    115 |     26 |  14.9 |
large    |  384 | complex.jpg     |       13336 |      10% |    115 |     27 |  15.0 |
large    |  384 | simple.jpg      |       14991 |       2% |    115 |     21 |  14.9 |
large    |  512 | complex.jpg     |       25580 |       9% |    115 |     25 |  13.6 |
large    |  512 | simple.jpg      |       31732 |       4% |    115 |     23 |  14.3 |
Grouped results saved to: performVis_grp_20260124_111010.csv
Individual results saved to: performVis_ind_20260124_111010.csv