# Future features

- Adapt prompt and search to different idioms;
- Face recognition;
- Configurable detail vs speed;
- Run the tool inside local explorer browsers;
- Run the tool inside photographers apps (adobe,etc);
- Run it in mobile;
- Configuration adaptable to the user hardware constrains;
- LocalEars to audio files;
- LocalMovs to video files;
- Concurrent improvments;
- Include subfolders in search
- Add flags and/or description to files metadata and give option to re-desrcirbe

# To-do:

- System prompt
- Sequential prompts
- Kronk and system metrics
- Search for "TODO" along the codebase;

# Issues

- How to display best scored images? A list, a link, a preview screen? 

# Performance

- Goals:      | accurate image description | fast description process | low hardware demand |
              |-----------------------------------------------------------------------------|
    Needs     | right prompt to mor outTok | small maxSize            | not tracking yet but|
  Constrains  | min viable maxSize         |                          | can be causing fails|


- Image maxSize directly impacts inference time. Ratio: time∝(image size)n, n≈2.2–2.3 => 256px - 6s | 384px - 14,8s | 512px - 28,3s
- Getting good descriptions more often with maxSize 128px than with 512px. Propably due to hardware constrains.

- Tokens/sec seems not to be related with config or image size. It remains aprox. 14-15 tok/s. Maybe it is related with hardware?

- Input tokens is always the same, despite model config, with a minimun viable maxToken value. Maybe it is related with the prompt? YES

- outToks = maxToks+1 means that the model halucinated! Sanity check added.
- outToks < 30 means poor descriptions. Saninty check added. Can be better tuned later.
- Output tokens varies a lot between equal repetitions. Some outstanding descriptions (many outToks) side by side with poor ones in identical runs. Need track hardware metrics to see why is it happening;

- Can we have more predictable descriptions using a system prompt? YES
- Can we achieve more detailed descriptions sending separated prompts for each category(people characteristics, actions, etc)?

- Some performance test outputs:

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