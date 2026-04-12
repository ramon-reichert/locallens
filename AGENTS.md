# LocalLens — Agent Guide

LocalLens is a local semantic image search application. It uses a vision model
(Qwen2-VL-2B via the Kronk SDK) to describe images, an embedding model
(embeddinggemma-300m) to vectorize those descriptions, and cosine similarity to
find images matching a natural language query. Everything runs locally — no
cloud APIs.

## Project structure (Package Oriented Design)

The codebase follows [Ardan Labs Package Oriented Design](https://www.ardanlabs.com/blog/2017/02/package-oriented-design.html).
Dependency flow is strictly **downward**: `cmd/` → `internal/` → `internal/platform/`.

```
cmd/
  locallens/       Entry point. Loads config, inits Kronk SDK, creates
                   service, injects dependencies into handlers.
  setup/           CLI tool for downloading llama.cpp libs and models.

internal/
  app/             HTTP handlers. Receives all dependencies via injection.
                   Must NOT import config, kronk, or the Kronk SDK directly.
  service/         Orchestrator. Manages describe→embed→index pipeline.
    description/   Vision model wrapper (Describe an image → text).
    embedding/     Embedding model wrapper (text → vector).
    image/         Image resizing before vision inference.
    index/         Per-folder .locallens.index file persistence.
    search/        Cosine similarity search over indexed vectors.
    exif/          EXIF metadata extraction.
    tests/         Integration and performance tests.

internal/platform/
  config/          Centralized configuration (config.json, defaults, types).
  kronk/           Kronk SDK boundary: downloads libs/models, resolves paths.
  logger/          Logger function type.
  sysmon/          System resource monitoring (RAM, page faults).
  web/             HTTP server with graceful shutdown.
```

### Key dependency rules

- `internal/platform/` **cannot** import from `internal/` or `cmd/`.
- `internal/` **cannot** import from `cmd/`.
- `internal/platform/` should not log directly or wrap errors (return root cause values).
- 3rd-party SDK types must not leak into exported APIs of internal packages.
  The project defines `config.ModelFilePaths` as its own type instead of using
  the Kronk SDK's `models.Path`.

## Configuration

All config lives in `internal/platform/config/config.go`. At runtime, defaults
are overlaid with `~/.locallens/config.json`.

Key config fields and their rationale:
- **`llamaCppVersion`**: Pin a specific llama.cpp library version for release stability.
  When empty, uses Kronk's default (latest).
- **`cacheTypeK` / `cacheTypeV`**: Must be `Q8_0`. Q4_0 causes hallucinations
  on Qwen2-VL-2B due to its 7:1 GQA ratio amplifying K-cache quantization noise.
- **`maxTokens: 300`**: Needs ≥200 output tokens for good descriptions. 300 is safe ceiling.
- **`temperature: 0.1`**: Near-deterministic — same image should produce same description.
- **`image.maxSide: 64`**: Image resize constraint. Larger = better detail but slower
  (128px→17s, 256px→23s, 384px→31s on CPU). Values >384 risk KV cache exhaustion.

## Build and test

All commands use `CGO_ENABLED=0` (the Kronk SDK uses FFI, not cgo).

```
make setup                         # Download llama.cpp libs + models
make build                         # Build without console (production)
make build-logs                    # Build with console output
make test-unit                     # Fast unit tests, no model loading
make test-service                  # Integration tests (loads models)
make test-performance-vision       # Vision model benchmarks
make test-performance-similarity   # Embedding similarity quality tests
```

### Test structure

- **`testsboot.Boot()`** initializes the Kronk SDK and resolves model paths once
  per test package. All test files that need models call this first.
- **Unit tests** (`tests/unittests/`): No model loading. Test image resize, index
  persistence, search math.
- **Integration tests** (`tests/service_test.go`): Full pipeline. Use `os.MkdirTemp`
  to avoid writing index files into the source tree.
- **Performance tests** (`tests/performance/`): Benchmark across config variants
  and image sizes. Track TTFT, generation time, tokens/sec, memory pressure.
  A warmup inference runs before measurements to exclude model loading from TTFT.

## Kronk SDK usage

LocalLens is built on top of the [Kronk SDK](https://github.com/ardanlabs/kronk)
(v1.21.4), which wraps llama.cpp via the yzma library using FFI (not cgo).
LocalLens serves as a practical demonstration of Kronk's capabilities for
local vision and embedding inference.

### SDK packages used

| Kronk package | Where used | Purpose |
|---|---|---|
| `sdk/kronk` | `description/`, `embedding/`, `cmd/` | `kronk.New(model.Config)` loads a model. `kronk.Init()` initializes the runtime. `krn.Unload(ctx)` frees it. |
| `sdk/kronk/model` | `description/`, `embedding/`, perf tests | `model.Config` for engine params (ContextWindow, NBatch, CacheType). `model.D` for chat/embed request data. `model.GGMLType` for cache quantization. |
| `sdk/tools/libs` | `kronk/` (platform) | `libs.New()` + `libs.Download()` to fetch llama.cpp shared libraries. `libs.WithVersion()` pins a specific build. |
| `sdk/tools/models` | `kronk/` (platform) | `models.NewWithPaths()` + `models.Download()` / `models.FullPath()` to download and resolve GGUF model files. |
| `sdk/tools/devices` | perf tests | `devices.List()` for GPU detection (name, type, VRAM, system RAM). |
| `sdk/tools/defaults` | `kronk/` (platform) | `defaults.LibVersion()` resolves pinned vs latest llama.cpp version. `defaults.Processor()` parses GPU backend names. |

### Chat (vision) API pattern

```go
// Build message list with image bytes and text prompts
messages := []model.D{
    {"role": "system", "content": systemPrompt},
    {"role": "user", "content": imageBytes},    // []byte from image.Resize
    {"role": "user", "content": userPrompt},
}
data := model.D{
    "messages":    messages,
    "temperature": temperature,
    "max_tokens":  maxTokens,
}
resp, err := krn.Chat(ctx, data)

// Check for decode errors (KV cache exhaustion, etc.)
if resp.Choices[0].FinishReason() == model.FinishReasonError { ... }

// Extract results
description := resp.Choices[0].Message.Content
ttft        := resp.Usage.TimeToFirstTokenMS
tps         := resp.Usage.TokensPerSecond
```

### Embeddings API pattern

```go
data := model.D{
    "input":    text,
    "truncate": true,
}
resp, err := krn.Embeddings(ctx, data)
vector := resp.Data[0].Embedding  // []float32
```

### Model lifecycle

Models are loaded on demand and unloaded after each phase. This is critical
for low-RAM machines where holding both vision (~935 MB) and embedding models
simultaneously would cause memory pressure:

```
service.IndexFolder:
  → describer.Load()     # Load vision model
  → describe all images
  → describer.Unload()   # Free vision model memory
  → embedder.Load()      # Load embedding model
  → embed all descriptions
  → embedder.Unload()    # Free embedding model memory
```

### Hardware detection

Performance tests use `devices.DetectGPU()` to detect GPU availability and
auto-select backends. The `KRONK_PROCESSOR` env var or `config.processor`
field can override auto-detection (values: `cuda`, `vulkan`, `metal`, `cpu`).

## Performance reference

Benchmarked with Qwen2-VL-2B-Instruct (Q4_K_M) + Q8_0 mmproj, KV cache Q8_0,
maxTokens=300, temperature=0.1, descriptive prompt.

### Laptop — NVIDIA RTX 2060 (6 GB VRAM), 24 GB RAM

```
Config   | MaxSz | AvgTime(ms) | TTFT(ms) | OutTok | Tok/s
---------+-------+-------------+----------+--------+------
app      |    64 |        7574 |      182 |    252 |  56.0
app      |   256 |        7231 |      223 |    176 |  36.6
small    |    64 |        6194 |      137 |    196 |  56.1
small    |   256 |        7934 |      235 |    205 |  36.2
```

Configs: app (ctx=8192, batch=2048/1024), small (ctx=2048, batch=1024/512).

### Desktop — CPU only, 8 GB RAM

```
Config   | MaxSz | AvgTime(ms) | TTFT(ms) | OutTok | Tok/s
---------+-------+-------------+----------+--------+------
app      |    64 |       29887 |     1684 |    202 |  10.4
app      |   256 |       23747 |     3333 |    180 |   9.8
small    |    64 |       24481 |     1276 |    225 |  10.3
small    |   256 |       24557 |     3156 |    183 |   9.1
```

Key observations:
- GPU gives ~5× speedup in tokens/sec (56 vs 10 tok/s) and ~4× in wall time.
- TTFT scales with image resolution (more pixels → more prefill tokens).
- Output tokens are consistent (~180–250) regardless of hardware, confirming
  descriptions are not truncated at these settings.
- The `small` config (ctx=2048) performs comparably on CPU, suggesting context
  window size has minimal impact on throughput for single-image inference.

## How the pipeline works

```
Image file
  → image.Resize(path, maxSide)        # Resize to maxSide px, JPEG encode
  → description.Describe(ctx, path)    # Vision model: image → text description
  → embedding.Embed(ctx, description)  # Embedding model: text → float32 vector
  → index.Add(Entry{Path, Description, Embedding})
  → index.Save()                       # Write .locallens.index in the folder

Search query
  → embedding.Embed(ctx, query)        # Query text → vector
  → search.FindTopK(queryVec, entries, k)  # Cosine similarity ranking
```

Models are loaded/unloaded between phases to avoid holding both in memory
simultaneously (important for low-RAM machines).

## Known issues and gotchas

- **Decode errors**: Large images (>512px) or demanding prompts can exhaust the
  KV cache, causing `slice bounds out of range` panics from the SDK. The code
  checks `FinishReasonError` to catch these.
- **Hallucinations**: Randomly repeated words in output. More frequent with Q4_0
  KV cache than Q8_0. Not yet tied to a specific root cause.
- **Template bug**: After a successful description, subsequent files sometimes
  fail. Suspected jinja/gonja template issue in the SDK.
- **SlotMemory from SDK**: `ModelInfo().SlotMemory` returns 0 for Qwen2-VL because
  the GGUF metadata lacks `attention.key_length`/`attention.value_length`. The perf
  test has a local `estimateSlotMemory()` fallback.
- **Development paths**: The project is developed across multiple machines. Do not
  hardcode absolute paths.
