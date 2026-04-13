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

## Future refactor: IndexFolder durability and progress

The current `IndexFolder` (service.go) batches all work in memory across two
phases (describe all → embed all) and only saves at the end. A crash mid-run
loses all progress. The following plan addresses durability, progress reporting,
and cancellation. Each step builds on the previous — implement in order.

### Step 1 — Keep the embedding model loaded

The embedding model (embeddinggemma-300m) is ~320 MB — small enough to stay
loaded for the lifetime of the `Service`. Search is the primary use case of
the app and may run many times per session; each Search needs the embedder to
vectorize the query. Keeping it always loaded avoids repeated load/unload
costs for every search. As a secondary benefit, it enables per-image
embed-after-describe during indexing (Step 4).

**Changes:**
- `Service.New()`: call `embedder.Load()` at construction time.
- `Service.Close()`: unload the embedder (already does this).
- Remove `embedder.Load()`/`Unload()` calls from `IndexFolder` and `Search`.
- Remove the `IsLoaded()` guard in `Search` — it's always loaded.

### Step 2 — Cache loaded indexes in memory

Currently `Search` re-reads every `.locallens.index` file from disk on every
call, deserializes all entries, runs cosine similarity, then discards
everything. With the embedder always loaded (Step 1), the remaining cost per
search is disk I/O — eliminate it with an in-memory cache.

**Changes:**
- Add a `map[string]*index.Index` field to `Service` (protected by the
  existing mutex pattern or a dedicated `sync.RWMutex`).
- On first search of a folder, load from disk and cache. Subsequent searches
  read directly from memory.
- When `IndexFolder` saves an index, update the cache for that folder (the
  Service already knows — it calls `idx.Save()`).
- Memory cost is negligible: each entry is a path + description + `[]float32`
  embedding (~1-2 KB per image). 10,000 images ≈ ~15-20 MB, small next to the
  320 MB embedder already in memory.

After Steps 1-2, Search becomes: embed query (one model call) → cosine
similarity over cached vectors → return results. No disk I/O for repeat
searches.

### Step 3 — Bound IndexFolder to a single folder

Currently `IndexFolder` handles recursion internally, grouping images by folder.
Move the recursive walk up to the caller (handler layer) so each `IndexFolder`
call processes exactly one folder. This makes each call a smaller, self-contained
unit of work.

**Changes:**
- Remove the `recursive` parameter and all subfolder logic from `IndexFolder`.
  It processes only the images directly inside `folderPath`.
- Add a new method or helper at the handler/caller level that walks subdirectories
  and calls `IndexFolder` once per folder.
- This wrapper also handles the vision model lifecycle: load once before the
  loop, unload after all folders are processed, so the load/unload cost is
  paid once per user-initiated indexing action, not once per folder.
- Each per-folder call loads one index, processes images, saves, and returns —
  crash-safe at folder granularity.

### Step 4 — Save index after each image

With the embedder always loaded (Step 1) and single-folder scope (Step 3),
the describe→embed→save cycle can run per image instead of per batch.

**Changes:**
- In `IndexFolder`, for each new image:
  1. Describe (vision model).
  2. Embed the description (embedder is already loaded).
  3. Add entry to the in-memory index.
  4. Save the index to disk.
- Remove the `descriptions` map — no intermediate accumulation needed.
- The vision model still loads once at the start and unloads at the end of
  `IndexFolder`, but each image's work is durable as soon as it completes.
- If the app crashes after image N, images 1..N are persisted. On restart,
  the existing-entry skip logic (`indexes[dir].Get(imgPath)`) resumes from N+1.

### Step 5 — Metrics, ETA, and cancellation

With per-image granularity (Step 4), add progress reporting and cancellation.

**Metrics and ETA:**
- Collect TTFT, tok/s, and embed time per image as today, but compute a
  running average after each image.
- Calculate ETA: `avgTimePerImage × remainingImages`.
- Report progress to the caller via a callback (similar to `SetupProgress`):
  image count, percent complete, ETA, current file name.
- The handler layer forwards this to the frontend via SSE (same pattern as
  `handleSetupRun`).

**Cancellation:**
- Check `ctx.Done()` at the top of each image iteration. If cancelled, save
  the index (already saved per-image from Step 4) and return early with the
  count of images processed so far.
- The handler layer wires a cancellable context to an API endpoint (e.g.,
  `POST /api/index/cancel`) or ties it to the SSE connection closing.
- The frontend shows a "Stop" button during indexing that hits the cancel
  endpoint. Progress up to that point is preserved.

### Design constraints

- **Vision model lifecycle**: The vision model (~935 MB) must still load/unload
  per `IndexFolder` call. Do not keep it loaded across calls.
- **Index format**: No schema changes needed. `index.Entry` already has
  `Description` and `Embedding` fields. Partial entries (description without
  embedding) are not stored — Step 4 ensures both are present before saving.
- **No image file modification**: Descriptions live only in the per-folder
  `.locallens.index` file. Do not write metadata to image files or create
  sidecar files.

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
