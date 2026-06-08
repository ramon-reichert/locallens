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
- Only `internal/platform/kronk/` imports the Kronk SDK directly (it is the
  SDK boundary). `description/` and `embedding/` also import the SDK for
  model loading and inference. `cmd/` must NOT import the SDK — it uses
  the `platform/kronk` wrapper.

## Configuration

All config lives in `internal/platform/config/config.go`. At runtime, defaults
are overlaid with `~/.locallens/config.json`. `config.Load()` returns
`(Config, error)` — callers must handle the error (log a warning and continue
with defaults). Config holds only static admin-set values; runtime state (e.g.,
whether setup is complete) is derived from system state, not persisted in config.

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
- **cross-platform CI**: GitHub's standard arm64 macOS runners are extremely slow for
  Qwen2-VL vision inference, do not reflecting real behavior. Keep macOs+Windows+Linux CI 
  to build/unit checks, but whole integration (setup+service) tests just for Linux; 
  Windows is not necessary because is already tested in the developer machine.

## Kronk SDK usage

LocalLens is built on top of the [Kronk SDK](https://github.com/ardanlabs/kronk)
(v1.21.4), which wraps llama.cpp via the yzma library using FFI (not cgo).
LocalLens serves as a practical demonstration of Kronk's capabilities for
local vision and embedding inference.

### SDK packages used

| Kronk package | Where used | Purpose |
|---|---|---|
| `sdk/kronk` | `description/`, `embedding/`, `kronk/` (platform) | `kronk.New(model.Config)` loads a model. `kronk.Init()` initializes the runtime. `krn.Unload(ctx)` frees it. |
| `sdk/kronk/model` | `description/`, `embedding/`, perf tests | `model.Config` for engine params (ContextWindow, NBatch, CacheType). `model.D` for chat/embed request data. `model.GGMLType` for cache quantization. |
| `sdk/tools/libs` | `kronk/` (platform) | `libs.New()` + `libs.Download()` to fetch llama.cpp shared libraries. `libs.WithVersion()` pins a specific build. |
| `sdk/tools/models` | `kronk/` (platform) | `models.NewWithPaths()` + `models.Download()` / `models.FullPath()` to download and resolve GGUF model files. |
| `sdk/tools/devices` | perf tests | `devices.List()` for GPU detection (name, type, VRAM, system RAM). |
| `sdk/tools/defaults` | `kronk/` (platform) | `defaults.LibVersion()` resolves pinned vs latest llama.cpp version. `defaults.Processor()` parses GPU backend names. |

### Chat (vision) API pattern

```go
// Build message list with image bytes and text prompts
messages := []model.D{}
messages = append(messages, model.TextMessage(model.RoleSystem, systemPrompt))
messages = append(messages, model.ImageMessage(userPrompt, imageBytes, "jpg")...)

data := model.D{
    "messages":    messages,
    "temperature": temperature,
    "max_tokens":  maxTokens,
}
resp, err := krn.Chat(ctx, data)
if err != nil { ... }
if len(resp.Choices) == 0 { ... } // empty response is a real error

choice := resp.Choices[0]
switch {
case choice.FinishReason() == model.FinishReasonError:
    ... // KV/cache/model decode errors
case choice.Message == nil || choice.Message.Content == "":
    ... // empty/blank message is a real error
case resp.Usage == nil:
    ... // metrics are required by LocalLens
}

// Extract results
description := choice.Message.Content
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

The two models have different lifetimes by design:

- **Embedder** (~320 MB) — loaded once at `service.New()` (app startup) and
  kept resident for the lifetime of the `Service`. Every Search needs it to
  vectorize the query, and indexing reuses it to embed each description.
  Holding it costs ~320 MB but eliminates repeated load/unload overhead.
- **Vision model** (~935 MB) — loaded at the start of each `IndexFolder` call
  and unloaded (via `defer`) when the call returns. It is only needed during
  indexing, and is too large to hold permanently on low-RAM machines.

```
service.New(ctx):
  → embedder.Load()          # stays loaded for app lifetime

service.IndexFolder(ctx, folderPath, progress):
  → countNewImages           # skip-already-indexed pre-pass
  → (short-circuit if none)  # avoids paying any model cost
  → describer.Load()         # vision model
  → for each new image:
      describer.Describe()   # vision: image → text
      embedder.Embed()       # embedder is already loaded
      index.Add() + Save()   # durable per-image
  → describer.Unload()       # via defer, even on cancellation

service.Close(ctx):
  → describer.Unload()
  → embedder.Unload()
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

Indexing processes a single folder (non-recursive). For each image not already
in the folder's `.locallens.index`, the full describe → embed → save cycle
runs end-to-end before moving to the next image, so a crash mid-run loses at
most the in-flight image.

```
service.IndexFolder(ctx, folderPath, progress):
  for each new image:
    → image.Resize(path, maxSide)        # Resize to maxSide px, JPEG encode
    → description.Describe(ctx, path)    # Vision model: image → text
    → embedding.Embed(ctx, description)  # Embedder: text → []float32
    → index.Add(Entry{Path, Description, Embedding})
    → index.Save()                       # Write .locallens.index in the folder
    → tracker.record(...)                # Progress callback (SSE event)

service.Search(ctx, folderPath, query, k):
  → loadIndex(folderPath)                # Cached after first call; disk hit
                                         # only on cache miss
  → embedding.Embed(ctx, query)          # Query text → vector
  → search.FindTopK(queryVec, entries, k)# Cosine similarity ranking
```

The embedder is always loaded (see Model lifecycle), so search is just one
embed call plus an in-memory similarity pass on cached entries. The vision
model is loaded once per `IndexFolder` call and unloaded via `defer` on the
way out — including on cancellation.

Recursive indexing across a directory tree is not currently supported and is
a planned future step. The Service exposes a single public indexing method,
`IndexFolder`.

## Browser UI decisions

The UI is a Go HTTP server serving a vanilla HTML/CSS/JS frontend. This was
chosen after evaluating alternatives:

- **Shell extension**: Rejected (requires COM/C++ on Windows, Explorer has no
  external sort API).
- **Electron**: Rejected (150 MB+ binary, overkill).
- **Tauri**: Rejected (requires Rust dependency).
- **Fyne**: Attempted, abandoned (requires CGO + OpenGL; hit a Go 1.25 Windows
  cgo bug). Incompatible with the CGO_ENABLED=0 constraint.
- **Browser UI**: Selected as the pragmatic starting point.

Key implementation decisions:

- **Vanilla stack**: No frameworks, no build tools, no JS dependencies.
- **Embedded static files**: `go:embed` in `cmd/locallens/` for single-binary
  distribution.
- **Standard library routing**: `http.ServeMux`, no third-party router.
- **SSE for progress**: Setup downloads and folder indexing both stream
  progress to the browser via Server-Sent Events. The `/api/index` endpoint
  uses the request context for cancellation — when the user clicks Stop, the
  frontend `AbortController` closes the connection, cancelling `r.Context()`
  on the server and stopping indexing after the in-flight image (already-saved
  progress is preserved).
- **CGO_ENABLED=0**: The entire stack avoids CGO (Kronk SDK uses FFI, not cgo).
- **503 pattern**: Handlers return `setup_required` when `Service` is nil;
  frontend opens the setup modal.
- **No console window**: Production builds use `-ldflags "-H windowsgui"`.

Known browser limitations:

- No native file drag-and-drop from browser to OS.
- Sandboxed filesystem — folder tree is a web-based recreation.
- Explorer focus issues on Windows solved with `cmd /C start`.

### Future: Wails migration

**Wails** (Go + webview) is the planned migration target for a native desktop
experience. The same HTML/JS frontend works inside Wails with minimal changes,
gaining native dialogs, single-click launch, and better filesystem access.
Migration is deferred until the browser UI stabilizes.

## Indexing: durability, progress, and cancellation

The indexing pipeline is built around four properties:

1. **Always-loaded embedder, on-demand vision model** — see Model lifecycle.
2. **In-memory index cache** — `Service.indexes` (`map[string]*index.Index`,
   guarded by `sync.RWMutex`) caches the `.locallens.index` deserialized from
   each folder. First search hits disk; subsequent searches and re-indexes
   on the same folder are pure in-memory. Because `idx.Add()`/`idx.Save()`
   mutate the cached pointer in place, no separate cache-write step is needed
   after indexing — the cache stays consistent automatically.
3. **Per-image durability** — `indexFolder()` runs describe → embed →
   `idx.Add` → `idx.Save` for each image. A crash mid-run loses at most the
   in-flight image. On resume, `idx.Get(imgPath)` short-circuits already-saved
   entries.
4. **Connection-bound cancellation** — `/api/index` is SSE. The handler passes
   `r.Context()` to `Service.IndexFolder`. When the user clicks Stop, the
   frontend `AbortController` closes the connection, which cancels the server
   context. `indexFolder()` checks `ctx.Err()` at the top of every image
   iteration and returns `ctx.Err()` as soon as the in-flight image completes
   (and saves). No separate cancel endpoint needed.

### Progress reporting

`Service.IndexFolder` returns `IndexResult{IndexedTotal, Added, Failed, Total}`
and accepts an `IndexProgress` callback (`func(IndexProgressInfo)`), invoked at
these stages per image:

- `Stage: "describing"` — fired right before the vision-model call. Used to
  show "Describing image X/N — file.jpg" in the UI status. ETA is not set
  (no timing data yet).
- `Stage: "indexed"` — fired after `idx.Save()`. Increments the running
  counter and computes a running-average ETA over completed images. Used to
  advance the progress bar and flip the per-file ✅ marker.
- `Stage: "failed"` — fired when describe/embed returns an error. The image is
  skipped, indexing continues, and the final UI summary reports the failure count.

`IndexProgressInfo` fields: `Stage`, `Folder`, `Current`, `Done`, `Failed`,
`Processed`, `Total`, `ETA`, `Error`.

`Total` is computed upfront by `countNewImages()`, which walks the folder's
image list and excludes entries already present in the cached index. When
`Total == 0` the function short-circuits before loading the vision model —
re-running Index on an already-indexed folder is nearly instant (~ms, not 30s).

### SSE event types on `/api/index`

The handler in [`handleIndex`](file:///b:/dev/locallens/internal/app/handlers.go)
emits the following JSON events:

| Event       | When                                       | Payload fields                              |
|-------------|--------------------------------------------|---------------------------------------------|
| `started`   | Immediately, before any work               | `folder`                                    |
| `progress`  | Per image stage update                     | `stage`, `folder`, `current`, `done`, `failed`, `processed`, `total`, `etaMs`, `error` |
| `done`      | All work finished successfully             | `count`, `indexedTotal`, `added`, `failed`, `total` |
| `cancelled` | Context cancelled (user Stop)              | `count`, `indexedTotal`, `added`, `failed`, `total` |
| `error`     | Any fatal error                            | `error`, `count`, `indexedTotal`, `added`, `failed`, `total` |

The `started` event is sent before the (slow) vision-model load specifically
so the browser's `await fetch(...)` resolves immediately and the UI can show
"Loading vision model..." rather than appearing frozen.

### Future: recursive indexing

`IndexFolder` is currently the only public indexing entry point and processes
exactly one folder. Recursion was removed during cleanup because the UI no
longer exposes it. When it returns, the natural shape is:

- Extract a private `indexFolders(ctx, []string, progress)` helper from the
  body of `IndexFolder` (the count/short-circuit/load/defer/loop plumbing).
- Add `IndexTree(ctx, root, progress)` that calls `findImageFolders(root)`
  (a `filepath.Walk` that returns every directory containing image files),
  then delegates to `indexFolders`.
- `IndexFolder` becomes a one-liner: `s.indexFolders(ctx, []string{folderPath}, progress)`.

This keeps the vision-model-loaded-once-per-action property and avoids
duplicating the plumbing that exists today.

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
