# LocalLens — Agent Guide

LocalLens is a local semantic image search application. Its pipeline uses three
local models (all via the Kronk SDK):

1. A **vision model** (Qwen2-VL-2B) describes each image as free-form prose.
2. A small **categorization model** (Qwen3-0.6B) reshapes that prose into a
   compact list of grammar-constrained "search expressions" (short 2–6 word
   phrases), because a single prose blob mixes too many concepts to embed well.
3. An **embedding model** (embeddinggemma-300m) vectorizes each expression
   individually, giving one vector per expression.

Search embeds the query, scores it against every expression vector, and ranks
images from each image's best expression matches. Everything runs locally — no cloud APIs.

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
  service/         Orchestrator. Manages describe→categorize→embed→index pipeline.
    description/   Vision model wrapper (Describe an image → prose text).
    categorization/ Chat model wrapper (prose → JSON list of search expressions).
    embedding/     Embedding model wrapper (text → vector).
    image/         Image resizing before vision inference.
    index/         Per-folder .locallens.index file persistence (gob-encoded).
    search/        Multi-vector cosine similarity search over indexed vectors.
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
  SDK boundary). `description/`, `categorization/`, and `embedding/` also
  import the SDK for model loading and inference. `cmd/` must NOT import the
  SDK — it uses the `platform/kronk` wrapper.

## Configuration

All config lives in `internal/platform/config/config.go`. At runtime, defaults
are overlaid with `~/.locallens/config.json`. LocalLens stores Kronk models and
llama.cpp libraries under `~/.locallens/kronk` by default, isolated from
Kronk's own `~/.kronk` directory. `config.Load()` returns
`(Config, error)` — callers must handle the error (log a warning and continue
with defaults). Config holds only static admin-set values; runtime state (e.g.,
whether setup is complete) is derived from system state, not persisted in config.

The config now defines **three model engine configs** (`Vision`, `Embed`,
`Categorize`) and **two prompt configs** (`Prompt`/`VisionPrompt` and
`CategorizePrompt`). `Models` holds four download URLs (vision, vision proj,
embed, categorize).

Key config fields and their rationale:
- **`llamaCppVersion: "b9750"`**: Pins a specific llama.cpp library version for
  release stability. When empty, uses Kronk's default (latest).
- **`cacheTypeK` / `cacheTypeV`**: Must be `Q8_0` for the vision model. Q4_0
  causes hallucinations on Qwen2-VL-2B due to its GQA ratio (2 KV heads / 14
  attn heads, llama.cpp PR #7412) amplifying K-cache quantization noise.
- **`maxTokens: 300`**: Needs ≥200 output tokens for good descriptions. 300 is safe ceiling.
- **`temperature: 0.1`** (vision), **`0.3`** (categorize): the vision model is
  near-deterministic; the categorizer runs slightly hotter to produce a fuller,
  more varied set of expressions.
- **`image.maxSide: 512`**: Image resize constraint. Larger = better detail
  (especially for text-heavy images) but slower and higher KV-cache pressure.
  The DRY/repetition samplers (below) mitigate the looping that larger images
  can trigger. See the Performance reference for the CPU/GPU tradeoff at
  smaller sizes.

### Vision sampling knobs (anti-repetition)

The vision model can loop / hallucinate on text-heavy images. `VisionPrompt`
exposes a stack of sampler knobs, each forwarded to Kronk only when enabled so
the SDK keeps its own defaults otherwise:

- **`dryMultiplier: 3.0`, `dryBase: 1.75`, `dryAllowedLength: 2`** — DRY
  ("Don't Repeat Yourself") sampler. Penalizes repeated *n-gram sequences*
  (multi-word phrase loops) with exponentially growing cost. Chosen over plain
  `repeatPenalty` because the pathology is a multi-token phrase cycle, not
  single-token spam, and DRY only penalizes actual repeated n-grams
  (length > `dryAllowedLength`), so an aggressive multiplier is safe for normal
  images. `1.05` (the SDK floor) had no measurable effect; `3.0` (top of the
  "aggressive" band) is needed to break the model's high-confidence repeat
  priors. `dryMultiplier == 0` disables the sampler.
- **`repeatPenalty: 1.15`, `repeatLastN: 64`** — blunt per-token fallback for
  single-token spam DRY's n-gram matching can miss. `<= 1.0` disables it;
  `repeatLastN <= 0` uses the SDK default (64).
- **`frequencyPenalty: 0.5`, `presencePenalty` (0)** — penalize tokens by
  recurrence count / any prior appearance. Both `0` means disabled.

### Categorization config

- **`categorizeModel`** (`CategorizeModelConfig`): `contextWindow: 4096` (room
  for the prose description plus the JSON expressions), `nBatch: 2048`,
  `nUBatch: 512`, K/V cache `Q8_0`.
- **`categorizePrompt`** (`CategorizePrompt`): a system prompt instructing the
  model to return a JSON object `{"expressions": [...]}` of ~15 short (2–6 word)
  phrases covering multiple aspects (subjects, objects, actions, setting,
  colors, style, text, context), preferring relationship-preserving phrases
  over isolated words. Output is grammar-constrained (see Kronk SDK usage).

## Build and test

All commands use `CGO_ENABLED=0` (the Kronk SDK uses FFI, not cgo).

```
make setup                         # Download llama.cpp libs + models
make build                         # Build without console (production)
make build-logs                    # Build with console output
make test-unit                     # Fast unit tests, no model loading
make test-service                  # Integration tests (loads models)
make test-performance-vision       # Vision model benchmarks
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
| `sdk/kronk` | `description/`, `categorization/`, `embedding/`, `kronk/` (platform) | `kronk.New(model.Config)` loads a model. `kronk.Init()` initializes the runtime. `krn.Unload(ctx)` frees it. |
| `sdk/kronk/model` | `description/`, `categorization/`, `embedding/`, perf tests | `model.Config` for engine params (ContextWindow, NBatch, CacheType). `model.D` for chat/embed request data. `model.GGMLType` for cache quantization. `model.TextMessage` for chat messages. |
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

### Chat with grammar-constrained JSON (categorization)

The categorizer forces valid JSON out of a tiny 0.6B model by passing a JSON
schema. Kronk converts the schema to a GBNF grammar and a grammar sampler sets
invalid tokens to `-inf` logits, so the model can *only* emit a matching object.
This replaces fragile free-text parsing with reliable `json.Unmarshal`.

```go
var expressionsSchema = model.D{
    "type": "object",
    "properties": model.D{
        "expressions": model.D{"type": "array", "items": model.D{"type": "string"}},
    },
    "required": []string{"expressions"},
}

data := model.D{
    "messages":        messages,          // system + user (prose description)
    "temperature":     temperature,
    "max_tokens":      maxTokens,
    "enable_thinking": false,             // Qwen3: skip <think> blocks — faster, stable output
    "json_schema":     expressionsSchema, // grammar-constrained output
}
resp, err := krn.Chat(ctx, data)
// choice.Message.Content is guaranteed-valid JSON → parseExpressions()
```

Two SDK facts drove this design (discovered by reading the vendored SDK):
- **`enable_thinking: false`** is required for Qwen3-0.6B — otherwise it emits
  `<think>` reasoning blocks, making it slow and unreliable.
- **`json_schema`** grammar-constrained output makes parsing robust. Note the
  JSON-schema grammar **cannot express `maxLength`/`maxItems`**, so length caps
  (max 15 expressions, blank-trimming) are enforced in code (`parseExpressions`).

### Vision sampling parameters (anti-repetition)

The vision Chat request conditionally adds DRY and repetition-penalty keys
(`dry_multiplier`, `dry_base`, `dry_allowed_length`, `repeat_penalty`,
`repeat_last_n`, `frequency_penalty`, `presence_penalty`) — see the
Configuration section. Each key is sent only when its config value is enabled,
so disabled knobs fall back to SDK defaults.

### Embeddings API pattern

```go
data := model.D{
    "input":    prefixedText,
    "truncate": true,
}
resp, err := krn.Embeddings(ctx, data)
vector := resp.Data[0].Embedding  // []float32
```

Each image produces up to 15 embeddings (one per expression); the query
produces one. LocalLens prefixes indexed expressions as retrieval documents
(`task: search result | document: ...`) and user searches as retrieval queries
(`task: search result | query: ...`) before calling Kronk. embeddinggemma is
~ms per call, so the extra embeds are negligible.

### Model lifecycle

The three models have different lifetimes by design:

- **Embedder** (~320 MB) — loaded once at `service.New()` (app startup) and
  kept resident for the lifetime of the `Service`. Every Search needs it to
  vectorize the query, and indexing reuses it to embed each expression.
  Holding it costs ~320 MB but eliminates repeated load/unload overhead.
- **Vision model** (~935 MB) and **Categorizer** (~805 MB, Qwen3-0.6B Q8_0) —
  both loaded at the start of each `IndexFolder` call and unloaded (via
  `defer`) when the call returns. They are only needed during indexing, and are
  too large to hold permanently on low-RAM machines. Peak indexing RAM ≈ 2 GB
  (vision + embedder + categorizer), within the 8 GB target.

```
service.New(ctx):
  → embedder.Load()          # stays loaded for app lifetime

service.IndexFolder(ctx, folderPath, progress):
  → countNewImages           # skip-already-indexed pre-pass
  → (short-circuit if none)  # avoids paying any model cost
  → describer.Load()         # vision model
  → categorizer.Load()       # Qwen3-0.6B chat model
  → for each new image:
      describer.Describe()   # vision: image → prose
      categorizer.Categorize()# prose → []expressions (grammar-constrained JSON)
      embedExpressions()     # embed each expression (embedder already loaded)
      index.Add() + Save()   # durable per-image
  → categorizer.Unload()     # via defer, even on cancellation
  → describer.Unload()       # via defer, even on cancellation

service.Close(ctx):
  → describer.Unload()
  → categorizer.Unload()
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
in the folder's `.locallens.index`, the full describe → categorize → embed →
save cycle runs end-to-end before moving to the next image, so a crash mid-run
loses at most the in-flight image.

```
service.IndexFolder(ctx, folderPath, progress):
  for each new image:
    → image.Resize(path, maxSide)          # Resize to maxSide px, JPEG encode
    → description.Describe(ctx, path)      # Vision model: image → prose
    → categorization.Categorize(ctx, prose)# Chat model: prose → []expressions
    → embedExpressions(ctx, expressions)   # Embedder: each expression → []float32
    → index.Add(Entry{Path, Description, Embeddings: []ExpressionEmbedding})
    → index.Save()                         # Write .locallens.index in the folder (gob)
    → tracker.record(...)                  # Progress callback (SSE event)

service.Search(ctx, folderPath, query, k):
  → embedding.Embed(ctx, embedding.Query, query) # Prefixed query text → single vector
  → loadIndex(folderPath)                # Cached after first call; disk hit
                                         # only on cache miss
  → search.FindTopK(queryVec, entries, k)# Multi-vector cosine + aggregate ranking
```

The embedder is always loaded (see Model lifecycle), so search is just one
embed call plus an in-memory similarity pass on cached entries. The vision and
categorization models are loaded once per `IndexFolder` call and unloaded via
`defer` on the way out — including on cancellation.

## Search vectors and ranking (the interesting part)

Aside from wrapping Kronk, this is what makes LocalLens's results good. The
design evolved through several tested iterations:

### Why categorize prose into expressions

The vision model emits a free-form paragraph. Embedding that whole blob into a
single vector mixes many concepts together and dilutes any single one, so
queries match poorly. Instead, a small grammar-constrained chat model reshapes
the prose into a flat list of **short, self-contained "search expressions"**
(2–6 words each, e.g. `"bright yellow parrot"`, `"dense tropical forest"`).
Each expression is a coherent unit that embeds cleanly.

Earlier iterations used a fixed **4-facet** structure
(scene/objects/actions/attributes); this was replaced by a flat expression list
because the fixed categories were arbitrary and the flat list is simpler to
generate, embed, and rank. The `Description` (prose) is still stored per entry
as an audit trail (prose → expressions → vectors).

### Multi-vector index

`index.Entry` stores `Embeddings []ExpressionEmbedding` — **one vector per
expression** (max 15), not one vector per image. Expressions are embedded as
retrieval documents via `embedding.Embed(ctx, embedding.Document, expression)`;
search queries use `embedding.Query`. This costs ~15× embeds at index time and
~15× cosines at search time, but embeddinggemma is fast and folders are small,
so it is negligible. Entries with `len(Embeddings) == 0` are treated as "not
indexed" and re-index automatically after the gob format changed — no manual
`.locallens.index` cleanup needed.

### Aggregate ranking (`search.aggregate`)

`FindTopK` computes the query's cosine similarity against **every** expression
vector of an image, sorts those scores, keeps the **top 5** expression scores,
records the sorted per-expression scores in `Result.ExpressionScores` (for
auditing why an image ranked where it did), then combines the top scores into
one image score:

```go
topScores = best 5 expression similarities for the image
aggregate(topScores) = mean(topScores)*0.5 + max(topScores)*0.5
```

Using only the top 5 prevents low-scoring unrelated expressions from diluting a
good match. The `0.5/0.5` split is the tested tradeoff and the tunable "search
specialization" knob:
- **Pure sum** (first attempt) rewarded images that simply populate more
  expressions, hurting precision (a richly-described image outranks a precise
  2-expression match).
- **Pure mean over all expressions** (second attempt) fixed that but
  under-weighted a single strong match diluted by unrelated expressions.
- **`mean*0.5 + max*0.5`** (current) keeps a strong single-expression match
  relevant (via `max`) while still rewarding broad coverage (via `mean`).
  A UI knob to expose this rate is on the roadmap.

`aggregate()` is a small pure function, so alternative weightings can be
A/B-tested against the `search_test.go` fixtures without touching the pipeline.

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

1. **Always-loaded embedder, on-demand vision + categorization models** — see
   Model lifecycle.
2. **In-memory index cache** — `Service.indexes` (`map[string]*index.Index`,
   guarded by `sync.RWMutex`) caches the `.locallens.index` deserialized from
   each folder. First search hits disk; subsequent searches and re-indexes
   on the same folder are pure in-memory. Because `idx.Add()`/`idx.Save()`
   mutate the cached pointer in place, no separate cache-write step is needed
   after indexing — the cache stays consistent automatically.
3. **Per-image durability** — `indexFolder()` runs describe → categorize →
   embed → `idx.Add` → `idx.Save` for each image. A crash mid-run loses at most
   the in-flight image. On resume, `idx.Get(imgPath)` short-circuits
   already-saved entries.
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
- `Stage: "failed"` — fired when describe/categorize/embed returns an error. The image is
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
