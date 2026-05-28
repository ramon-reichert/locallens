# LocalLens

LocalLens is a local semantic image search application. It uses a vision model
(Qwen2-VL-2B) to describe images, an embedding model
(embeddinggemma-300m) to vectorize those descriptions, and cosine similarity to
find images matching a natural language query. Everything runs locally — no
cloud APIs.

## Kronk SDK usage

LocalLens is built on top of the [Kronk SDK](https://github.com/ardanlabs/kronk), 
and serves as a practical demonstration of Kronk's capabilities for embedding AI local inference into Go applications.

| Kronk package | Where used | Purpose |
|---|---|---|
| `sdk/kronk` | `description/`, `embedding/`, `kronk/` (platform) | `kronk.New(model.Config)` loads a model. `kronk.Init()` initializes the runtime. `krn.Unload(ctx)` frees it. |
| `sdk/kronk/model` | `description/`, `embedding/`, perf tests | `model.Config` for engine params (ContextWindow, NBatch, CacheType). `model.D` for chat/embed request data. `model.GGMLType` for cache quantization. |
| `sdk/tools/libs` | `kronk/` (platform) | `libs.New()` + `libs.Download()` to fetch llama.cpp shared libraries. `libs.WithVersion()` pins a specific build. |
| `sdk/tools/models` | `kronk/` (platform) | `models.NewWithPaths()` + `models.Download()` / `models.FullPath()` to download and resolve GGUF model files. |
| `sdk/tools/devices` | perf tests | `devices.List()` for GPU detection (name, type, VRAM, system RAM). |
| `sdk/tools/defaults` | `kronk/` (platform) | `defaults.LibVersion()` resolves pinned vs latest llama.cpp version. `defaults.Processor()` parses GPU backend names. |
