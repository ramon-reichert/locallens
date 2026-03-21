// Package kronk provides Kronk SDK setup and model management for LocalLens.
package kronk

import (
	"context"
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

// Model download URLs. These are the single source of truth for model
// identifiers across the project. Update the makefile if these change.
const (
	VisionModelURL = "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
	VisionProjURL  = "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
	//VisionModelURL = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
	//VisionProjURL  = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf"
	EmbedModelURL = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf"
)

// Model IDs derived from URLs (filename without .gguf extension).
var (
	VisionModelID = modelIDFromURL(VisionModelURL)
	EmbedModelID  = modelIDFromURL(EmbedModelURL)
)

func modelIDFromURL(url string) string {
	name := path.Base(url)
	return strings.TrimSuffix(name, path.Ext(name))
}

// ModelPaths holds the paths to downloaded model files.
type ModelPaths struct {
	Vision models.Path
	Embed  models.Path
}

// InstallDependencies downloads llama.cpp libraries, templates, and catalog.
func InstallDependencies(ctx context.Context, log logger.Logger) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()

	log(ctx, "installing dependencies")

	libsys, err := libs.New(libs.WithVersion(defaults.LibVersion("")))
	if err != nil {
		return fmt.Errorf("llama.cpp libs new: %w", err)
	}

	if _, err := libsys.Download(ctx, kronk.FmtLogger); err != nil {
		return fmt.Errorf("llama.cpp libs download: %w", err)
	}

	return nil
}

// ResolvePaths resolves the file paths for already-downloaded models.
// Models must be downloaded first via `make setup`.
func ResolvePaths(basePath string) (ModelPaths, error) {
	mdls, err := models.NewWithPaths(basePath)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("models new: %w", err)
	}

	vision, err := mdls.FullPath(VisionModelID)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("resolve vision model: %w (run 'make setup' first)", err)
	}

	embed, err := mdls.FullPath(EmbedModelID)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("resolve embed model: %w (run 'make setup' first)", err)
	}

	return ModelPaths{Vision: vision, Embed: embed}, nil
}

// DownloadModels downloads vision and embedding models.
func DownloadModels(ctx context.Context, log logger.Logger, basePath string) (ModelPaths, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	log(ctx, "downloading models")

	mdls, err := models.NewWithPaths(basePath)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("models new: %w", err)
	}

	log(ctx, "downloading vision model")
	vision, err := mdls.Download(ctx, kronk.FmtLogger, VisionModelURL, VisionProjURL)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("vision download: %w", err)
	}

	log(ctx, "downloading embedding model")
	embed, err := mdls.Download(ctx, kronk.FmtLogger, EmbedModelURL, "")
	if err != nil {
		return ModelPaths{}, fmt.Errorf("embed download: %w", err)
	}

	return ModelPaths{Vision: vision, Embed: embed}, nil
}
