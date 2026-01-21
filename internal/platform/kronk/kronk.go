// Package kronk provides Kronk SDK setup and model management for LocalLens.
package kronk

import (
	"context"
	"fmt"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

const (
	VisionModelURL = "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
	VisionProjURL  = "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
	EmbedModelURL  = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf"
)

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

	// Don't need a template system for this app
	// tmpl, err := templates.New()
	// if err != nil {
	// 	return fmt.Errorf("templates new: %w", err)
	// }

	// if err := tmpl.Download(ctx); err != nil {
	// 	return fmt.Errorf("templates download: %w", err)
	// }

	// if err := tmpl.Catalog().Download(ctx); err != nil {
	// 	return fmt.Errorf("catalog download: %w", err)
	// }

	return nil
}

// DownloadModels downloads vision and embedding models.
func DownloadModels(ctx context.Context, log logger.Logger) (ModelPaths, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	log(ctx, "downloading models")

	mdls, err := models.New()
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

// Init initializes the Kronk runtime. Must be called once before loading models.
func Init() error {
	return kronk.Init()
}

// VisionConfig returns a model.Config suitable for vision inference.
func VisionConfig(mp models.Path) model.Config {
	return model.Config{
		ModelFiles:    mp.ModelFiles,
		ProjFile:      mp.ProjFile,
		ContextWindow: 8192,
		NBatch:        2048,
		NUBatch:       2048,
		CacheTypeK:    model.GGMLTypeQ8_0,
		CacheTypeV:    model.GGMLTypeQ8_0,
	}
}

// EmbedConfig returns a model.Config suitable for embedding inference.
func EmbedConfig(mp models.Path) model.Config {
	return model.Config{
		ModelFiles:     mp.ModelFiles,
		ContextWindow:  2048,
		NBatch:         2048,
		NUBatch:        512,
		CacheTypeK:     model.GGMLTypeQ8_0,
		CacheTypeV:     model.GGMLTypeQ8_0,
		FlashAttention: model.FlashAttentionEnabled,
	}
}

// LoadModel loads a model with the given configuration.
func LoadModel(cfg model.Config) (*kronk.Kronk, error) {
	krn, err := kronk.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("kronk new: %w", err)
	}
	return krn, nil
}
