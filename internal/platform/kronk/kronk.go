// Package kronk provides Kronk SDK setup and model management for LocalLens.
package kronk

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

// ModelPaths holds the resolved file paths for all models.
type ModelPaths struct {
	Vision config.ModelFilePaths
	Embed  config.ModelFilePaths
}

// Init initializes the Kronk SDK runtime. Must be called before loading
// any models.
func Init() error {
	return kronk.Init()
}

// InstallDependencies downloads llama.cpp libraries, templates, and catalog.
// It uses cfg.Processor to select the correct GPU backend (e.g., "cuda",
// "vulkan", "metal", "cpu"). When empty, auto-detects GPU hardware.
// The KRONK_PROCESSOR env var takes precedence if set.
func InstallDependencies(ctx context.Context, log logger.Logger, cfg config.Config) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()

	// Config processor is used when KRONK_PROCESSOR env var is not set.
	processor := cfg.Processor
	if v := os.Getenv("KRONK_PROCESSOR"); v != "" {
		processor = v
	}

	log(ctx, "installing dependencies", "processor", processor)

	opts := []libs.Option{libs.WithVersion(defaults.LibVersion(cfg.LlamaCppVersion))}

	if processor != "" {
		p, err := defaults.Processor(processor)
		if err != nil {
			return fmt.Errorf("parse processor %q: %w", processor, err)
		}
		opts = append(opts, libs.WithProcessor(p))
	}

	libsys, err := libs.New(opts...)
	if err != nil {
		return fmt.Errorf("llama.cpp libs new: %w", err)
	}

	if _, err := libsys.Download(ctx, kronk.FmtLogger); err != nil {
		return fmt.Errorf("llama.cpp libs download: %w", err)
	}

	return nil
}

// ResolvePaths resolves the file paths for already-downloaded models.
// Models must be downloaded first.
func ResolvePaths(cfg config.Config) (ModelPaths, error) {
	mdls, err := models.NewWithPaths(cfg.BasePath)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("models new: %w", err)
	}

	vision, err := mdls.FullPath(cfg.ModelsURLs.VisionModelID())
	if err != nil {
		return ModelPaths{}, fmt.Errorf("resolve vision model: %w", err)
	}

	embed, err := mdls.FullPath(cfg.ModelsURLs.EmbedModelID())
	if err != nil {
		return ModelPaths{}, fmt.Errorf("resolve embed model: %w", err)
	}

	return ModelPaths{Vision: toModelFilePaths(vision), Embed: toModelFilePaths(embed)}, nil
}

// DownloadModels downloads vision and embedding models.
func DownloadModels(ctx context.Context, log logger.Logger, cfg config.Config) (ModelPaths, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	log(ctx, "downloading models")

	mdls, err := models.NewWithPaths(cfg.BasePath)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("models new: %w", err)
	}

	log(ctx, "downloading vision model")
	vision, err := mdls.Download(ctx, kronk.FmtLogger, cfg.ModelsURLs.VisionModelURL, cfg.ModelsURLs.VisionProjURL)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("vision download: %w", err)
	}

	log(ctx, "downloading embedding model")
	embed, err := mdls.Download(ctx, kronk.FmtLogger, cfg.ModelsURLs.EmbedModelURL, "")
	if err != nil {
		return ModelPaths{}, fmt.Errorf("embed download: %w", err)
	}

	return ModelPaths{Vision: toModelFilePaths(vision), Embed: toModelFilePaths(embed)}, nil
}

func toModelFilePaths(p models.Path) config.ModelFilePaths {
	return config.ModelFilePaths{
		ModelFiles: p.ModelFiles,
		ProjFile:   p.ProjFile,
	}
}
