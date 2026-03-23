// Package kronk provides Kronk SDK setup and model management for LocalLens.
package kronk

import (
	"context"
	"fmt"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
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

	return nil
}

// ResolvePaths resolves the file paths for already-downloaded models.
// Models must be downloaded first via `make setup`.
func ResolvePaths(cfg config.Config) (ModelPaths, error) {
	mdls, err := models.NewWithPaths(cfg.BasePath)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("models new: %w", err)
	}

	vision, err := mdls.FullPath(cfg.Models.VisionModelID())
	if err != nil {
		return ModelPaths{}, fmt.Errorf("resolve vision model: %w (run 'make setup' first)", err)
	}

	embed, err := mdls.FullPath(cfg.Models.EmbedModelID())
	if err != nil {
		return ModelPaths{}, fmt.Errorf("resolve embed model: %w (run 'make setup' first)", err)
	}

	return ModelPaths{Vision: vision, Embed: embed}, nil
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
	vision, err := mdls.Download(ctx, kronk.FmtLogger, cfg.Models.VisionModelURL, cfg.Models.VisionProjURL)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("vision download: %w", err)
	}

	log(ctx, "downloading embedding model")
	embed, err := mdls.Download(ctx, kronk.FmtLogger, cfg.Models.EmbedModelURL, "")
	if err != nil {
		return ModelPaths{}, fmt.Errorf("embed download: %w", err)
	}

	return ModelPaths{Vision: vision, Embed: embed}, nil
}
