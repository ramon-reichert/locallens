// Package main provides the entry point for LocalLens, a local semantic
// image search application.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()
	log := logger.New()

	if err := kronk.InstallDependencies(ctx, log); err != nil {
		return fmt.Errorf("install dependencies: %w", err)
	}

	paths, err := kronk.DownloadModels(ctx, log)
	if err != nil {
		return fmt.Errorf("download models: %w", err)
	}

	log(ctx, "models ready",
		"vision", paths.Vision.ModelFiles[0],
		"embed", paths.Embed.ModelFiles[0],
	)

	if err := kronk.Init(); err != nil {
		return fmt.Errorf("kronk init: %w", err)
	}

	log(ctx, "testing vision model")

	krn, err := kronk.LoadModel(kronk.VisionConfig(paths.Vision))
	if err != nil {
		return fmt.Errorf("load vision model: %w", err)
	}

	defer func() {
		log(ctx, "unloading vision model")
		if err := krn.Unload(context.Background()); err != nil {
			log(ctx, "unload error", "error", err)
		}
	}()

	log(ctx, "vision model loaded",
		"context_window", krn.ModelConfig().ContextWindow,
		"template", krn.ModelInfo().Template.FileName,
	)

	return nil
}
