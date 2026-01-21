// Package main provides the entry point for LocalLens, a local semantic
// image search application.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ramon-reichert/locallens/internal/indexer"
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

	idx, err := indexer.New(indexer.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
	})
	if err != nil {
		return fmt.Errorf("new indexer: %w", err)
	}
	defer idx.Close(ctx)

	log(ctx, "indexer ready")

	return nil
}
