// Package main provides the entry point for LocalLens, a local semantic
// image search application.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"

	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service"
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

	// Install dependencies and download models.
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

	// Initialize Kronk runtime.
	if err := kronksdk.Init(); err != nil {
		return fmt.Errorf("kronk init: %w", err)
	}

	// Create service.
	homeDir, _ := os.UserHomeDir()
	indexPath := filepath.Join(homeDir, ".locallens", "index.gob")

	if err := os.MkdirAll(filepath.Dir(indexPath), 0755); err != nil {
		return fmt.Errorf("create data dir: %w", err)
	}

	svc := service.New(service.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
		IndexPath:   indexPath,
	})
	defer svc.Close(ctx)

	if err := svc.Index.Load(); err != nil {
		return fmt.Errorf("load index: %w", err)
	}

	log(ctx, "service ready", "indexed_images", svc.Index.Len())

	// TODO: Start UI

	return nil
}
