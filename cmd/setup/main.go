// Command setup downloads llama.cpp libraries and model files.
// Used by `make setup` to avoid duplicating model URLs in the makefile.
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "setup: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
	defer cancel()

	log := logger.New()

	cfg := config.Load()
	if v := os.Getenv("KRONK_BASE_PATH"); v != "" {
		cfg.BasePath = v
	}

	log(ctx, "installing llama.cpp libraries")
	if err := kronk.InstallDependencies(ctx, log); err != nil {
		return err
	}

	log(ctx, "initializing kronk")
	if err := kronksdk.Init(); err != nil {
		return err
	}

	log(ctx, "downloading models", "basePath", cfg.BasePath)
	if _, err := kronk.DownloadModels(ctx, log, cfg); err != nil {
		return err
	}

	log(ctx, "setup complete")
	return nil
}
