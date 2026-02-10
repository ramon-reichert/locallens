// Package main provides the entry point for LocalLens, a local semantic
// image search application.
package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"net/http"
	"os"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"

	"github.com/ramon-reichert/locallens/internal/app"
	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/platform/web"
	"github.com/ramon-reichert/locallens/internal/service"
)

//go:embed static
var staticFiles embed.FS

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()
	log := logger.New()

	// Resolve model paths (models must be downloaded via 'make setup').
	paths, err := kronk.ResolvePaths(`B:/dev/kronkdata`)
	if err != nil {
		return fmt.Errorf("resolve paths: %w", err)
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
	svc := service.New(service.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
	})
	defer svc.Close(ctx)

	log(ctx, "service ready")

	// Setup HTTP routes.
	mux := http.NewServeMux()

	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return fmt.Errorf("static fs: %w", err)
	}

	handlers := app.New(log, svc)
	handlers.Register(mux, staticFS)

	// Start server.
	srv := web.New(web.Config{
		Log:  log,
		Mux:  mux,
		Host: "localhost",
		Port: "8080",
	})

	log(ctx, "open http://localhost:8080 in your browser")

	return srv.ListenAndServe()
}
