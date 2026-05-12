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

	"github.com/ramon-reichert/locallens/internal/app"
	"github.com/ramon-reichert/locallens/internal/platform/config"
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

	cfg, err := config.Load()
	if err != nil {
		log(ctx, "config warning, using defaults", "error", err)
	}

	// Initialize the Kronk SDK runtime, then try to create the service.
	// If models aren't downloaded yet, this fails gracefully and the
	// handler layer returns 503 until setup completes.
	var svc *service.Service
	if err := kronk.Init(cfg); err != nil {
		log(ctx, "kronk init failed, setup may be needed", "error", err)
	} else if s, err := initService(log, cfg); err != nil {
		log(ctx, "service init deferred, setup may be needed", "error", err)
	} else {
		svc = s
	}

	// Wire dependencies into the handler layer.
	handlers := app.New(app.Config{
		Log:         log,
		Service:     svc,
		SetupStatus: setupStatus,
		SetupRunner: setupRunner,
	})
	defer handlers.Close(ctx)

	// Serve embedded static files (HTML/CSS/JS) and API routes.
	mux := http.NewServeMux()

	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return fmt.Errorf("static fs: %w", err)
	}

	handlers.Register(mux, staticFS)

	server := web.New(web.Config{
		Log:  log,
		Mux:  mux,
		Host: "localhost",
		Port: "8080",
	})

	return server.ListenAndServe()
}

// setupStatus returns the current setup state for the UI.
// Re-reads config each call to reflect any changes from a completed setup.
func setupStatus() app.SetupStatusInfo {
	c, _ := config.Load()
	return app.SetupStatusInfo{
		BasePath:    c.BasePath,
		DefaultPath: config.DefaultBasePath(),
	}
}

// setupRunner orchestrates the full setup flow: install llama.cpp libraries,
// initialize the Kronk SDK, download models, and create the service.
// Progress is reported to the caller via SSE through the progress callback.
func setupRunner(ctx context.Context, log logger.Logger, basePath string, progress app.SetupProgress) (*service.Service, error) {
	cfg, _ := config.Load()
	cfg.BasePath = basePath

	progress("libs", "downloading")
	if err := kronk.InstallDependencies(ctx, log, cfg); err != nil {
		progress("libs", "error: "+err.Error())
		return nil, err
	}
	progress("libs", "complete")

	progress("models", "downloading")
	if err := kronk.Init(cfg); err != nil {
		progress("models", "error: "+err.Error())
		return nil, err
	}
	if _, err := kronk.DownloadModels(ctx, log, cfg); err != nil {
		progress("models", "error: "+err.Error())
		return nil, err
	}
	progress("models", "complete")

	progress("service", "initializing")
	svc, err := initService(log, cfg)
	if err != nil {
		progress("service", "error: "+err.Error())
		return nil, err
	}

	config.Save(cfg)

	return svc, nil
}

// initService resolves model file paths and creates the Service.
func initService(log logger.Logger, cfg config.Config) (*service.Service, error) {
	paths, err := kronk.ResolvePaths(cfg)
	if err != nil {
		return nil, err
	}

	return service.New(service.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
		AppCfg:      cfg,
	}), nil
}
