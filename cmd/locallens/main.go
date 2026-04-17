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
	if err := kronksdk.Init(); err != nil {
		log(ctx, "kronk init failed, setup may be needed", "error", err)
	} else if s, err := initService(log, cfg); err != nil {
		log(ctx, "service init deferred, setup may be needed", "error", err)
	} else {
		svc = s
	}

	handlers := app.New(app.Config{
		Log:     log,
		Service: svc,
		SetupStatus: func() app.SetupStatusInfo {
			c, _ := config.Load()
			return app.SetupStatusInfo{
				BasePath:    c.BasePath,
				DefaultPath: config.DefaultBasePath(),
			}
		},
		SetupRunner: func(ctx context.Context, log logger.Logger, basePath string, progress app.SetupProgress) (*service.Service, error) {
			cfg, _ := config.Load()
			cfg.BasePath = basePath

			progress("libs", "downloading")
			if err := kronk.InstallDependencies(ctx, log, cfg); err != nil {
				progress("libs", "error: "+err.Error())
				return nil, err
			}
			progress("libs", "complete")

			progress("models", "downloading")
			if err := kronksdk.Init(); err != nil {
				progress("models", "error: "+err.Error())
				return nil, err
			}
			if _, err := kronk.DownloadModels(ctx, log, cfg); err != nil {
				progress("models", "error: "+err.Error())
				return nil, err
			}
			progress("models", "complete")

			progress("init", "initializing")
			svc, err := initService(log, cfg)
			if err != nil {
				progress("init", "error: "+err.Error())
				return nil, err
			}

			config.Save(cfg)

			return svc, nil
		},
	})
	defer handlers.Close(ctx)

	mux := http.NewServeMux()

	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return fmt.Errorf("static fs: %w", err)
	}

	handlers.Register(mux, staticFS)

	srv := web.New(web.Config{
		Log:  log,
		Mux:  mux,
		Host: "localhost",
		Port: "8080",
	})

	return srv.ListenAndServe()
}

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
