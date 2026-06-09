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

// version is the build-time version string. It defaults to "dev" for local
// builds between releases and is overridden by GoReleaser at release time via
// -ldflags "-X main.version={{.Version}}".
var version = "dev"

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()
	log := logger.New()

	log(ctx, "locallens starting", "version", version)

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
	} else {
		logSystemInfo(ctx, log, cfg)
		if s, err := initService(ctx, log, cfg); err != nil {
			log(ctx, "service init deferred, setup may be needed", "error", err)
		} else {
			svc = s
		}
	}

	// Wire dependencies into the handler layer.
	handlers := app.New(app.Config{
		Log:         log,
		Version:     version,
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
		Host: "127.0.0.1",
		Port: "0",
	})

	return server.ListenAndServe()
}

// setupStatus returns the current setup state for the UI.
// Re-reads config each call to reflect any changes from a completed setup.
func setupStatus() app.SetupStatusInfo {
	c, _ := config.Load()
	return app.SetupStatusInfo{
		BasePath:          c.BasePath,
		DefaultPath:       config.DefaultBasePath(),
		Processor:         c.Processor,
		DetectedProcessor: kronk.DetectedProcessor(),
		ActiveProcessor:   kronk.ActiveProcessor(),
	}
}

// setupRunner orchestrates the full setup flow: install llama.cpp libraries,
// initialize the Kronk SDK, download models, and create the service.
// Progress is reported to the caller via SSE through the progress callback.
//
// The Kronk SDK loads the llama.cpp shared library exactly once per process,
// so when the chosen processor differs from the one already resident in this
// process, setupRunner installs the new libs and saves config but skips the
// in-process Service rebuild — the new backend only takes effect on the
// next launch. In that case it returns (nil, nil) after emitting a
// "restart_required" progress event.
func setupRunner(ctx context.Context, log logger.Logger, req app.SetupRequest, progress app.SetupProgress) (*service.Service, error) {
	cfg, _ := config.Load()

	// Remember what was persisted so we can decide whether the new
	// request changes the processor field in config.json.
	savedProcessor := cfg.Processor

	cfg.BasePath = req.BasePath
	cfg.Processor = req.Processor

	// Any processor change in the saved config requires a restart, because
	// the Kronk SDK can't swap llama.cpp libraries mid-process. We compare
	// against the saved config (not the active runtime) so a user who
	// already changed processor in a prior session without restarting
	// still sees consistent UI. On a true first install there is no
	// resident Kronk runtime yet, so no restart is needed — we let the
	// Service initialize in-process below.
	processorSwitch := kronk.ActiveProcessor() != "" &&
		req.Processor != "" &&
		req.Processor != savedProcessor

	progress("libs", "downloading")
	if err := kronk.InstallDependencies(ctx, log, cfg); err != nil {
		log(ctx, "setup: install dependencies failed", "error", err)
		progress("libs", "error: "+err.Error())
		return nil, err
	}
	progress("libs", "complete")

	progress("models", "downloading")
	if err := kronk.Init(cfg); err != nil {
		log(ctx, "setup: kronk init failed", "error", err)
		progress("models", "error: "+err.Error())
		return nil, err
	}
	if _, err := kronk.DownloadModels(ctx, log, cfg); err != nil {
		log(ctx, "setup: download models failed", "error", err)
		progress("models", "error: "+err.Error())
		return nil, err
	}
	progress("models", "complete")

	if processorSwitch {
		if err := config.Save(cfg); err != nil {
			log(ctx, "setup: save config failed", "error", err)
			progress("save", "error: "+err.Error())
			return nil, err
		}
		log(ctx, "processor changed", "config saved")
		progress("restart_required", "Restart LocalLens to apply the processor change.")
		return nil, nil
	}

	progress("service", "initializing")
	svc, err := initService(ctx, log, cfg)
	if err != nil {
		log(ctx, "setup: service init failed", "error", err)
		progress("service", "error: "+err.Error())
		return nil, err
	}

	if err := config.Save(cfg); err != nil {
		log(ctx, "setup: save config failed", "error", err)
	}

	log(ctx, "setup finished", "config saved")

	return svc, nil
}

// logSystemInfo emits a one-time log of the active processor backend and the
// installed llama.cpp library build (version/arch/os/processor). Called once
// after a successful kronk.Init so the reported triple matches the loaded
// runtime. Failures are logged but non-fatal.
func logSystemInfo(ctx context.Context, log logger.Logger, cfg config.Config) {
	info, err := kronk.Info(cfg)
	if err != nil {
		log(ctx, "system info unavailable", "error", err)
		return
	}

	log(ctx, "system info",
		"processor", kronk.ActiveProcessor(),
		"llamaCppVersion", info.Version,
		"arch", info.Arch,
		"os", info.OS,
	)
}

// initService resolves model file paths and creates the Service.
func initService(ctx context.Context, log logger.Logger, cfg config.Config) (*service.Service, error) {
	paths, err := kronk.ResolvePaths(cfg)
	if err != nil {
		return nil, err
	}

	return service.New(ctx, service.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
		AppCfg:      cfg,
	})
}
