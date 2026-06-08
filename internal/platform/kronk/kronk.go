// Package kronk provides Kronk SDK setup and model management for LocalLens.
package kronk

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/devices"
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

// SystemInfo describes the llama.cpp library build resident in this process.
type SystemInfo struct {
	Version   string // llama.cpp library version (e.g. "b9247")
	Arch      string // CPU architecture (e.g. "amd64")
	OS        string // operating system (e.g. "windows")
	Processor string // backend processor (e.g. "cuda", "vulkan", "cpu")
}

// Info reports the installed llama.cpp library metadata for the processor
// selected by cfg (or KRONK_PROCESSOR). It reads the version.json of the
// resolved per-triple install, mirroring how Init selects its library path.
// Call after Init so the reported triple matches the loaded runtime.
func Info(cfg config.Config) (SystemInfo, error) {
	opts := []libs.Option{libs.WithBasePath(cfg.BasePath)}

	if processor := resolveProcessor(cfg); processor != "" {
		p, err := defaults.Processor(processor)
		if err != nil {
			return SystemInfo{}, fmt.Errorf("parse processor %q: %w", processor, err)
		}
		opts = append(opts, libs.WithProcessor(p))
	}

	libsys, err := libs.New(opts...)
	if err != nil {
		return SystemInfo{}, fmt.Errorf("llama.cpp libs new: %w", err)
	}

	tag, err := libsys.InstalledVersion()
	if err != nil {
		return SystemInfo{}, fmt.Errorf("read installed libs version: %w", err)
	}

	return SystemInfo{
		Version:   tag.Version,
		Arch:      tag.Arch,
		OS:        tag.OS,
		Processor: tag.Processor,
	}, nil
}

// activeProcessor records the processor name that was actually loaded into
// the process by the first successful kronk.Init call. The Kronk SDK loads
// the llama.cpp shared library exactly once per process and short-circuits
// subsequent Init calls, so this value cannot change at runtime — switching
// processors requires restarting the binary.
var activeProcessor string

// ActiveProcessor returns the processor backend currently resident in the
// process (e.g. "cuda", "vulkan", "cpu"), or "" if Kronk has not been
// initialized yet.
func ActiveProcessor() string {
	return activeProcessor
}

// Init initializes the Kronk SDK runtime. Must be called before loading
// any models. When cfg.Processor (or KRONK_PROCESSOR) selects a backend,
// the corresponding library path is pinned so the runtime does not auto-
// detect a different processor (e.g. Vulkan) at load time.
func Init(cfg config.Config) error {
	processor := resolveProcessor(cfg)
	if processor == "" {
		libsys, err := libs.New(libs.WithBasePath(cfg.BasePath))
		if err != nil {
			return fmt.Errorf("llama.cpp libs new: %w", err)
		}

		if err := kronk.Init(kronk.WithLibPath(libsys.LibsPath())); err != nil {
			return err
		}
		if activeProcessor == "" {
			activeProcessor = "cpu"
		}
		return nil
	}

	p, err := defaults.Processor(processor)
	if err != nil {
		return fmt.Errorf("parse processor %q: %w", processor, err)
	}

	libsys, err := libs.New(libs.WithBasePath(cfg.BasePath), libs.WithProcessor(p))
	if err != nil {
		return fmt.Errorf("llama.cpp libs new: %w", err)
	}

	if err := kronk.Init(kronk.WithLibPath(libsys.LibsPath())); err != nil {
		return err
	}
	if activeProcessor == "" {
		activeProcessor = processor
	}
	return nil
}

// InstallDependencies downloads llama.cpp libraries, templates, and catalog.
// It uses cfg.Processor to select the correct GPU backend (e.g., "cuda",
// "vulkan", "metal", "cpu"). When empty, auto-detects GPU hardware.
// The KRONK_PROCESSOR env var takes precedence if set.
func InstallDependencies(ctx context.Context, log logger.Logger, cfg config.Config) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()

	processor := resolveProcessor(cfg)

	log(ctx, "installing dependencies", "processor", processor)

	opts := []libs.Option{
		libs.WithBasePath(cfg.BasePath),
		libs.WithVersion(defaults.LibVersion(cfg.LlamaCppVersion)),
	}

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

// resolveProcessor returns the configured processor name, with the
// KRONK_PROCESSOR env var taking precedence over cfg.Processor.
func resolveProcessor(cfg config.Config) string {
	if v := os.Getenv("KRONK_PROCESSOR"); v != "" {
		return v
	}
	return cfg.Processor
}

// DetectedProcessor returns the processor name Kronk has auto-detected for
// the current machine (e.g. "cuda", "vulkan", "metal", "rocm", or "cpu").
// It is a thin pass-through to the Kronk SDK so the SDK boundary stays
// inside this package.
func DetectedProcessor() string {
	return devices.DetectGPU().String()
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
	vision, err := mdls.DownloadURLs(ctx, kronk.FmtLogger, []string{cfg.ModelsURLs.VisionModelURL}, cfg.ModelsURLs.VisionProjURL)
	if err != nil {
		return ModelPaths{}, fmt.Errorf("vision download: %w", err)
	}

	log(ctx, "downloading embedding model")
	embed, err := mdls.DownloadURLs(ctx, kronk.FmtLogger, []string{cfg.ModelsURLs.EmbedModelURL}, "")
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
