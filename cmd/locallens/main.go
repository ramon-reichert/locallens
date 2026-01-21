// Package main provides the entry point for LocalLens, a local semantic
// image search application using the Kronk SDK.
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
	"github.com/ardanlabs/kronk/sdk/tools/templates"
)

const (
	visionModelURL = "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
	visionProjURL  = "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
	embedModelURL  = "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf"
)

// Logger provides a function for logging messages.
type Logger func(ctx context.Context, msg string, args ...any)

// log is the package-level logger.
var log Logger = func(ctx context.Context, msg string, args ...any) {
	fmt.Printf("%s:", msg)
	for i := 0; i < len(args); i += 2 {
		if i+1 < len(args) {
			fmt.Printf(" %v[%v]", args[i], args[i+1])
		}
	}
	fmt.Println()
}

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()

	if err := installDependencies(ctx); err != nil {
		return fmt.Errorf("install dependencies: %w", err)
	}

	visionPath, embedPath, err := downloadModels(ctx)
	if err != nil {
		return fmt.Errorf("download models: %w", err)
	}

	log(ctx, "models ready",
		"vision", visionPath.ModelFiles[0],
		"embed", embedPath.ModelFiles[0],
	)

	if err := testVision(ctx, visionPath); err != nil {
		return fmt.Errorf("test vision: %w", err)
	}

	return nil
}

func installDependencies(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()

	log(ctx, "installing dependencies")

	libsys, err := libs.New(libs.WithVersion(defaults.LibVersion("")))
	if err != nil {
		return fmt.Errorf("libs new: %w", err)
	}

	if _, err := libsys.Download(ctx, kronk.FmtLogger); err != nil {
		return fmt.Errorf("libs download: %w", err)
	}

	tmpl, err := templates.New()
	if err != nil {
		return fmt.Errorf("templates new: %w", err)
	}

	if err := tmpl.Download(ctx); err != nil {
		return fmt.Errorf("templates download: %w", err)
	}

	if err := tmpl.Catalog().Download(ctx); err != nil {
		return fmt.Errorf("catalog download: %w", err)
	}

	return nil
}

func downloadModels(ctx context.Context) (vision, embed models.Path, err error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	log(ctx, "downloading models")

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("models new: %w", err)
	}

	log(ctx, "downloading vision model")
	vision, err = mdls.Download(ctx, kronk.FmtLogger, visionModelURL, visionProjURL)
	if err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("vision download: %w", err)
	}

	log(ctx, "downloading embedding model")
	embed, err = mdls.Download(ctx, kronk.FmtLogger, embedModelURL, "")
	if err != nil {
		return models.Path{}, models.Path{}, fmt.Errorf("embed download: %w", err)
	}

	return vision, embed, nil
}

func testVision(ctx context.Context, mp models.Path) error {
	log(ctx, "testing vision model")

	if err := kronk.Init(); err != nil {
		return fmt.Errorf("kronk init: %w", err)
	}

	krn, err := kronk.New(model.Config{
		ModelFiles:    mp.ModelFiles,
		ProjFile:      mp.ProjFile,
		ContextWindow: 8192,
		NBatch:        2048,
		NUBatch:       2048,
		CacheTypeK:    model.GGMLTypeQ8_0,
		CacheTypeV:    model.GGMLTypeQ8_0,
	})
	if err != nil {
		return fmt.Errorf("kronk new: %w", err)
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
