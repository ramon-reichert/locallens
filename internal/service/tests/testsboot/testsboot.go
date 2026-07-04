// Package testsboot provides the Boot to be run Once before any test or set of tests.
package testsboot

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	once            sync.Once
	VisionPaths     config.ModelFilePaths
	CategorizePaths config.ModelFilePaths
	EmbedPaths      config.ModelFilePaths
	Log             logger.Logger
	Cfg             config.Config
)

// Boot initializes Kronk and resolves model paths for tests.
// Models and libraries must already be downloaded (run 'make setup').
func Boot() {
	once.Do(func() {
		Log = logger.New()

		var err error
		Cfg, err = config.Load()
		if err != nil {
			fmt.Printf("config warning, using defaults: %v\n", err)
		}

		if v := os.Getenv("KRONK_BASE_PATH"); v != "" {
			Cfg.BasePath = v
		}

		if err := kronk.Init(Cfg); err != nil {
			fmt.Printf("kronk init: %v\n", err)
			os.Exit(1)
		}

		if info, err := kronk.Info(Cfg); err != nil {
			Log(context.Background(), "system info unavailable", "error", err)
		} else {
			Log(context.Background(), "system info",
				"processor", kronk.ActiveProcessor(),
				"llamaCppVersion", info.Version,
				"arch", info.Arch,
				"os", info.OS,
			)
		}

		paths, err := kronk.ResolvePaths(Cfg)
		if err != nil {
			fmt.Printf("resolve model paths: %v\n", err)
			os.Exit(1)
		}

		VisionPaths = paths.Vision
		CategorizePaths = paths.Categorize
		EmbedPaths = paths.Embed
	})
}
