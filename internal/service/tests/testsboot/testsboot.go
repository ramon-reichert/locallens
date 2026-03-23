package testsboot

import (
	"fmt"
	"os"
	"sync"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	once        sync.Once
	VisionPaths models.Path
	EmbedPaths  models.Path
	Log         logger.Logger
	Cfg         config.Config
)

// Boot initializes Kronk and resolves model paths for tests.
// Models and libraries must already be downloaded (run 'make setup').
func Boot() {
	once.Do(func() {
		Log = logger.New()
		Cfg = config.Load()

		if v := os.Getenv("KRONK_BASE_PATH"); v != "" {
			Cfg.BasePath = v
		}

		if err := kronksdk.Init(); err != nil {
			fmt.Printf("kronk init: %v\n", err)
			os.Exit(1)
		}

		paths, err := kronk.ResolvePaths(Cfg)
		if err != nil {
			fmt.Printf("resolve model paths: %v\n", err)
			os.Exit(1)
		}

		VisionPaths = paths.Vision
		EmbedPaths = paths.Embed
	})
}
