package testsboot

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	once        sync.Once
	VisionPaths models.Path
	EmbedPaths  models.Path
	Log         logger.Logger
)

// Boot initialize dependencies for tests once
func Boot() {
	once.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()

		Log = logger.New()

		fmt.Println("installing dependencies for tests")
		if err := kronk.InstallDependencies(ctx, Log); err != nil {
			fmt.Printf("install dependencies: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("downloading models for tests")
		paths, err := kronk.DownloadModels(ctx, Log)
		if err != nil {
			fmt.Printf("download models: %v\n", err)
			os.Exit(1)
		}

		VisionPaths = paths.Vision
		EmbedPaths = paths.Embed

		fmt.Println("initializing kronk")
		if err := kronksdk.Init(); err != nil {
			fmt.Printf("kronk init: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("test system initialized")
	})
}
