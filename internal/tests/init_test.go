package tests

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	visionPaths models.Path
	embedPaths  models.Path
	log         logger.Logger
)

func TestMain(m *testing.M) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	log = logger.New()

	log(ctx, "downloading models for tests")
	paths, err := kronk.DownloadModels(ctx, log)
	if err != nil {
		fmt.Printf("download models: %v\n", err)
		os.Exit(1)
	}

	visionPaths = paths.Vision
	embedPaths = paths.Embed

	log(ctx, "initializing kronk")
	if err := kronksdk.Init(); err != nil {
		fmt.Printf("kronk init: %v\n", err)
		os.Exit(1)
	}

	log(ctx, "running tests")
	os.Exit(m.Run())
}
