package unittests

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service/description"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

func TestDescribe(t *testing.T) {
	testsboot.Boot()
	imageFile := filepath.Join("..", "testdata", "simple.jpg")
	if _, err := os.Stat(imageFile); os.IsNotExist(err) {
		t.Fatalf("%s not found", imageFile)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	d := description.New(description.Config{
		Log:   testsboot.Log,
		Paths: testsboot.VisionPaths,
	})
	defer d.Unload(ctx)

	if err := d.Load(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}

	desc, err := d.Describe(ctx, imageFile)
	if err != nil {
		t.Fatalf("describe: %v", err)
	}

	if desc == "" {
		t.Error("expected non-empty description")
	}

	t.Logf("description: %s", desc)
}

func TestDescribe_NotLoaded(t *testing.T) {
	testsboot.Boot()
	d := description.New(description.Config{
		Log: testsboot.Log,
	})

	_, err := d.Describe(context.Background(), "any.jpg")
	if err != description.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}
