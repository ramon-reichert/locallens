package tests

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/description"
)

func TestDescribe(t *testing.T) {
	imageFile := filepath.Join("testdata", "sample.jpg")
	if _, err := os.Stat(imageFile); os.IsNotExist(err) {
		t.Skip("testdata/sample.jpg not found")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	d := description.New(description.Config{
		Log:   log,
		Paths: visionPaths,
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
	d := description.New(description.Config{
		Log: log,
	})

	_, err := d.Describe(context.Background(), "any.jpg")
	if err != description.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}
