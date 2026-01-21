package indexer_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/indexer"
	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

func TestDescribeImage(t *testing.T) {
	imageFile := filepath.Join("testdata", "sample.jpg")
	if _, err := os.Stat(imageFile); os.IsNotExist(err) {
		t.Skip("testdata/sample.jpg not found")
	}

	ctx := context.Background()
	log := logger.New()

	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	paths, err := kronk.DownloadModels(ctx, log)
	if err != nil {
		t.Fatalf("download models: %v", err)
	}

	idx, err := indexer.New(indexer.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
	})
	if err != nil {
		t.Fatalf("new indexer: %v", err)
	}
	defer idx.Close(ctx)

	if err := idx.LoadVision(ctx); err != nil {
		t.Fatalf("load vision: %v", err)
	}

	desc, err := idx.DescribeImage(ctx, imageFile)
	if err != nil {
		t.Fatalf("describe image: %v", err)
	}

	if desc == "" {
		t.Error("expected non-empty description")
	}

	t.Logf("description: %s", desc)
}

func TestEmbedText(t *testing.T) {
	ctx := context.Background()
	log := logger.New()

	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	paths, err := kronk.DownloadModels(ctx, log)
	if err != nil {
		t.Fatalf("download models: %v", err)
	}

	idx, err := indexer.New(indexer.Config{
		Log:         log,
		VisionPaths: paths.Vision,
		EmbedPaths:  paths.Embed,
	})
	if err != nil {
		t.Fatalf("new indexer: %v", err)
	}
	defer idx.Close(ctx)

	if err := idx.LoadEmbed(ctx); err != nil {
		t.Fatalf("load embed: %v", err)
	}

	text := "A brown dog running through a green field"

	vec, err := idx.EmbedText(ctx, text)
	if err != nil {
		t.Fatalf("embed text: %v", err)
	}

	if len(vec) == 0 {
		t.Error("expected non-empty embedding vector")
	}

	if vec[0] == 0 && vec[len(vec)-1] == 0 {
		t.Error("expected embedding to have non-zero values")
	}

	t.Logf("embedding dimensions: %d", len(vec))
}

func TestDescribeImage_NotLoaded(t *testing.T) {
	ctx := context.Background()
	log := logger.Discard()

	idx, err := indexer.New(indexer.Config{
		Log: log,
	})
	if err != nil {
		t.Fatalf("new indexer: %v", err)
	}
	defer idx.Close(ctx)

	_, err = idx.DescribeImage(ctx, "any.jpg")
	if err != indexer.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}

func TestEmbedText_NotLoaded(t *testing.T) {
	ctx := context.Background()
	log := logger.Discard()

	idx, err := indexer.New(indexer.Config{
		Log: log,
	})
	if err != nil {
		t.Fatalf("new indexer: %v", err)
	}
	defer idx.Close(ctx)

	_, err = idx.EmbedText(ctx, "any text")
	if err != indexer.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	c := []float32{0, 1, 0}

	simSame := indexer.CosineSimilarity(a, b)
	if simSame < 0.99 {
		t.Errorf("identical vectors should have similarity ~1, got %f", simSame)
	}

	simOrthogonal := indexer.CosineSimilarity(a, c)
	if simOrthogonal > 0.01 {
		t.Errorf("orthogonal vectors should have similarity ~0, got %f", simOrthogonal)
	}
}
