package tests

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

func TestIndexFolderAndSearch(t *testing.T) {
	testsboot.Boot()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test.index")
	testdataPath := filepath.Join("testdata")

	// Phase 1: Index folder
	svc := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})

	if err := svc.IndexFolder(ctx, testdataPath); err != nil {
		t.Fatalf("index folder: %v", err)
	}

	if svc.Index.Len() == 0 {
		t.Fatal("expected indexed images, got 0")
	}

	originalLen := svc.Index.Len()
	t.Logf("indexed %d images", originalLen)

	// Phase 2: Search
	results, err := svc.Search(ctx, "kids", 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results")
	}

	for i, r := range results {
		t.Logf("result %d: %s (score: %.4f) - %s", i+1, r.Path, r.Score, truncate(r.Description, 80))
	}

	// Phase 3: Close and verify persistence
	svc.Close(ctx)

	svc2 := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})
	defer svc2.Close(ctx)

	if err := svc2.Index.Load(); err != nil {
		t.Fatalf("load index: %v", err)
	}

	if svc2.Index.Len() != originalLen {
		t.Errorf("expected %d entries after reload, got %d", originalLen, svc2.Index.Len())
	}

	// Search should still work after reload
	results2, err := svc2.Search(ctx, "animals", 2)
	if err != nil {
		t.Fatalf("search after reload: %v", err)
	}

	t.Logf("search after reload returned %d results", len(results2))

	for i, r := range results {
		t.Logf("result %d: %s (score: %.4f) - %s", i+1, r.Path, r.Score, truncate(r.Description, 80))
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	testsboot.Boot()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "empty.index")

	svc := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})
	defer svc.Close(ctx)

	results, err := svc.Search(ctx, "anything", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results for empty index, got %d", len(results))
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
