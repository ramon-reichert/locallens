package tests

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

const IndexPath = "testdata/test.index"

func TestIndexFolderAndSearch(t *testing.T) {
	testsboot.Boot()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	testdataPath := "testdata"

	svc := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   IndexPath,
	})

	if err := svc.IndexFolder(ctx, testdataPath); err != nil {
		t.Fatalf("index folder: %v", err)
	}
	defer svc.Close(ctx)

	if svc.Index.Len() == 0 {
		t.Fatal("expected indexed images, got 0")
	}

	fmt.Printf("indexed %d images to %s\n", svc.Index.Len(), IndexPath)

	// Quick search test
	query := "busy city"
	results, err := svc.Search(ctx, query, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results")
	}

	fmt.Printf("\nquery: %s\n", query)
	for i, r := range results {
		fmt.Printf("  %d. %s (score: %.4f)\n", i+1, filepath.Base(r.Path), r.Score)
	}

	// Verify reload works
	svc2 := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   IndexPath,
	})
	defer svc2.Close(ctx)

	if err := svc2.Index.Load(); err != nil {
		t.Fatalf("reload index: %v", err)
	}

	if svc2.Index.Len() != svc.Index.Len() {
		t.Errorf("expected %d entries after reload, got %d", svc.Index.Len(), svc2.Index.Len())
	}

	fmt.Printf("\nindex reloaded successfully with %d entries\n", svc2.Index.Len())
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
