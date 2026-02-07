package tests

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

const testdataPath = "testdata"

// Shared service instance, indexed once in TestMain.
var svc *service.Service

const mustMatch = 2

// expectedSearchResults defines queries and their expected result ordering, for n topk results defined in mustMatch.
var expectedSearchResults = []struct {
	query    string
	expected []string
}{
	{"busy city street", []string{"vietnam.jpg", "night.jpg"}},
	{"animals or plants with predominat yellow", []string{"parrot.jpg", "forest.jpg"}},
	{"people celebrating, wedding", []string{"wedding.jpg", "graduate.jpg"}},
	{"nature landscape with trees", []string{"forest.jpg", "lighthouse.jpg"}},
	{"colorful cartoon characters", []string{"marvel.jpg", "parrot.jpg"}},
}

func TestMain(m *testing.M) {

	fmt.Println("\n===== TEST MAIN > Boot =====")

	testsboot.Boot()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	indexPath := filepath.Join(os.TempDir(), "locallens_test.index")

	svc = service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})

	fmt.Println("\n===== TEST MAIN > IndexFolder =====")

	if err := svc.IndexFolder(ctx, testdataPath); err != nil {
		fmt.Printf("index folder: %v\n", err)
		os.Exit(1)
	}

	code := m.Run()

	svc.Close(ctx)
	os.Remove(indexPath)
	os.Exit(code)

	fmt.Println("===== TEST MAIN > completed =====")
}

func TestIndexFolder(t *testing.T) {

	entries := svc.Index.All()
	if len(entries) == 0 {
		t.Fatal("expected indexed images, got 0")
	}

	testImages := []string{
		"forest.jpg", "graduate.jpg", "lighthouse.jpg", "marvel.jpg",
		"night.jpg", "parrot.jpg", "vietnam.jpg", "wedding.jpg",
	}
	if len(entries) != len(testImages) {
		t.Errorf("expected %d indexed images, got %d", len(testImages), len(entries))
	}

	for _, entry := range entries {
		filename := filepath.Base(entry.Path)
		if entry.Description == "" {
			t.Errorf("%s: empty description", filename)
		}
		if len(entry.Embedding) == 0 {
			t.Errorf("%s: empty embedding", filename)
		}
	}
}

func TestIndexPersistence(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	indexPath := filepath.Join(t.TempDir(), "persist.index")

	// Save current index to a new path
	svc2 := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})

	for _, entry := range svc.Index.All() {
		svc2.Index.Add(entry)
	}
	if err := svc2.Index.Save(); err != nil {
		t.Fatalf("save index: %v", err)
	}

	originalLen := svc2.Index.Len()
	svc2.Close(ctx)

	// Reload from disk
	svc3 := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})
	defer svc3.Close(ctx)

	if err := svc3.Index.Load(); err != nil {
		t.Fatalf("load index: %v", err)
	}

	if svc3.Index.Len() != originalLen {
		t.Errorf("expected %d entries after reload, got %d", originalLen, svc3.Index.Len())
	}

	for _, entry := range svc3.Index.All() {
		if entry.Description == "" {
			t.Errorf("%s: description lost after reload", filepath.Base(entry.Path))
		}
		if len(entry.Embedding) == 0 {
			t.Errorf("%s: embedding lost after reload", filepath.Base(entry.Path))
		}
	}
}

func TestSearch(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	totalImages := svc.Index.Len()

	results, err := svc.Search(ctx, "any image", 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Error("expected search results, got 0")
	}
	if len(results) > 3 {
		t.Errorf("requested top 3, got %d", len(results))
	}

	// Scores must be in descending order
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: position %d (%.4f) > position %d (%.4f)",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// Request more than available returns all
	allResults, err := svc.Search(ctx, "anything", totalImages+10)
	if err != nil {
		t.Fatalf("search all: %v", err)
	}
	if len(allResults) != totalImages {
		t.Errorf("expected %d results when k > total, got %d", totalImages, len(allResults))
	}
}

func TestSearchExpectedOrder(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	for _, tc := range expectedSearchResults {
		t.Run(tc.query, func(t *testing.T) {

			results, err := svc.Search(ctx, tc.query, mustMatch)
			if err != nil {
				t.Fatalf("search: %v", err)
			}

			for i, want := range tc.expected {
				if i >= len(results) {
					t.Errorf("expected result at position %d (%s), but only got %d results", i, want, len(results))
					continue
				}
				got := filepath.Base(results[i].Path)
				if got != want {
					t.Errorf("position %d: expected %s, got %s (score: %.4f)", i, want, got, results[i].Score)
				}
			}
		})
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	indexPath := filepath.Join(t.TempDir(), "empty.index")

	emptySvc := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		IndexPath:   indexPath,
	})
	defer emptySvc.Close(ctx)

	results, err := emptySvc.Search(ctx, "anything", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results for empty index, got %d", len(results))
	}
}
