package tests

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

const mustMatch = 2

// Shared service and folder path, indexed once in TestMain.
var (
	svc        *service.Service
	testFolder string
)

// expectedSearchResults defines queries and their expected result ordering, for n topk results defined in mustMatch.
var expectedSearchResults = []struct {
	query    string
	expected []string
}{
	{"animals or plants with predominat yellow", []string{"parrot.jpg", "forest.jpg"}},
	{"people celebrating, wedding", []string{"wedding.jpg", "graduate.jpg"}},
}

func TestMain(m *testing.M) {

	fmt.Println("\n===== TEST MAIN > Boot =====")

	testsboot.Boot()

	ctx := context.Background()

	// Copy testdata to a temp dir so we don't write .locallens.index into the source tree
	tmp, err := os.MkdirTemp("", "locallens_test_*")
	if err != nil {
		fmt.Printf("create temp dir: %v\n", err)
		os.Exit(1)
	}
	testFolder = tmp

	if err := copyImages("testdata", testFolder); err != nil {
		fmt.Printf("copy testdata: %v\n", err)
		os.Exit(1)
	}

	svc = service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
		AppCfg:      testsboot.Cfg,
	})

	fmt.Println("\n===== TEST MAIN > IndexFolder =====")

	indexStart := time.Now()
	count, err := svc.IndexFolder(ctx, testFolder, false)
	if err != nil {
		fmt.Printf("index folder: %v\n", err)
		os.Exit(1)
	}
	indexElapsed := time.Since(indexStart)

	fmt.Printf("indexed %d images in %v (avg %v per image)\n=============\n\n", count, indexElapsed, indexElapsed/time.Duration(count))

	code := m.Run()

	svc.Close(ctx)
	os.RemoveAll(tmp)
	os.Exit(code)

	fmt.Println("===== TEST MAIN > completed =====")
}

func TestIndexFolder(t *testing.T) {
	count := svc.IndexInfo(testFolder)
	if count == 0 {
		t.Fatal("expected indexed images, got 0")
	}

	testImages := []string{
		"forest.jpg", "graduate.jpg", "parrot.jpg", "wedding.jpg",
	}
	if count != len(testImages) {
		t.Errorf("expected %d indexed images, got %d", len(testImages), count)
	}

	// Verify .locallens.index file was created inside the folder
	indexFile := filepath.Join(testFolder, ".locallens.index")
	if _, err := os.Stat(indexFile); os.IsNotExist(err) {
		t.Error("expected .locallens.index file inside the folder")
	}
}

func TestIndexPersistence(t *testing.T) {
	// The index was saved by IndexFolder. Load it fresh and verify.
	count := svc.IndexInfo(testFolder)
	if count == 0 {
		t.Fatal("expected entries after reload, got 0")
	}

	testImages := 4
	if count != testImages {
		t.Errorf("expected %d entries after reload, got %d", testImages, count)
	}
}

func TestSearch(t *testing.T) {
	ctx := context.Background()

	results, err := svc.Search(ctx, testFolder, "any image", 3, false)
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
	totalImages := svc.IndexInfo(testFolder)
	allResults, err := svc.Search(ctx, testFolder, "anything", totalImages+10, false)
	if err != nil {
		t.Fatalf("search all: %v", err)
	}
	if len(allResults) != totalImages {
		t.Errorf("expected %d results when k > total, got %d", totalImages, len(allResults))
	}
}

func TestSearchExpectedOrder(t *testing.T) {
	ctx := context.Background()

	for _, tc := range expectedSearchResults {
		t.Run(tc.query, func(t *testing.T) {
			results, err := svc.Search(ctx, testFolder, tc.query, mustMatch, false)
			if err != nil {
				t.Fatalf("search: %v", err)
			}

			t.Logf("query: %q", tc.query)
			for i, r := range results {
				t.Logf("  [%d] %.4f %s", i, r.Score, filepath.Base(r.Path))
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
	ctx := context.Background()

	emptyDir := t.TempDir()

	results, err := svc.Search(ctx, emptyDir, "anything", 5, false)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results for empty folder, got %d", len(results))
	}
}

// copyImages copies image files from src to dst directory.
func copyImages(src, dst string) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(entry.Name()))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
		default:
			continue
		}

		data, err := os.ReadFile(filepath.Join(src, entry.Name()))
		if err != nil {
			return err
		}

		if err := os.WriteFile(filepath.Join(dst, entry.Name()), data, 0644); err != nil {
			return err
		}
	}

	return nil
}
