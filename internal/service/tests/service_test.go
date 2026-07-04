//go:build integration

package tests

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ramon-reichert/locallens/internal/service"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

const mustMatch = 2

// Shared service and folder path, indexed once in TestMain.
// indexProgressEvents captures every IndexProgress callback fired during
// the TestMain index run, so TestIndexFolder can assert on the sequence
// without paying for a second model load.
var (
	svc                 *service.Service
	testFolder          string
	indexProgressEvents []service.IndexProgressInfo
)

// expectedSearchResults defines queries and their expected result ordering, for n topk results defined in mustMatch.
var expectedSearchResults = []struct {
	query    string
	expected []string
}{
	{"animals or plants with predominat yellow", []string{"parrot.jpg", "forest.jpg"}},
	{"wedding party", []string{"wedding.jpg", "graduate.jpg"}}, // TODO: make it more context sensible with the old query: "people celebrating, wedding"
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

	var svcErr error
	svc, svcErr = service.New(ctx, service.Config{
		Log:             testsboot.Log,
		VisionPaths:     testsboot.VisionPaths,
		CategorizePaths: testsboot.CategorizePaths,
		EmbedPaths:      testsboot.EmbedPaths,
		AppCfg:          testsboot.Cfg,
	})
	if svcErr != nil {
		fmt.Printf("create service: %v\n", svcErr)
		os.Exit(1)
	}

	fmt.Println("\n===== TEST MAIN > IndexFolder =====")

	progress := func(p service.IndexProgressInfo) {
		indexProgressEvents = append(indexProgressEvents, p)
	}
	result, err := svc.IndexFolder(ctx, testFolder, progress)
	if err != nil {
		fmt.Printf("index folder: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n===== TEST MAIN > IndexFolder completed with: %d indexed images (%d new, %d failed) =====\n\n", result.IndexedTotal, result.Added, result.Failed)

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
		"forest.jpg", "graduate.jpg", "parrot.jpg", "wedding.jpg", "app.gif", "setup.jpeg",
	}
	if count != len(testImages) {
		t.Errorf("expected %d indexed images, got %d", len(testImages), count)
	}

	// Verify .locallens.index file was created inside the folder
	indexFile := filepath.Join(testFolder, ".locallens.index")
	if _, err := os.Stat(indexFile); os.IsNotExist(err) {
		t.Error("expected .locallens.index file inside the folder")
	}

	// Progress callback assertions — every image must fire, in order, one
	// "describing" event, then one "categorized" event (carrying the four
	// facets), then one "indexed" event. This lets the UI show "Describing
	// image X", then the extracted facets, then flip to "indexed" after save.
	//
	// Done semantics:
	//   - describing/categorized: count of images already fully indexed (this
	//     one not yet done), so the first pair has Done == 0.
	//   - indexed: count including this image, so the Nth indexed has Done == N.
	const stagesPerImage = 3
	wantEvents := stagesPerImage * count
	if got := len(indexProgressEvents); got != wantEvents {
		t.Fatalf("expected %d progress events (3 per image), got %d", wantEvents, got)
	}

	stageCycle := []string{"describing", "categorized", "indexed"}
	doneSoFar := 0
	for i, p := range indexProgressEvents {
		wantStage := stageCycle[i%stagesPerImage]
		if p.Stage != wantStage {
			t.Errorf("event %d: stage = %q, want %q", i, p.Stage, wantStage)
		}
		if p.Folder != testFolder {
			t.Errorf("event %d: folder = %q, want %q", i, p.Folder, testFolder)
		}
		if p.Current == "" {
			t.Errorf("event %d: empty Current path", i)
		}
		if p.Total != count {
			t.Errorf("event %d: Total = %d, want %d", i, p.Total, count)
		}

		switch p.Stage {
		case "describing":
			if p.Done != doneSoFar {
				t.Errorf("event %d (describing): Done = %d, want %d", i, p.Done, doneSoFar)
			}
			if p.ETA != 0 {
				t.Errorf("event %d (describing): expected ETA == 0, got %v", i, p.ETA)
			}
		case "categorized":
			if p.Facets == nil {
				t.Errorf("event %d (categorized): expected non-nil Facets", i)
				break
			}
			if p.Facets.IsEmpty() {
				t.Errorf("event %d (categorized): facets are empty for %s", i, filepath.Base(p.Current))
			}
		case "indexed":
			if p.Done != doneSoFar+1 {
				t.Errorf("event %d (indexed): Done = %d, want %d", i, p.Done, doneSoFar+1)
			}
			doneSoFar = p.Done
			// ETA is set on every indexed event except the last (no remaining work).
			if doneSoFar < count && p.ETA <= 0 {
				t.Errorf("event %d (indexed): expected positive ETA before completion, got %v", i, p.ETA)
			}
			if doneSoFar == count && p.ETA != 0 {
				t.Errorf("event %d (final indexed): expected ETA == 0, got %v", i, p.ETA)
			}
		}
	}

	if doneSoFar != count {
		t.Errorf("final indexed Done = %d, want %d", doneSoFar, count)
	}
}

func TestIndexPersistence(t *testing.T) {
	// The index was saved by IndexFolder. Load it fresh and verify.
	count := svc.IndexInfo(testFolder)
	if count == 0 {
		t.Fatal("expected entries after reload, got 0")
	}

	testImages := 6
	if count != testImages {
		t.Errorf("expected %d entries after reload, got %d", testImages, count)
	}
}

func TestSearch(t *testing.T) {
	ctx := context.Background()

	results, err := svc.Search(ctx, testFolder, "any image", 3)
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
	allResults, err := svc.Search(ctx, testFolder, "anything", totalImages+10)
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
			results, err := svc.Search(ctx, testFolder, tc.query, mustMatch)
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

	results, err := svc.Search(ctx, emptyDir, "anything", 5)
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
