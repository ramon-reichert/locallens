package performance

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service"
	"github.com/ramon-reichert/locallens/internal/service/index"
	"github.com/ramon-reichert/locallens/internal/service/search"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

// ------------------------------
// search with detailed queries from many images in the same context - wedding photos:
// const (
// 	indexPath  = "B:/RAMON/Fechados/fotos/Fotos casorio/escolhidas tratadas 2/test.index"
// 	resultsDir = "results/similarity"
// )

// var testQueries = []string{
// 	"bride and groom hugging each other",
// 	"receiving the blessings",
// 	"procession leaving the church",
// 	"The bride and groom at the altar with all the wedding party.",
// 	"at the altar with all the godparents",
// 	"bride and groom looking towards the altar",
// 	"the wedding kiss",
// 	"celebrating in front of the church",
// 	"entering the wedding party",
// 	"at the party with the grandparents",
// 	"The bride and groom at the party with the best man and maid of honor.",
// 	"newlyweds receiving communion",
// 	"Only the bride and groom with the altar vault in the background.",
// }

// ------------------------------

const (
	testdataDir = "../testdata"
	resultsDir  = "results/similarity"
)

var testQueries = []string{
	//specific:
	"a bride with her bridesmaids",
	"busy street with motorcycles",
	"natural landscape with trees",
	//generic:
	"moment of celebration",
	"nature",
	"busy city",
	//attributes:
	"colorful, lightening",
	"at night",
	"cartoon",
}

type queryResult struct {
	query   string
	results []search.Result
}

func TestSimilarity(t *testing.T) {
	testsboot.Boot()
	ctx := context.Background()

	// Copy testdata to temp dir to avoid writing index into source tree
	tmp := t.TempDir()
	if err := copyDir(testdataDir, tmp); err != nil {
		t.Fatalf("copy testdata: %v", err)
	}

	svc := service.New(service.Config{
		Log:         testsboot.Log,
		VisionPaths: testsboot.VisionPaths,
		EmbedPaths:  testsboot.EmbedPaths,
	})
	defer svc.Close(ctx)

	// Index the folder
	count, err := svc.IndexFolder(ctx, tmp, false)
	if err != nil {
		t.Fatalf("index folder: %v", err)
	}

	if count == 0 {
		t.Fatal("no images indexed")
	}

	fmt.Printf("indexed %d images\n", count)

	// Load entries for CSV export
	idx := index.New(filepath.Join(tmp, ".locallens.index"))
	idx.Load()
	entries := idx.All()

	// Run similarity tests for each query
	queryResults := make([]queryResult, 0, len(testQueries))

	for _, query := range testQueries {
		fmt.Printf("\nQuery: %q\n", query)

		results, err := svc.Search(ctx, tmp, query, len(entries), false)
		if err != nil {
			t.Errorf("search %q: %v", query, err)
			continue
		}

		for i, r := range results {
			fmt.Printf("  %d. %s (%.4f)\n", i+1, filepath.Base(r.Path), r.Score)
		}

		queryResults = append(queryResults, queryResult{
			query:   query,
			results: results,
		})
	}

	// Export to CSV
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		t.Fatalf("create results dir: %v", err)
	}

	timestamp := time.Now().Format("2006-01-02_15-04-05")
	csvPath := filepath.Join(resultsDir, fmt.Sprintf("similarity_%s.csv", timestamp))

	if err := exportMatrixCSV(csvPath, queryResults, entries); err != nil {
		t.Fatalf("export CSV: %v", err)
	}

	fmt.Printf("\n\nResults saved to %s\n", csvPath)
}

func exportMatrixCSV(path string, queryResults []queryResult, entries []index.Entry) error {
	if len(queryResults) == 0 || len(entries) == 0 {
		return fmt.Errorf("no data to export")
	}

	var b strings.Builder

	b.WriteString("Rank")
	for _, qr := range queryResults {
		b.WriteString(",")
		b.WriteString(escapeCSV(qr.query))
	}
	b.WriteString(",||,Filename,Description\n")

	numRanks := len(entries)
	for rank := 0; rank < numRanks; rank++ {
		b.WriteString(fmt.Sprintf("%d", rank+1))

		for _, qr := range queryResults {
			if rank < len(qr.results) {
				r := qr.results[rank]
				filename := filepath.Base(r.Path)
				b.WriteString(fmt.Sprintf(",%s (%.4f)", filename, r.Score))
			} else {
				b.WriteString(",")
			}
		}

		if rank < len(entries) {
			entry := entries[rank]
			filename := filepath.Base(entry.Path)
			desc := escapeCSV(entry.Description)
			b.WriteString(fmt.Sprintf(",||,%s,%s", filename, desc))
		} else {
			b.WriteString(",||,,")
		}

		b.WriteString("\n")
	}

	return os.WriteFile(path, []byte(b.String()), 0644)
}

func escapeCSV(s string) string {
	if strings.ContainsAny(s, ",\"\n") {
		s = strings.ReplaceAll(s, "\"", "\"\"")
		return "\"" + s + "\""
	}
	return s
}

func copyDir(src, dst string) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() {
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
