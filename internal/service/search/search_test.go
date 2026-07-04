package search_test

import (
	"testing"

	"github.com/ramon-reichert/locallens/internal/service/search"
)

func TestFindTopK(t *testing.T) {
	entries := []search.Entry{
		{Path: "dog.jpg", Description: "A brown dog", Facets: []search.FacetVector{
			{Facet: "scene", Vector: []float32{1, 0, 0, 0}},
		}},
		{Path: "cat.jpg", Description: "A white cat", Facets: []search.FacetVector{
			{Facet: "scene", Vector: []float32{0, 1, 0, 0}},
		}},
		{Path: "bird.jpg", Description: "A blue bird", Facets: []search.FacetVector{
			{Facet: "scene", Vector: []float32{0, 0, 1, 0}},
		}},
	}

	query := []float32{1, 0.1, 0, 0}
	results := search.FindTopK(query, entries, 2)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].Path != "dog.jpg" {
		t.Errorf("expected dog.jpg as top result, got %s", results[0].Path)
	}

	t.Logf("top result: %s (score: %.4f)", results[0].Path, results[0].Score)
}

// TestFindTopK_AveragesFacetScores verifies that images are scored by the mean
// of their per-facet similarities — so an image with one strong facet outranks
// an image whose extra facet dilutes the average — and that FacetScores is
// populated per facet.
func TestFindTopK_AveragesFacetScores(t *testing.T) {
	query := []float32{1, 0, 0, 0}

	entries := []search.Entry{
		// One perfectly-matching facet: mean = 1.0.
		{Path: "precise.jpg", Facets: []search.FacetVector{
			{Facet: "objects", Vector: []float32{1, 0, 0, 0}},
		}},
		// Same strong facet plus an unrelated one: mean = (1.0 + 0) / 2 = 0.5.
		{Path: "diluted.jpg", Facets: []search.FacetVector{
			{Facet: "objects", Vector: []float32{1, 0, 0, 0}},
			{Facet: "attributes", Vector: []float32{0, 1, 0, 0}},
		}},
	}

	results := search.FindTopK(query, entries, 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].Path != "precise.jpg" {
		t.Errorf("expected precise.jpg first (higher average), got %s", results[0].Path)
	}
	if results[0].Score <= results[1].Score {
		t.Errorf("average score not greater: precise=%.4f diluted=%.4f", results[0].Score, results[1].Score)
	}

	// Averaging must not reward the extra facet: diluted ≈ 0.5, precise ≈ 1.0.
	if results[0].Score < 0.99 {
		t.Errorf("precise.jpg mean = %.4f, want ~1.0", results[0].Score)
	}
	if results[1].Score > 0.51 || results[1].Score < 0.49 {
		t.Errorf("diluted.jpg mean = %.4f, want ~0.5", results[1].Score)
	}

	diluted := results[1]
	if len(diluted.FacetScores) != 2 {
		t.Errorf("expected 2 facet scores for diluted.jpg, got %d", len(diluted.FacetScores))
	}
	if _, ok := diluted.FacetScores["attributes"]; !ok {
		t.Errorf("expected an 'attributes' facet score, got %v", diluted.FacetScores)
	}
}

func TestFindTopK_LessThanK(t *testing.T) {
	entries := []search.Entry{
		{Path: "a.jpg", Description: "A", Facets: []search.FacetVector{
			{Facet: "scene", Vector: []float32{1, 0}},
		}},
	}

	results := search.FindTopK([]float32{1, 0}, entries, 5)

	if len(results) != 1 {
		t.Errorf("expected 1 result when k > entries, got %d", len(results))
	}
}

func TestFindTopK_Empty(t *testing.T) {
	results := search.FindTopK([]float32{1, 0}, nil, 5)

	if results != nil {
		t.Errorf("expected nil for empty entries, got %v", results)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{
			name: "identical vectors",
			a:    []float32{1, 0, 0},
			b:    []float32{1, 0, 0},
			want: 1.0,
		},
		{
			name: "orthogonal vectors",
			a:    []float32{1, 0, 0},
			b:    []float32{0, 1, 0},
			want: 0.0,
		},
		{
			name: "opposite vectors",
			a:    []float32{1, 0, 0},
			b:    []float32{-1, 0, 0},
			want: -1.0,
		},
		{
			name: "empty vectors",
			a:    []float32{},
			b:    []float32{},
			want: 0.0,
		},
		{
			name: "different lengths",
			a:    []float32{1, 0},
			b:    []float32{1, 0, 0},
			want: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := search.CosineSimilarity(tt.a, tt.b)
			if got < tt.want-0.01 || got > tt.want+0.01 {
				t.Errorf("CosineSimilarity() = %v, want %v", got, tt.want)
			}
		})
	}
}
