package search_test

import (
	"testing"

	"github.com/ramon-reichert/locallens/internal/service/search"
)

func TestFindTopK(t *testing.T) {
	entries := []search.Entry{
		{Path: "dog.jpg", Description: "A brown dog", Expressions: []search.ExpressionVector{
			{Expression: "scene", Vector: []float32{1, 0, 0, 0}},
		}},
		{Path: "cat.jpg", Description: "A white cat", Expressions: []search.ExpressionVector{
			{Expression: "scene", Vector: []float32{0, 1, 0, 0}},
		}},
		{Path: "bird.jpg", Description: "A blue bird", Expressions: []search.ExpressionVector{
			{Expression: "scene", Vector: []float32{0, 0, 1, 0}},
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

// TestFindTopK_AggregatesExpressionScores verifies that images are scored from
// their per-expression similarities and that ExpressionScores is populated per
// expression.
func TestFindTopK_AveragesExpressionScores(t *testing.T) {
	query := []float32{1, 0, 0, 0}

	entries := []search.Entry{
		// One perfectly-matching expression: mean = 1.0.
		{Path: "precise.jpg", Expressions: []search.ExpressionVector{
			{Expression: "objects", Vector: []float32{1, 0, 0, 0}},
		}},
		// Same strong expression plus an unrelated one: mean = 0.5, max = 1.0,
		// aggregate = 0.75.
		{Path: "diluted.jpg", Expressions: []search.ExpressionVector{
			{Expression: "objects", Vector: []float32{1, 0, 0, 0}},
			{Expression: "attributes", Vector: []float32{0, 1, 0, 0}},
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

	// Aggregation must not reward the extra expression: diluted ≈ 0.75,
	// precise ≈ 1.0.
	if results[0].Score < 0.99 {
		t.Errorf("precise.jpg aggregate = %.4f, want ~1.0", results[0].Score)
	}
	if results[1].Score > 0.76 || results[1].Score < 0.74 {
		t.Errorf("diluted.jpg aggregate = %.4f, want ~0.75", results[1].Score)
	}

}

func TestFindTopK_LessThanK(t *testing.T) {
	entries := []search.Entry{
		{Path: "a.jpg", Description: "A", Expressions: []search.ExpressionVector{
			{Expression: "scene", Vector: []float32{1, 0}},
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
