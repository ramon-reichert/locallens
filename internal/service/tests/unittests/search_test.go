package unittests

import (
	"testing"

	"github.com/ramon-reichert/locallens/internal/service/search"
)

// TODO: Add conceptual search tests to see if its accurate.

func TestFindTopK(t *testing.T) {
	entries := []search.Entry{
		{Path: "dog.jpg", Description: "A brown dog", Embedding: []float32{1, 0, 0, 0}},
		{Path: "cat.jpg", Description: "A white cat", Embedding: []float32{0, 1, 0, 0}},
		{Path: "bird.jpg", Description: "A blue bird", Embedding: []float32{0, 0, 1, 0}},
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

func TestFindTopK_LessThanK(t *testing.T) {
	entries := []search.Entry{
		{Path: "a.jpg", Description: "A", Embedding: []float32{1, 0}},
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
