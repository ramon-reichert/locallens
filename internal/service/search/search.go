// Package search provides similarity search algorithms.
package search

import (
	"math"
	"sort"
)

// Result represents a search result with similarity score.
type Result struct {
	Path        string
	Description string
	Score       float32
}

// Entry represents a searchable item with an embedding vector.
type Entry struct {
	Path        string
	Description string
	Embedding   []float32
}

// FindTopK finds the top-k most similar entries to the query vector.
func FindTopK(query []float32, entries []Entry, k int) []Result {
	if len(entries) == 0 || k <= 0 {
		return nil
	}

	results := make([]Result, 0, len(entries))

	for _, entry := range entries {
		score := CosineSimilarity(query, entry.Embedding)
		results = append(results, Result{
			Path:        entry.Path,
			Description: entry.Description,
			Score:       score,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if k > len(results) {
		k = len(results)
	}

	return results[:k]
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}
