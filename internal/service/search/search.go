// Package search provides similarity search algorithms.
package search

import (
	"fmt"
	"math"
	"path/filepath"
	"sort"
)

// FacetVector is one facet's embedding for a searchable image.
type FacetVector struct {
	Facet  string
	Vector []float32
}

// Result represents a search result with similarity score.
type Result struct {
	Path        string
	Description string
	Score       float32
	// FacetScores holds the per-facet cosine similarity to the query, keyed by
	// facet name. Useful for auditing why an image ranked where it did.
	FacetScores map[string]float32
}

// Entry represents a searchable item with one embedding vector per facet.
type Entry struct {
	Path        string
	Description string
	Facets      []FacetVector
}

// aggregate combines per-facet similarity scores into a single image score.
// The current strategy is the mean: an image is scored by how well the query
// matches its facets on average, so images are not favored simply for having
// more facets. Swap this one function to try sum, max, or a weighted scheme.
func aggregate(scores []float32) float32 {
	if len(scores) == 0 {
		return 0
	}
	var total float32
	for _, s := range scores {
		total += s
	}
	return total / float32(len(scores))
}

// FindTopK finds the top-k images most similar to the query vector. Each image
// is scored by aggregating the query's cosine similarity against every one of
// its facet vectors.
func FindTopK(query []float32, entries []Entry, k int) []Result {
	if len(entries) == 0 || k <= 0 {
		return nil
	}

	results := make([]Result, 0, len(entries))

	for _, entry := range entries {
		facetScores := make(map[string]float32, len(entry.Facets))
		scores := make([]float32, 0, len(entry.Facets))
		for _, fv := range entry.Facets {
			s := CosineSimilarity(query, fv.Vector)
			facetScores[fv.Facet] = s
			scores = append(scores, s)
		}

		results = append(results, Result{
			Path:        entry.Path,
			Description: entry.Description,
			Score:       aggregate(scores),
			FacetScores: facetScores,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if k > len(results) {
		k = len(results)
	}

	// debug code:
	for _, img := range results[:k] {

		fmt.Println(filepath.Base(img.Path), "average score: ", img.Score, "facets scores: ", img.FacetScores)
	}

	//------------

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
