// Package search provides similarity search algorithms.
package search

import (
	"math"
	"sort"
)

// ExpressionVector is one expression's embedding for a searchable image.
type ExpressionVector struct {
	Expression string
	Vector     []float32
}

// Result represents a search result with similarity score.
type Result struct {
	Path        string
	Description string
	Score       float32
	// ExpressionScores holds the per-expression cosine similarity to the query, keyed by
	// expression name. Useful for auditing why an image ranked where it did.
	ExpressionScores []scoredExpressions
}

// Entry represents a searchable item with one embedding vector per expression.
type Entry struct {
	Path        string
	Description string
	Expressions []ExpressionVector
}

// FindTopK finds the top-k images most similar to the query vector. Each image
// is scored by aggregating the query's cosine similarity against every one of
// its expression vectors.
func FindTopK(query []float32, entries []Entry, k int) []Result {
	if len(entries) == 0 || k <= 0 {
		return nil
	}

	results := make([]Result, 0, len(entries))

	for _, entry := range entries {
		expressionScores := make(map[string]float32, len(entry.Expressions))
		for _, fv := range entry.Expressions {
			s := CosineSimilarity(query, fv.Vector)
			expressionScores[fv.Expression] = s
		}

		topScores, topExpr := sortExpressions(expressionScores)

		results = append(results, Result{
			Path:             entry.Path,
			Description:      entry.Description,
			Score:            aggregate(topScores),
			ExpressionScores: topExpr,
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

type scoredExpressions struct {
	expr  string
	score float32
}

// sortExpressions sort the expressions by score and keep just the top n
func sortExpressions(m map[string]float32) ([]float32, []scoredExpressions) {
	n := 5
	cut := min(n, len(m))
	var scExp []scoredExpressions
	for k, v := range m {
		scExp = append(scExp, scoredExpressions{k, v})
	}

	sort.Slice(scExp, func(i, j int) bool {
		return scExp[i].score > scExp[j].score // descending order
	})

	var topScores []float32
	var sorted []scoredExpressions
	for i := 0; i < cut; i++ {
		sorted = append(sorted, scoredExpressions{"\n\t\t" + scExp[i].expr, scExp[i].score}) //adjust for logging
		topScores = append(topScores, scExp[i].score)                                        //collect top scores
	}

	if len(m) > n {
		sorted = append(sorted, scoredExpressions{"\n\texpressions below were discarded", float32(len(m) - n)})
		for i := cut; i < len(m); i++ {
			sorted = append(sorted, scoredExpressions{"\n\t\t" + scExp[i].expr, scExp[i].score})
		}
	}

	return topScores, sorted
}

// aggregate combines per-expression similarity scores into a single image score.
// The most proeminent score is amplified over the mean.
func aggregate(scores []float32) float32 {
	if len(scores) == 0 {
		return 0
	}

	max := scores[0]

	//mean score:
	var total float32
	for _, s := range scores {
		total += s
	}
	mean := total / float32(len(scores))

	return (mean * 0.5) + (max * 0.5) // This rate defines the search specialization
}
