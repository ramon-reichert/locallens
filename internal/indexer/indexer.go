// Package indexer provides image description and text embedding for LocalLens.
package indexer

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	ErrNotInitialized = errors.New("indexer not initialized")
	ErrEmptyImage     = errors.New("empty image data")
	ErrEmptyText      = errors.New("empty text")
	ErrModelNotLoaded = errors.New("model not loaded")
)

// Indexer manages vision and embedding models for image indexing.
type Indexer struct {
	log         logger.Logger
	visionPaths models.Path
	embedPaths  models.Path

	mu        sync.Mutex
	visionKrn *kronk.Kronk
	embedKrn  *kronk.Kronk
}

// Config holds configuration for creating an Indexer.
type Config struct {
	Log         logger.Logger
	VisionPaths models.Path
	EmbedPaths  models.Path
}

// New creates an Indexer with the given configuration.
func New(cfg Config) (*Indexer, error) {
	if err := kronk.Init(); err != nil {
		return nil, fmt.Errorf("kronk init: %w", err)
	}

	return &Indexer{
		log:         cfg.Log,
		visionPaths: cfg.VisionPaths,
		embedPaths:  cfg.EmbedPaths,
	}, nil
}

// Close unloads all models.
func (idx *Indexer) Close(ctx context.Context) error {
	if err := idx.UnloadVision(ctx); err != nil {
		return err
	}
	return idx.UnloadEmbed(ctx)
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
