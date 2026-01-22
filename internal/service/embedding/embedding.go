// Package embedding provides embedding model operations for text vectorization.
package embedding

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	ErrModelNotLoaded = errors.New("embedding model not loaded")
	ErrEmptyText      = errors.New("empty text")
)

// Embedder manages the embedding model for text vectorization.
type Embedder struct {
	log   logger.Logger
	paths models.Path

	mu  sync.Mutex
	krn *kronk.Kronk
}

// Config holds configuration for creating an Embedder.
type Config struct {
	Log   logger.Logger
	Paths models.Path
}

// New creates an Embedder with the given configuration.
func New(cfg Config) *Embedder {
	return &Embedder{
		log:   cfg.Log,
		paths: cfg.Paths,
	}
}

// Load loads the embedding model into memory.
func (e *Embedder) Load(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.krn != nil {
		return nil
	}

	start := time.Now()
	e.log(ctx, "loading embedding model")

	cfg := model.Config{
		ModelFiles:     e.paths.ModelFiles,
		ContextWindow:  2048,
		NBatch:         2048,
		NUBatch:        512,
		CacheTypeK:     model.GGMLTypeQ8_0,
		CacheTypeV:     model.GGMLTypeQ8_0,
		FlashAttention: model.FlashAttentionEnabled,
	}

	krn, err := kronk.New(cfg)
	if err != nil {
		return fmt.Errorf("load embedding model: %w", err)
	}

	e.krn = krn
	e.log(ctx, "embedding model loaded",
		"loading time", time.Since(start),
		"context_window", krn.ModelConfig().ContextWindow,
	)

	return nil
}

// Unload unloads the embedding model from memory.
func (e *Embedder) Unload(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.krn == nil {
		return nil
	}

	e.log(ctx, "unloading embedding model")

	if err := e.krn.Unload(ctx); err != nil {
		return fmt.Errorf("unload embedding model: %w", err)
	}

	e.krn = nil
	return nil
}

// IsLoaded returns true if the embedding model is loaded.
func (e *Embedder) IsLoaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.krn != nil
}

// Embed converts text into a vector embedding.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	e.mu.Lock()
	krn := e.krn
	e.mu.Unlock()

	if krn == nil {
		return nil, ErrModelNotLoaded
	}

	if text == "" {
		return nil, ErrEmptyText
	}

	data := model.D{
		"input":    text,
		"truncate": true,
	}

	start := time.Now()
	resp, err := krn.Embeddings(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("embeddings: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, errors.New("no embedding data returned")
	}

	e.log(ctx, "embedding finished", "embedding time", time.Since(start))
	return resp.Data[0].Embedding, nil
}
