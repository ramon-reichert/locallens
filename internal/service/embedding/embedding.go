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

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	ErrModelNotLoaded = errors.New("embedding model not loaded")
	ErrEmptyText      = errors.New("empty text")
)

// Kind identifies the retrieval role of text being embedded.
type Kind string

const (
	Document Kind = "document"
	Query    Kind = "query"

	documentPrefix = "task: search result | document: "
	queryPrefix    = "task: search result | query: "
)

// Embedder manages the embedding model for text vectorization.
type Embedder struct {
	log   logger.Logger
	paths config.ModelFilePaths
	embed config.EmbedModelConfig

	mu  sync.Mutex
	krn *kronk.Kronk
}

// Config holds configuration for creating an Embedder.
type Config struct {
	Log   logger.Logger
	Paths config.ModelFilePaths
	Embed config.EmbedModelConfig
}

// New creates an Embedder with the given configuration.
func New(cfg Config) *Embedder {
	return &Embedder{
		log:   cfg.Log,
		paths: cfg.Paths,
		embed: cfg.Embed,
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
	e.log(ctx, "\n=============\nembedder load", "embedding model", e.paths.ModelFiles)

	em := e.embed

	fa := model.FlashAttentionDisabled
	if em.FlashAttention {
		fa = model.FlashAttentionEnabled
	}

	krn, err := kronk.New(
		model.WithModelFiles(e.paths.ModelFiles),
		model.WithContextWindow(em.ContextWindow),
		model.WithNBatch(em.NBatch),
		model.WithNUBatch(em.NUBatch),
		model.WithCacheTypeK(model.GGMLType(config.ParseGGMLType(em.CacheTypeK))),
		model.WithCacheTypeV(model.GGMLType(config.ParseGGMLType(em.CacheTypeV))),
		model.WithFlashAttention(fa),
	)
	if err != nil {
		return fmt.Errorf("load embedding model: %w", err)
	}

	e.krn = krn
	e.log(ctx, "embedder load", "loading time", time.Since(start))

	return nil
}

// Unload unloads the embedding model from memory.
func (e *Embedder) Unload(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.krn == nil {
		return nil
	}

	e.log(ctx, "embedder unload")

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

// EmbedResult holds the output of an Embed call.
type EmbedResult struct {
	Embedding []float32
	Elapsed   time.Duration
}

// Embed converts text into a vector embedding using the prompt prefix for kind.
func (e *Embedder) Embed(ctx context.Context, kind Kind, text string) (EmbedResult, error) {
	e.mu.Lock()
	krn := e.krn
	e.mu.Unlock()

	if krn == nil {
		return EmbedResult{}, ErrModelNotLoaded
	}

	if text == "" {
		return EmbedResult{}, ErrEmptyText
	}

	input := documentPrefix + text
	if kind == Query {
		input = queryPrefix + text
	}

	data := model.D{
		"input":    input,
		"truncate": true,
	}

	start := time.Now()
	resp, err := krn.Embeddings(ctx, data)
	if err != nil {
		return EmbedResult{}, fmt.Errorf("embeddings: %w", err)
	}

	if len(resp.Data) == 0 {
		return EmbedResult{}, errors.New("no embedding data returned")
	}

	elapsed := time.Since(start)
	e.log(ctx, "embed image", "elapsed time", elapsed)

	return EmbedResult{
		Embedding: resp.Data[0].Embedding,
		Elapsed:   elapsed,
	}, nil
}
