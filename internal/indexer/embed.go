package indexer

import (
	"context"
	"errors"
	"fmt"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// LoadEmbed loads the embedding model into memory.
func (idx *Indexer) LoadEmbed(ctx context.Context) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.embedKrn != nil {
		return nil
	}

	idx.log(ctx, "loading embedding model")

	cfg := model.Config{
		ModelFiles:     idx.embedPaths.ModelFiles,
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

	idx.embedKrn = krn
	idx.log(ctx, "embedding model loaded",
		"context_window", krn.ModelConfig().ContextWindow,
	)

	return nil
}

// UnloadEmbed unloads the embedding model from memory.
func (idx *Indexer) UnloadEmbed(ctx context.Context) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.embedKrn == nil {
		return nil
	}

	idx.log(ctx, "unloading embedding model")

	if err := idx.embedKrn.Unload(ctx); err != nil {
		return fmt.Errorf("unload embedding model: %w", err)
	}

	idx.embedKrn = nil
	return nil
}

// EmbedText converts text into a vector embedding.
// Embedding model must be loaded first via LoadEmbed.
func (idx *Indexer) EmbedText(ctx context.Context, text string) ([]float32, error) {
	idx.mu.Lock()
	krn := idx.embedKrn
	idx.mu.Unlock()

	if krn == nil {
		return nil, ErrModelNotLoaded
	}

	if text == "" {
		return nil, ErrEmptyText
	}

	d := model.D{
		"input":    text,
		"truncate": true,
	}

	resp, err := krn.Embeddings(ctx, d)
	if err != nil {
		return nil, fmt.Errorf("embeddings: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, errors.New("no embedding data returned")
	}

	return resp.Data[0].Embedding, nil
}
