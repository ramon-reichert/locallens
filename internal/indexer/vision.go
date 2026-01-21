package indexer

import (
	"context"
	"fmt"
	"os"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// LoadVision loads the vision model into memory.
func (idx *Indexer) LoadVision(ctx context.Context) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.visionKrn != nil {
		return nil
	}

	idx.log(ctx, "loading vision model")

	cfg := model.Config{
		ModelFiles:    idx.visionPaths.ModelFiles,
		ProjFile:      idx.visionPaths.ProjFile,
		ContextWindow: 8192,
		NBatch:        2048,
		NUBatch:       2048,
		CacheTypeK:    model.GGMLTypeQ8_0,
		CacheTypeV:    model.GGMLTypeQ8_0,
	}

	krn, err := kronk.New(cfg)
	if err != nil {
		return fmt.Errorf("load vision model: %w", err)
	}

	idx.visionKrn = krn
	idx.log(ctx, "vision model loaded",
		"context_window", krn.ModelConfig().ContextWindow,
		"template", krn.ModelInfo().Template.FileName,
	)

	return nil
}

// UnloadVision unloads the vision model from memory.
func (idx *Indexer) UnloadVision(ctx context.Context) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.visionKrn == nil {
		return nil
	}

	idx.log(ctx, "unloading vision model")

	if err := idx.visionKrn.Unload(ctx); err != nil {
		return fmt.Errorf("unload vision model: %w", err)
	}

	idx.visionKrn = nil
	return nil
}

// DescribeImage generates a text description of the image at the given path.
// Vision model must be loaded first via LoadVision.
func (idx *Indexer) DescribeImage(ctx context.Context, imagePath string) (string, error) {
	idx.mu.Lock()
	krn := idx.visionKrn
	idx.mu.Unlock()

	if krn == nil {
		return "", ErrModelNotLoaded
	}

	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("read image: %w", err)
	}

	if len(imageData) == 0 {
		return "", ErrEmptyImage
	}

	prompt := "Describe this image in detail for semantic search. Focus on objects, people, actions, colors, and setting."

	d := model.D{
		"messages":    model.RawMediaMessage(prompt, imageData),
		"temperature": 0.3,
		"max_tokens":  256,
	}

	ch, err := krn.ChatStreaming(ctx, d)
	if err != nil {
		return "", fmt.Errorf("chat streaming: %w", err)
	}

	var description string
	for resp := range ch {
		if resp.Choice[0].FinishReason == model.FinishReasonError {
			return "", fmt.Errorf("model error: %s", resp.Choice[0].Delta.Content)
		}
		description += resp.Choice[0].Delta.Content
	}

	return description, nil
}
