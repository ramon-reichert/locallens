// Package description provides vision model operations for image description.
package description

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
	"github.com/ramon-reichert/locallens/internal/service/image"
)

var (
	ErrModelNotLoaded = errors.New("vision model not loaded")
	ErrEmptyImage     = errors.New("empty image data")
)

// Describer manages the vision model for image description.
type Describer struct {
	log   logger.Logger
	paths models.Path

	mu  sync.Mutex
	krn *kronk.Kronk
}

// Config holds configuration for creating a Describer.
type Config struct {
	Log   logger.Logger
	Paths models.Path
}

// New creates a Describer with the given configuration.
func New(cfg Config) *Describer {
	return &Describer{
		log:   cfg.Log,
		paths: cfg.Paths,
	}
}

// Load loads the vision model into memory.
func (d *Describer) Load(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.krn != nil {
		return nil
	}

	start := time.Now()
	d.log(ctx, "loading vision model")

	// TODO: tune this settings to fit app needs, maybe make it hardware dependent
	cfg := model.Config{
		ModelFiles:    d.paths.ModelFiles,
		ProjFile:      d.paths.ProjFile,
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

	d.krn = krn
	d.log(ctx, "vision model loaded",
		"loading time", time.Since(start),
		"context_window", krn.ModelConfig().ContextWindow,
		"template", krn.ModelInfo().Template.FileName,
	)

	return nil
}

// Unload unloads the vision model from memory.
func (d *Describer) Unload(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.krn == nil {
		return nil
	}

	d.log(ctx, "unloading vision model")

	if err := d.krn.Unload(ctx); err != nil {
		return fmt.Errorf("unload vision model: %w", err)
	}

	d.krn = nil
	return nil
}

// IsLoaded returns true if the vision model is loaded.
func (d *Describer) IsLoaded() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.krn != nil
}

// Describe generates a text description of the image at the given path.
func (d *Describer) Describe(ctx context.Context, imagePath string) (string, error) {
	d.mu.Lock()
	krn := d.krn
	d.mu.Unlock()

	if krn == nil {
		return "", ErrModelNotLoaded
	}

	d.log(ctx, "\nresizing image", "path", imagePath)

	imageData, err := image.Resize(imagePath, image.DefaultMaxSide)
	if err != nil {
		return "", fmt.Errorf("resize image: %w", err)
	}

	prompt := `Describe the image for semantic search.
Mention visible objects, with details only if clear, attributes, actions, colors, backgroud, place and setting.
Use short phrases or short clauses.
Output a single comma-separated list.
Avoid stylistic language.
Do not output bare numbers.`

	data := model.D{
		"messages":    model.RawMediaMessage(prompt, imageData),
		"temperature": 0.3,
		"max_tokens":  256,
	}

	d.log(ctx, "describing image", "path", imagePath)

	start := time.Now()

	resp, err := krn.Chat(ctx, data)
	if err != nil {
		return "", fmt.Errorf("chat: %w", err)
	}

	description := resp.Choice[0].Message.Content

	d.log(ctx, "description finished", "elapsed", time.Since(start), "description", description)
	return description, nil
}
