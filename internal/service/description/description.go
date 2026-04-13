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

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/image"
)

var (
	ErrModelNotLoaded = errors.New("vision model not loaded")
)

// DescribeResult holds the output of a Describe call.
type DescribeResult struct {
	Description        string
	TimeToFirstTokenMS float64
	TokensPerSecond    float64
}

// Describer manages the vision model for image description.
type Describer struct {
	log     logger.Logger
	paths   config.ModelFilePaths
	vision  config.VisionModelConfig
	prompt  config.VisionPrompt
	maxSide int

	mu  sync.Mutex
	krn *kronk.Kronk
}

// Config holds configuration for creating a Describer.
type Config struct {
	Log     logger.Logger
	Paths   config.ModelFilePaths
	Vision  config.VisionModelConfig
	Prompt  config.VisionPrompt
	MaxSide int
}

// New creates a Describer with the given configuration.
func New(cfg Config) *Describer {
	return &Describer{
		log:     cfg.Log,
		paths:   cfg.Paths,
		vision:  cfg.Vision,
		prompt:  cfg.Prompt,
		maxSide: cfg.MaxSide,
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
	d.log(ctx, "describer load", "vision model", d.paths.ModelFiles)

	v := d.vision

	cfg := model.Config{
		ModelFiles:    d.paths.ModelFiles,
		ProjFile:      d.paths.ProjFile,
		ContextWindow: v.ContextWindow,
		NBatch:        v.NBatch,
		NUBatch:       v.NUBatch,
		CacheTypeK:    model.GGMLType(config.ParseGGMLType(v.CacheTypeK)),
		CacheTypeV:    model.GGMLType(config.ParseGGMLType(v.CacheTypeV)),
	}

	krn, err := kronk.New(cfg)
	if err != nil {
		return fmt.Errorf("load vision model: %w", err)
	}

	d.krn = krn
	d.log(ctx, "describer load", "loading time", time.Since(start))

	return nil
}

// Unload unloads the vision model from memory.
func (d *Describer) Unload(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.krn == nil {
		return nil
	}

	d.log(ctx, "describer unload")

	if err := d.krn.Unload(ctx); err != nil {
		return fmt.Errorf("unload vision model: %w", err)
	}

	d.krn = nil
	return nil
}

// IsLoaded returns true if the vision model is loaded.
func (d *Describer) IsLoaded() bool { // TODO: Add an idle timer to the models. Than this function can be usefull. The same for the Embedder ones.
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.krn != nil
}

// Describe generates a text description of the image at the given path.
func (d *Describer) Describe(ctx context.Context, imagePath string) (DescribeResult, error) {
	d.mu.Lock()
	krn := d.krn
	d.mu.Unlock()

	if krn == nil {
		return DescribeResult{}, ErrModelNotLoaded
	}

	p := d.prompt
	maxSide := d.maxSide

	d.log(ctx, "\n=============\ndescribe image", "resize to", maxSide, "path", imagePath)

	imageData, err := image.Resize(imagePath, maxSide)
	if err != nil {
		return DescribeResult{}, fmt.Errorf("resize image: %w", err)
	}

	messages := []model.D{
		{"role": "system", "content": p.SystemPrompt},
		{"role": "user", "content": imageData},
		{"role": "user", "content": p.UserPrompt},
	}

	data := model.D{
		"messages":    messages,
		"temperature": p.Temperature,
		"max_tokens":  p.MaxTokens,
	}

	start := time.Now()

	resp, err := krn.Chat(ctx, data)
	if err != nil {
		return DescribeResult{}, fmt.Errorf("chat: %w", err)
	}

	if len(resp.Choices) > 0 && resp.Choices[0].FinishReason() == model.FinishReasonError {
		errMsg := ""
		if resp.Choices[0].Delta != nil {
			errMsg = resp.Choices[0].Delta.Content
		} else if resp.Choices[0].Message != nil {
			errMsg = resp.Choices[0].Message.Content
		}
		return DescribeResult{}, fmt.Errorf("describe: model error: %s", errMsg)
	}

	result := DescribeResult{
		Description:        resp.Choices[0].Message.Content,
		TimeToFirstTokenMS: resp.Usage.TimeToFirstTokenMS,
		TokensPerSecond:    resp.Usage.TokensPerSecond,
	}

	d.log(ctx, "describe image", "elapsed time", time.Since(start), "description", result.Description+"\n=============\n")

	return result, nil
}
