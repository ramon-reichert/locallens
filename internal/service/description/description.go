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

	krn, err := kronk.New(
		model.WithModelFiles(d.paths.ModelFiles),
		model.WithProjFile(d.paths.ProjFile),
		model.WithContextWindow(v.ContextWindow),
		model.WithNBatch(v.NBatch),
		model.WithNUBatch(v.NUBatch),
		model.WithCacheTypeK(model.GGMLType(config.ParseGGMLType(v.CacheTypeK))),
		model.WithCacheTypeV(model.GGMLType(config.ParseGGMLType(v.CacheTypeV))),
		model.WithIncrementalCache(false),
	)
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

	d.log(ctx, "describe image", "resize to", maxSide)

	imageData, err := image.Resize(imagePath, maxSide)
	if err != nil {
		return DescribeResult{}, fmt.Errorf("resize image: %w", err)
	}

	messages := []model.D{}
	messages = append(messages, model.TextMessage(model.RoleSystem, p.SystemPrompt))
	messages = append(messages, model.ImageMessage(p.UserPrompt, imageData, "jpg")...)

	data := model.D{
		"messages":    messages,
		"temperature": p.Temperature,
		"max_tokens":  p.MaxTokens,
	}

	// DRY sampler: penalize repeated n-gram sequences to stop the vision
	// model from looping on text-heavy images. Only sent when enabled so the
	// SDK keeps its own defaults otherwise.
	if p.DryMultiplier > 0 {
		data["dry_multiplier"] = p.DryMultiplier
		data["dry_base"] = p.DryBase
		data["dry_allowed_length"] = p.DryAllowedLength
	}

	start := time.Now()

	resp, err := krn.Chat(ctx, data)
	if err != nil {
		return DescribeResult{}, fmt.Errorf("chat: %w", err)
	}

	choice, err := validateChatResponse(resp)
	if err != nil {
		return DescribeResult{}, err
	}

	result := DescribeResult{
		Description:        choice.Message.Content,
		TimeToFirstTokenMS: resp.Usage.TimeToFirstTokenMS,
		TokensPerSecond:    resp.Usage.TokensPerSecond,
	}

	d.log(ctx, "describe image", "elapsed time", time.Since(start), "description", result.Description)

	return result, nil
}

func validateChatResponse(resp model.ChatResponse) (model.Choice, error) {
	if len(resp.Choices) == 0 {
		return model.Choice{}, fmt.Errorf("chat: empty response")
	}

	choice := resp.Choices[0]
	switch {
	case choice.FinishReason() == model.FinishReasonError:
		errMsg := ""
		if choice.Delta != nil {
			errMsg = choice.Delta.Content
		} else if choice.Message != nil {
			errMsg = choice.Message.Content
		}
		return model.Choice{}, fmt.Errorf("describe: model error: %s", errMsg)
	case choice.Message == nil:
		return model.Choice{}, fmt.Errorf("chat: empty message")
	case choice.Message.Content == "":
		return model.Choice{}, fmt.Errorf("chat: blank message")
	case resp.Usage == nil:
		return model.Choice{}, fmt.Errorf("chat: empty usage")
	}

	return choice, nil
}
