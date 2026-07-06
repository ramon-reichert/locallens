// Package categorization turns a prose image description into four
// search-oriented expressions using a small chat model (e.g. Qwen3-0.6B).
//
// The vision model produces a free-form paragraph; that paragraph is hard to
// embed well because it mixes many concepts into one blob. This step asks a
// tiny, grammar-constrained chat model to reshape the paragraph into four
// labeled lists — Scene, Objects, Actions, Attributes — which are then
// serialized into a single, embedding-friendly string.
package categorization

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

var (
	ErrModelNotLoaded = errors.New("categorization model not loaded")
	ErrEmptyText      = errors.New("empty description")
)

// Expressions holds the search-oriented expressions extracted from a description.
type Expressions []string

type expressionsResponse struct {
	Expressions []string `json:"expressions"`
}

// IsEmpty reports whether there are no expressions.
func (e Expressions) IsEmpty() bool {
	return len(e) == 0
}

// CategorizeResult holds the output of a Categorize call.
type CategorizeResult struct {
	Expressions        Expressions
	TimeToFirstTokenMS float64
	TokensPerSecond    float64
}

// Categorizer manages the small chat model used to categorize descriptions.
type Categorizer struct {
	log    logger.Logger
	paths  config.ModelFilePaths
	engine config.CategorizeModelConfig
	prompt config.CategorizePrompt

	mu  sync.Mutex
	krn *kronk.Kronk
}

// Config holds configuration for creating a Categorizer.
type Config struct {
	Log    logger.Logger
	Paths  config.ModelFilePaths
	Engine config.CategorizeModelConfig
	Prompt config.CategorizePrompt
}

// New creates a Categorizer with the given configuration.
func New(cfg Config) *Categorizer {
	return &Categorizer{
		log:    cfg.Log,
		paths:  cfg.Paths,
		engine: cfg.Engine,
		prompt: cfg.Prompt,
	}
}

// Load loads the categorization model into memory.
func (c *Categorizer) Load(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.krn != nil {
		return nil
	}

	start := time.Now()
	c.log(ctx, "categorizer load", "model", c.paths.ModelFiles)

	e := c.engine

	krn, err := kronk.New(
		model.WithModelFiles(c.paths.ModelFiles),
		model.WithContextWindow(e.ContextWindow),
		model.WithNBatch(e.NBatch),
		model.WithNUBatch(e.NUBatch),
		model.WithCacheTypeK(model.GGMLType(config.ParseGGMLType(e.CacheTypeK))),
		model.WithCacheTypeV(model.GGMLType(config.ParseGGMLType(e.CacheTypeV))),
		model.WithIncrementalCache(false),
	)
	if err != nil {
		return fmt.Errorf("load categorization model: %w", err)
	}

	c.krn = krn
	c.log(ctx, "categorizer load", "loading time", time.Since(start))

	return nil
}

// Unload unloads the categorization model from memory.
func (c *Categorizer) Unload(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.krn == nil {
		return nil
	}

	c.log(ctx, "categorizer unload")

	if err := c.krn.Unload(ctx); err != nil {
		return fmt.Errorf("unload categorization model: %w", err)
	}

	c.krn = nil
	return nil
}

// IsLoaded returns true if the categorization model is loaded.
func (c *Categorizer) IsLoaded() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.krn != nil
}

// expressionsSchema is the JSON schema the model output is constrained to. Kronk
// converts it to a GBNF grammar so the model can only emit a JSON object with
// one array-of-string field.
var expressionsSchema = model.D{
	"type": "object",
	"properties": model.D{
		"expressions": model.D{"type": "array", "items": model.D{"type": "string"}},
	},
	"required": []string{"expressions"},
}

// Categorize turns a prose description into search-oriented expressions.
func (c *Categorizer) Categorize(ctx context.Context, description string) (CategorizeResult, error) {
	c.mu.Lock()
	krn := c.krn
	c.mu.Unlock()

	if krn == nil {
		return CategorizeResult{}, ErrModelNotLoaded
	}

	if strings.TrimSpace(description) == "" {
		return CategorizeResult{}, ErrEmptyText
	}

	p := c.prompt

	messages := []model.D{
		model.TextMessage(model.RoleSystem, p.SystemPrompt),
		model.TextMessage(model.RoleUser, p.UserPrompt+"\n\n"+description),
	}

	data := model.D{
		"messages":        messages,
		"temperature":     p.Temperature,
		"max_tokens":      p.MaxTokens,
		"enable_thinking": false, // Qwen3: skip <think> reasoning for speed and stable output
		"json_schema":     expressionsSchema,
	}

	start := time.Now()

	resp, err := krn.Chat(ctx, data)
	if err != nil {
		return CategorizeResult{}, fmt.Errorf("chat: %w", err)
	}

	choice, err := validateChatResponse(resp)
	if err != nil {
		return CategorizeResult{}, err
	}

	expressions, err := parseExpressions([]byte(choice.Message.Content))
	if err != nil {
		return CategorizeResult{}, fmt.Errorf("parse expressions: %w", err)
	}

	if expressions.IsEmpty() {
		return CategorizeResult{}, errors.New("categorize: empty expressions")
	}

	result := CategorizeResult{
		Expressions:        expressions,
		TimeToFirstTokenMS: resp.Usage.TimeToFirstTokenMS,
		TokensPerSecond:    resp.Usage.TokensPerSecond,
	}

	// Print the expression text exactly as it is embedded (content-only,
	// comma-joined) so the service-test output matches the embedder input.
	c.log(ctx, "categorize",
		"elapsed time", time.Since(start),
		"expressions", strings.Join(expressions, ", "),
	)

	return result, nil
}

// parseExpressions decodes the model's JSON output into Expressions. It is a pure
// function (no model) so it can be unit-tested directly.
func parseExpressions(data []byte) (Expressions, error) {
	var resp expressionsResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return Expressions{}, err
	}

	expressions := make(Expressions, 0, min(len(resp.Expressions), 15))
	for _, expression := range resp.Expressions {
		expression = strings.TrimSpace(expression)
		if expression == "" {
			continue
		}
		expressions = append(expressions, expression)
		if len(expressions) == 15 {
			break
		}
	}

	return expressions, nil
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
		return model.Choice{}, fmt.Errorf("categorize: model error: %s", errMsg)
	case choice.Message == nil:
		return model.Choice{}, fmt.Errorf("chat: empty message")
	case choice.Message.Content == "":
		return model.Choice{}, fmt.Errorf("chat: blank message")
	case resp.Usage == nil:
		return model.Choice{}, fmt.Errorf("chat: empty usage")
	}

	return choice, nil
}
