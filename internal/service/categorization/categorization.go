// Package categorization turns a prose image description into four
// search-oriented facets using a small chat model (e.g. Qwen3-0.6B).
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

// Facets holds the four search-oriented categories extracted from a
// description. Scene is a short natural-language summary; the other three are
// lists of discrete keywords/expressions.
type Facets struct {
	Scene      string   `json:"scene"`      // concise natural-language summary (1–2 sentences)
	Objects    []string `json:"objects"`    // nouns and key entities, including useful synonyms
	Actions    []string `json:"actions"`    // verbs and subject–object relationships
	Attributes []string `json:"attributes"` // colors, materials, style, lighting, mood, time of day
}

// IsEmpty reports whether every facet is empty.
func (f Facets) IsEmpty() bool {
	return strings.TrimSpace(f.Scene) == "" && len(f.Objects) == 0 && len(f.Actions) == 0 && len(f.Attributes) == 0
}

// FacetText pairs a facet name with its content text, ready to embed.
type FacetText struct {
	Name string
	Text string
}

// FacetTexts returns the content of each non-empty facet, one entry per facet.
// The text is content-only (no labels): Scene as-is, the list facets joined
// with ", ". Each entry becomes its own embedding vector.
func (f Facets) FacetTexts() []FacetText {
	var out []FacetText
	if s := strings.TrimSpace(f.Scene); s != "" {
		out = append(out, FacetText{Name: "scene", Text: s})
	}
	if len(f.Objects) > 0 {
		out = append(out, FacetText{Name: "objects", Text: strings.Join(f.Objects, ", ")})
	}
	if len(f.Actions) > 0 {
		out = append(out, FacetText{Name: "actions", Text: strings.Join(f.Actions, ", ")})
	}
	if len(f.Attributes) > 0 {
		out = append(out, FacetText{Name: "attributes", Text: strings.Join(f.Attributes, ", ")})
	}
	return out
}

// CategorizeResult holds the output of a Categorize call.
type CategorizeResult struct {
	Facets             Facets
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

// facetsSchema is the JSON schema the model output is constrained to. Kronk
// converts it to a GBNF grammar so the model can only emit a JSON object with
// a string "scene" and three array-of-string facets.
var facetsSchema = model.D{
	"type": "object",
	"properties": model.D{
		"scene":      model.D{"type": "string"},
		"objects":    model.D{"type": "array", "items": model.D{"type": "string"}},
		"actions":    model.D{"type": "array", "items": model.D{"type": "string"}},
		"attributes": model.D{"type": "array", "items": model.D{"type": "string"}},
	},
	"required": []string{"scene", "objects", "actions", "attributes"},
}

// Categorize turns a prose description into four search-oriented facets.
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
		"json_schema":     facetsSchema,
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

	facets, err := parseFacets([]byte(choice.Message.Content))
	if err != nil {
		return CategorizeResult{}, fmt.Errorf("parse facets: %w", err)
	}

	// Small models ignore length instructions, so enforce the scene cap here.
	facets.Scene = trimScene(facets.Scene, c.prompt.SceneMaxWords)

	if facets.IsEmpty() {
		return CategorizeResult{}, errors.New("categorize: empty facets")
	}

	result := CategorizeResult{
		Facets:             facets,
		TimeToFirstTokenMS: resp.Usage.TimeToFirstTokenMS,
		TokensPerSecond:    resp.Usage.TokensPerSecond,
	}

	// Print the facet texts exactly as they are embedded (content-only,
	// comma-joined) so the service-test output matches the embedder input.
	c.log(ctx, "categorize",
		"elapsed time", time.Since(start),
		"scene", facets.Scene,
		"objects", strings.Join(facets.Objects, ", "),
		"actions", strings.Join(facets.Actions, ", "),
		"attributes", strings.Join(facets.Attributes, ", "),
	)

	return result, nil
}

// parseFacets decodes the model's JSON output into Facets. It is a pure
// function (no model) so it can be unit-tested directly.
func parseFacets(data []byte) (Facets, error) {
	var f Facets
	if err := json.Unmarshal(data, &f); err != nil {
		return Facets{}, err
	}
	return f, nil
}

// trimScene caps s to at most maxWords words, dropping any trailing dangling
// punctuation left by the cut. maxWords <= 0 leaves s unchanged (only
// surrounding whitespace is trimmed).
func trimScene(s string, maxWords int) string {
	s = strings.TrimSpace(s)
	if maxWords <= 0 {
		return s
	}
	words := strings.Fields(s)
	if len(words) <= maxWords {
		return s
	}
	return strings.TrimRight(strings.Join(words[:maxWords], " "), " ,;:.-")
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
