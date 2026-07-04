// Package config manages LocalLens application configuration.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

const (
	appDir     = ".locallens"
	configFile = "config.json"
)

// =========================================================================
// Model URLs and IDs

// Models holds download URLs for all models.
type Models struct {
	VisionModelURL     string `json:"visionModelURL"`
	VisionProjURL      string `json:"visionProjURL"`
	EmbedModelURL      string `json:"embedModelURL"`
	CategorizeModelURL string `json:"categorizeModelURL"`
}

// VisionModelID returns the model ID derived from the vision model URL.
func (m Models) VisionModelID() string {
	return modelIDFromURL(m.VisionModelURL)
}

// EmbedModelID returns the model ID derived from the embedding model URL.
func (m Models) EmbedModelID() string {
	return modelIDFromURL(m.EmbedModelURL)
}

// CategorizeModelID returns the model ID derived from the categorization model URL.
func (m Models) CategorizeModelID() string {
	return modelIDFromURL(m.CategorizeModelURL)
}

func modelIDFromURL(url string) string {
	name := path.Base(url)
	return strings.TrimSuffix(name, path.Ext(name))
}

// =========================================================================
// Model engine configs

// VisionModelConfig holds Kronk configuration for the vision model.
type VisionModelConfig struct {
	ContextWindow int    `json:"contextWindow"`
	NBatch        int    `json:"nBatch"`
	NUBatch       int    `json:"nUBatch"`
	CacheTypeK    string `json:"cacheTypeK"`
	CacheTypeV    string `json:"cacheTypeV"`
}

// EmbedModelConfig holds Kronk configuration for the embedding model.
type EmbedModelConfig struct {
	ContextWindow  int    `json:"contextWindow"`
	NBatch         int    `json:"nBatch"`
	NUBatch        int    `json:"nUBatch"`
	CacheTypeK     string `json:"cacheTypeK"`
	CacheTypeV     string `json:"cacheTypeV"`
	FlashAttention bool   `json:"flashAttention"`
}

// CategorizeModelConfig holds Kronk configuration for the categorization model.
type CategorizeModelConfig struct {
	ContextWindow int    `json:"contextWindow"`
	NBatch        int    `json:"nBatch"`
	NUBatch       int    `json:"nUBatch"`
	CacheTypeK    string `json:"cacheTypeK"`
	CacheTypeV    string `json:"cacheTypeV"`
}

// =========================================================================
// Vision prompt and image config

// VisionPrompt holds prompt configuration for image description.
type VisionPrompt struct {
	SystemPrompt string  `json:"systemPrompt"`
	UserPrompt   string  `json:"userPrompt"`
	MaxTokens    int     `json:"maxTokens"`
	Temperature  float64 `json:"temperature"`

	// DRY (Don't Repeat Yourself) sampler. Penalizes repeated n-gram
	// sequences (multi-word patterns), which prevents the model from getting
	// stuck in structural loops on text-heavy images while still allowing
	// legitimate repeated words. DryMultiplier == 0 disables the sampler.
	DryMultiplier    float64 `json:"dryMultiplier"`
	DryBase          float64 `json:"dryBase"`
	DryAllowedLength int     `json:"dryAllowedLength"`

	// Repetition penalty. A blunt per-token penalty on tokens that appeared in
	// the last RepeatLastN tokens. RepeatPenalty <= 1.0 disables it (1.0 = no
	// penalty). Useful as a fallback for single-token spam that DRY's n-gram
	// matching may miss. RepeatLastN <= 0 uses the SDK default (64).
	RepeatPenalty float64 `json:"repeatPenalty"`
	RepeatLastN   int     `json:"repeatLastN"`

	// FrequencyPenalty penalizes a token proportionally to how often it has
	// already appeared, directly discouraging a word that keeps recurring.
	// PresencePenalty applies a flat penalty to any token that appeared at
	// least once. Both default to 0 (disabled).
	FrequencyPenalty float64 `json:"frequencyPenalty"`
	PresencePenalty  float64 `json:"presencePenalty"`
}

// CategorizePrompt holds prompt configuration for description categorization.
type CategorizePrompt struct {
	SystemPrompt string  `json:"systemPrompt"`
	UserPrompt   string  `json:"userPrompt"`
	MaxTokens    int     `json:"maxTokens"`
	Temperature  float64 `json:"temperature"`
}

// ImageConfig holds image preprocessing settings.
type ImageConfig struct {
	MaxSide int `json:"maxSide"`
}

// =========================================================================
// Top-level config

// Config holds the full application configuration.
type Config struct {
	BasePath         string                `json:"basePath"`
	Processor        string                `json:"processor"`
	LlamaCppVersion  string                `json:"llamaCppVersion"`
	ModelsURLs       Models                `json:"models"`
	Vision           VisionModelConfig     `json:"visionModel"`
	Embed            EmbedModelConfig      `json:"embedModel"`
	Categorize       CategorizeModelConfig `json:"categorizeModel"`
	Prompt           VisionPrompt          `json:"prompt"`
	CategorizePrompt CategorizePrompt      `json:"categorizePrompt"`
	Image            ImageConfig           `json:"image"`
}

// ModelFilePaths holds resolved file system paths for a single model.
// This is the project's own type, decoupled from the Kronk SDK's models.Path.
type ModelFilePaths struct {
	ModelFiles []string
	ProjFile   string
}

// DefaultBasePath returns the default base path for model storage.
func DefaultBasePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, appDir, "kronk")
}

// Defaults returns the default configuration with sensible values.
func Defaults() Config {
	return Config{
		BasePath:        DefaultBasePath(),
		Processor:       "cpu",
		LlamaCppVersion: "b9750",
		ModelsURLs: Models{
			VisionModelURL:     "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
			VisionProjURL:      "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
			EmbedModelURL:      "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf",
			CategorizeModelURL: "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
		},
		Vision: VisionModelConfig{ // TODO: Check this config at model provider site
			ContextWindow: 8192,
			NBatch:        2048,
			NUBatch:       1024,
			CacheTypeK:    "Q8_0", // K cache is highly sensitive to quantization (llama.cpp PR #7412); Q4_0 causes hallucinations on Qwen2-VL due to aggressive GQA (2 KV heads / 14 attn heads)
			CacheTypeV:    "Q8_0",
		},
		Embed: EmbedModelConfig{
			ContextWindow:  2048, // TODO: Check this config at model provider site
			NBatch:         2048,
			NUBatch:        512,
			CacheTypeK:     "Q8_0",
			CacheTypeV:     "Q8_0",
			FlashAttention: true,
		},
		Categorize: CategorizeModelConfig{
			ContextWindow: 4096, // room for the prose description plus the JSON facets
			NBatch:        2048,
			NUBatch:       512,
			CacheTypeK:    "Q8_0",
			CacheTypeV:    "Q8_0",
		},
		Prompt: VisionPrompt{
			SystemPrompt:     "You extract image keywords for semantic search.",
			UserPrompt:       "Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.",
			MaxTokens:        300,
			Temperature:      0.1,
			DryMultiplier:    3.0,
			DryBase:          1.75,
			DryAllowedLength: 2,
			RepeatPenalty:    1.15,
			RepeatLastN:      64,
			FrequencyPenalty: 0.5,
		},
		CategorizePrompt: CategorizePrompt{ // TODO: check if this prompt can be cached
			SystemPrompt: "You extract info from an image description and organize it into search indexes. " +
				"Reply with a JSON object with one string key and three array-of-string keys: " +
				"\"objects\" (simple list of nouns, noun chunks, and key entities, including useful synonyms - use no sentences, just terms), " +
				"\"actions\" (simple list of verbs and phrasal verbs - use no sentences, just terms in infinitive tense), and " +
				"\"attributes\" (simple list - use no sentences, just terms - of most proeminent aspects of the image, like main colors, style, lighting, mood, time of day, feeling - do not add labels, do not repeat the terms listed as objects). " +
				"\"scene\" (a string: just one clause sentence (shortened, with 30 tokens at most) picking only the main subject of the image - do not add details that belong to other facets; be concise and confident), " +
				"Use only information present in the description, but filter unnecessary natural speech words to build just objective lists. If some facet do not apply, live it empty.",
			UserPrompt:  "Categorize this image description into search facets:",
			MaxTokens:   300,
			Temperature: 0.1,
		},
		Image: ImageConfig{
			MaxSide: 512,
		},
	}
}

// Load reads the config from disk. Starts with defaults, then applies any
// JSON overrides found in the config file. Returns an error if the config
// file exists but contains invalid JSON.
func Load() (Config, error) {
	cfg := Defaults()

	data, err := os.ReadFile(configPath())
	if err != nil {
		return cfg, nil
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config %s: %w", configPath(), err)
	}

	return cfg, nil
}

// Save writes the config to disk.
func Save(cfg Config) error {
	dir := filepath.Dir(configPath())
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(configPath(), data, 0644)
}

func configPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, appDir, configFile)
}

// ParseGGMLType converts a config string to the corresponding GGMLType constant
// value used by the Kronk SDK. Returns the default (Q8_0 = 8) for unknown values.
func ParseGGMLType(s string) int32 {
	switch s {
	case "F32":
		return 50
	case "F16":
		return 1
	case "Q4_0":
		return 2
	case "Q4_1":
		return 3
	case "Q5_0":
		return 6
	case "Q5_1":
		return 7
	case "Q8_0":
		return 8
	case "BF16":
		return 30
	default:
		return 8
	}
}
