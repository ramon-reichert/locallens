// Package config manages LocalLens application configuration.
package config

import (
	"encoding/json"
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
	VisionModelURL string `json:"visionModelURL"`
	VisionProjURL  string `json:"visionProjURL"`
	EmbedModelURL  string `json:"embedModelURL"`
}

// VisionModelID returns the model ID derived from the vision model URL.
func (m Models) VisionModelID() string {
	return modelIDFromURL(m.VisionModelURL)
}

// EmbedModelID returns the model ID derived from the embedding model URL.
func (m Models) EmbedModelID() string {
	return modelIDFromURL(m.EmbedModelURL)
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

// =========================================================================
// Vision prompt and image config

// VisionPrompt holds prompt configuration for image description.
type VisionPrompt struct {
	SystemPrompt string  `json:"systemPrompt"`
	UserPrompt   string  `json:"userPrompt"`
	MaxTokens    int     `json:"maxTokens"`
	Temperature  float64 `json:"temperature"`
}

// ImageConfig holds image preprocessing settings.
type ImageConfig struct {
	MaxSide int `json:"maxSide"`
	Quality int `json:"quality"`
}

// =========================================================================
// Top-level config

// Config holds the full application configuration.
type Config struct {
	BasePath        string            `json:"basePath"`
	Processor       string            `json:"processor"`
	LlamaCppVersion string           `json:"llamaCppVersion"`
	Models         Models            `json:"models"`
	Vision        VisionModelConfig `json:"visionModel"`
	Embed         EmbedModelConfig  `json:"embedModel"`
	Prompt        VisionPrompt      `json:"prompt"`
	Image         ImageConfig       `json:"image"`
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
	return filepath.Join(home, ".kronk")
}

// Defaults returns the default configuration with sensible values.
func Defaults() Config {
	return Config{
		BasePath: DefaultBasePath(),
		Models: Models{
			VisionModelURL: "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
			VisionProjURL:  "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
			EmbedModelURL:  "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf",
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
		Prompt: VisionPrompt{
			SystemPrompt: "You extract image keywords for semantic search.",
			UserPrompt:   "Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.",
			MaxTokens:    300,
			Temperature:  0.1,
		},
		Image: ImageConfig{
			MaxSide: 64,
			Quality: 90,
		},
	}
}

// Load reads the config from disk. Starts with defaults, then applies any
// JSON overrides found in the config file.
func Load() Config {
	cfg := Defaults()

	data, err := os.ReadFile(configPath())
	if err != nil {
		return cfg
	}

	json.Unmarshal(data, &cfg)
	return cfg
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
