// Package config manages LocalLens application configuration.
package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

const (
	appDir     = ".locallens"
	configFile = "config.json"
)

// Config holds the application configuration.
type Config struct {
	BasePath      string `json:"basePath"`
	SetupComplete bool   `json:"setupComplete"`
}

// DefaultBasePath returns the default base path for model storage.
func DefaultBasePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".kronk")
}

// Load reads the config from disk. Returns a default config if the file
// does not exist.
func Load() Config {
	cfg := Config{
		BasePath: DefaultBasePath(),
	}

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
