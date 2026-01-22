// Package internal provides the service orchestrator for LocalLens.
package internal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/description"
	"github.com/ramon-reichert/locallens/internal/embedding"
	"github.com/ramon-reichert/locallens/internal/index"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/search"
)

// Service orchestrates indexing and search operations.
type Service struct {
	log       logger.Logger
	describer *description.Describer
	embedder  *embedding.Embedder
	Index     *index.Index
}

// Config holds configuration for creating a Service.
type Config struct {
	Log         logger.Logger
	VisionPaths models.Path
	EmbedPaths  models.Path
	IndexPath   string
}

// New creates a Service with the given configuration.
func New(cfg Config) *Service {
	return &Service{
		log: cfg.Log,
		describer: description.New(description.Config{
			Log:   cfg.Log,
			Paths: cfg.VisionPaths,
		}),
		embedder: embedding.New(embedding.Config{
			Log:   cfg.Log,
			Paths: cfg.EmbedPaths,
		}),
		Index: index.New(cfg.IndexPath),
	}
}

// IndexFolder indexes all images in a folder.
func (s *Service) IndexFolder(ctx context.Context, folderPath string) error {
	images, err := findImages(folderPath)
	if err != nil {
		return fmt.Errorf("find images: %w", err)
	}

	if len(images) == 0 {
		s.log(ctx, "no images found", "folder", folderPath)
		return nil
	}

	s.log(ctx, "found images", "count", len(images))

	// Phase 1: Describe all images
	descriptions := make(map[string]string)

	if err := s.describer.Load(ctx); err != nil {
		return fmt.Errorf("load describer: %w", err)
	}

	for _, imgPath := range images {
		s.log(ctx, "describing image", "path", imgPath)

		desc, err := s.describer.Describe(ctx, imgPath)
		if err != nil {
			s.log(ctx, "describe error", "path", imgPath, "error", err)
			continue
		}

		descriptions[imgPath] = desc
	}

	if err := s.describer.Unload(ctx); err != nil {
		return fmt.Errorf("unload describer: %w", err)
	}

	// Phase 2: Embed all descriptions
	if err := s.embedder.Load(ctx); err != nil {
		return fmt.Errorf("load embedder: %w", err)
	}

	for imgPath, desc := range descriptions {
		s.log(ctx, "embedding description", "path", imgPath)

		vec, err := s.embedder.Embed(ctx, desc)
		if err != nil {
			s.log(ctx, "embed error", "path", imgPath, "error", err)
			continue
		}

		s.Index.Add(index.Entry{
			Path:        imgPath,
			Description: desc,
			Embedding:   vec,
		})
	}

	if err := s.embedder.Unload(ctx); err != nil {
		return fmt.Errorf("unload embedder: %w", err)
	}

	// Save index
	if err := s.Index.Save(); err != nil {
		return fmt.Errorf("save index: %w", err)
	}

	s.log(ctx, "indexing complete", "indexed", s.Index.Len())

	return nil
}

// Search finds images similar to the query text.
func (s *Service) Search(ctx context.Context, query string, k int) ([]search.Result, error) {
	if !s.embedder.IsLoaded() {
		if err := s.embedder.Load(ctx); err != nil {
			return nil, fmt.Errorf("load embedder: %w", err)
		}
	}

	queryVec, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	entries := s.Index.All()
	searchEntries := make([]search.Entry, len(entries))
	for i, e := range entries {
		searchEntries[i] = search.Entry{
			Path:        e.Path,
			Description: e.Description,
			Embedding:   e.Embedding,
		}
	}

	return search.FindTopK(queryVec, searchEntries, k), nil
}

// Close releases all resources.
func (s *Service) Close(ctx context.Context) error {
	if err := s.describer.Unload(ctx); err != nil {
		return err
	}
	return s.embedder.Unload(ctx)
}

func findImages(folderPath string) ([]string, error) {
	var images []string

	err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
			images = append(images, path)
		}

		return nil
	})

	return images, err
}
