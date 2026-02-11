// Package service provides the service orchestrator for LocalLens.
package service

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ardanlabs/kronk/sdk/tools/models"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/description"
	"github.com/ramon-reichert/locallens/internal/service/embedding"
	"github.com/ramon-reichert/locallens/internal/service/index"
	"github.com/ramon-reichert/locallens/internal/service/search"
)

const (
	indexFileName       = ".locallens.index"
	describeImageTimeout = 2 * time.Minute
	embedTimeout         = 1 * time.Minute
)

// Service orchestrates indexing and search operations.
type Service struct {
	log       logger.Logger
	describer *description.Describer
	embedder  *embedding.Embedder
}

// Config holds configuration for creating a Service.
type Config struct {
	Log         logger.Logger
	VisionPaths models.Path
	EmbedPaths  models.Path
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
	}
}

// IndexFolder indexes all images in a folder.
// Creates or updates a .locallens.index file inside the folder.
func (s *Service) IndexFolder(ctx context.Context, folderPath string) (int, error) {
	images, err := findImages(folderPath)
	if err != nil {
		return 0, fmt.Errorf("find images: %w", err)
	}

	if len(images) == 0 {
		s.log(ctx, "no images found", "folder", folderPath)
		return 0, nil
	}

	start := time.Now()
	s.log(ctx, "index folder", "found images", len(images))

	idx := index.New(indexPathFor(folderPath))
	idx.Load()

	// Phase 1: Describe new images
	descriptions := make(map[string]string)

	if err := s.describer.Load(ctx); err != nil {
		return 0, fmt.Errorf("load describer: %w", err)
	}

	for _, imgPath := range images {
		if _, exists := idx.Get(imgPath); exists {
			s.log(ctx, "already indexed, skipping", "path", imgPath)
			continue
		}

		imgCtx, imgCancel := context.WithTimeout(ctx, describeImageTimeout)
		desc, err := s.describer.Describe(imgCtx, imgPath)
		imgCancel()

		if err != nil {
			s.log(ctx, "describe error", "path", imgPath, "error", err)
			continue
		}

		descriptions[imgPath] = desc
	}

	if err := s.describer.Unload(ctx); err != nil {
		return 0, fmt.Errorf("unload describer: %w", err)
	}

	// Phase 2: Embed new descriptions
	if len(descriptions) > 0 {
		if err := s.embedder.Load(ctx); err != nil {
			return 0, fmt.Errorf("load embedder: %w", err)
		}

		for imgPath, desc := range descriptions {
			s.log(ctx, "embed image", "path", imgPath)

			embedCtx, embedCancel := context.WithTimeout(ctx, embedTimeout)
			vec, err := s.embedder.Embed(embedCtx, desc)
			embedCancel()
			if err != nil {
				s.log(ctx, "embed error", "path", imgPath, "error", err)
				continue
			}

			idx.Add(index.Entry{
				Path:        imgPath,
				Description: desc,
				Embedding:   vec,
			})
		}

		if err := s.embedder.Unload(ctx); err != nil {
			return 0, fmt.Errorf("unload embedder: %w", err)
		}
	}

	if err := idx.Save(); err != nil {
		return 0, fmt.Errorf("save index: %w", err)
	}

	s.log(ctx, "index folder", "indexed images", idx.Len(), "elapsed time", time.Since(start))

	return idx.Len(), nil
}

// Search finds images similar to the query text in the given folder.
// The folder must have been indexed first.
func (s *Service) Search(ctx context.Context, folderPath string, query string, k int) ([]search.Result, error) {
	s.log(ctx, "search images", "folder", folderPath, "top k", k, "query", query)

	idx := index.New(indexPathFor(folderPath))
	if err := idx.Load(); err != nil {
		return nil, fmt.Errorf("load index: %w", err)
	}

	if idx.Len() == 0 {
		return nil, nil
	}

	if !s.embedder.IsLoaded() {
		if err := s.embedder.Load(ctx); err != nil {
			return nil, fmt.Errorf("load embedder: %w", err)
		}
	}

	embedCtx, embedCancel := context.WithTimeout(ctx, embedTimeout)
	queryVec, err := s.embedder.Embed(embedCtx, query)
	embedCancel()
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	entries := idx.All()
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

// IndexInfo returns the number of indexed images in a folder.
func (s *Service) IndexInfo(folderPath string) int {
	idx := index.New(indexPathFor(folderPath))
	idx.Load()
	return idx.Len()
}

// IndexedPaths returns the set of image paths that have been indexed in a folder.
func (s *Service) IndexedPaths(folderPath string) map[string]bool {
	idx := index.New(indexPathFor(folderPath))
	idx.Load()

	paths := make(map[string]bool, idx.Len())
	for _, e := range idx.All() {
		paths[e.Path] = true
	}
	return paths
}

// Close releases all resources.
func (s *Service) Close(ctx context.Context) error {
	if err := s.describer.Unload(ctx); err != nil {
		return err
	}
	return s.embedder.Unload(ctx)
}

func indexPathFor(folderPath string) string {
	return filepath.Join(folderPath, indexFileName)
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

		// TODO: validate image formats
		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
			images = append(images, path)
		}

		return nil
	})

	return images, err
}
