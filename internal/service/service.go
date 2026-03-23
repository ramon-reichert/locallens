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

	"github.com/ramon-reichert/locallens/internal/platform/config"
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
	AppCfg      config.Config
}

// New creates a Service with the given configuration.
func New(cfg Config) *Service {
	return &Service{
		log: cfg.Log,
		describer: description.New(description.Config{
			Log:    cfg.Log,
			Paths:  cfg.VisionPaths,
			AppCfg: cfg.AppCfg,
		}),
		embedder: embedding.New(embedding.Config{
			Log:    cfg.Log,
			Paths:  cfg.EmbedPaths,
			AppCfg: cfg.AppCfg,
		}),
	}
}

// IndexFolder indexes all images in a folder.
// Creates or updates a .locallens.index file inside the folder.
// If recursive is true, it indexes each subfolder separately, storing a
// per-folder .locallens.index so that non-recursive search only sees images
// from the selected folder.
func (s *Service) IndexFolder(ctx context.Context, folderPath string, recursive bool) (int, error) {
	images, err := findImages(folderPath, recursive)
	if err != nil {
		return 0, fmt.Errorf("find images: %w", err)
	}

	if len(images) == 0 {
		s.log(ctx, "no images found", "folder", folderPath)
		return 0, nil
	}

	start := time.Now()
	s.log(ctx, "index folder", "found images", len(images))

	// Group images by their parent directory so each folder gets its own index.
	byFolder := make(map[string][]string)
	for _, imgPath := range images {
		dir := filepath.Dir(imgPath)
		byFolder[dir] = append(byFolder[dir], imgPath)
	}

	// Load all per-folder indexes.
	indexes := make(map[string]*index.Index, len(byFolder))
	for dir := range byFolder {
		idx := index.New(indexPathFor(dir))
		idx.Load()
		indexes[dir] = idx
	}

	// Phase 1: Describe new images
	descriptions := make(map[string]string)

	if err := s.describer.Load(ctx); err != nil {
		return 0, fmt.Errorf("load describer: %w", err)
	}

	for _, imgPath := range images {
		dir := filepath.Dir(imgPath)
		if _, exists := indexes[dir].Get(imgPath); exists {
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

			dir := filepath.Dir(imgPath)
			indexes[dir].Add(index.Entry{
				Path:        imgPath,
				Description: desc,
				Embedding:   vec,
			})
		}

		if err := s.embedder.Unload(ctx); err != nil {
			return 0, fmt.Errorf("unload embedder: %w", err)
		}
	}

	// Save each per-folder index.
	total := 0
	for _, idx := range indexes {
		if err := idx.Save(); err != nil {
			return 0, fmt.Errorf("save index: %w", err)
		}
		total += idx.Len()
	}

	s.log(ctx, "index folder", "indexed images", total, "elapsed time", time.Since(start))

	return total, nil
}

// Search finds images similar to the query text in the given folder.
// The folder must have been indexed first.
// If recursive is true, it searches across all indexed subfolders.
func (s *Service) Search(ctx context.Context, folderPath string, query string, k int, recursive bool) ([]search.Result, error) {
	s.log(ctx, "search images", "folder", folderPath, "top k", k, "recursive", recursive, "query", query)

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

	var allEntries []search.Entry

	if recursive {
		// Walk all subdirectories and collect entries from each index.
		filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
			if err != nil || !info.IsDir() {
				return nil
			}
			idxPath := indexPathFor(path)
			if _, statErr := os.Stat(idxPath); os.IsNotExist(statErr) {
				return nil
			}

			idx := index.New(idxPath)
			if loadErr := idx.Load(); loadErr != nil {
				return nil
			}

			for _, e := range idx.All() {
				allEntries = append(allEntries, search.Entry{
					Path:        e.Path,
					Description: e.Description,
					Embedding:   e.Embedding,
				})
			}
			return nil
		})
	} else {
		idx := index.New(indexPathFor(folderPath))
		if err := idx.Load(); err != nil {
			return nil, fmt.Errorf("load index: %w", err)
		}

		for _, e := range idx.All() {
			allEntries = append(allEntries, search.Entry{
				Path:        e.Path,
				Description: e.Description,
				Embedding:   e.Embedding,
			})
		}
	}

	if len(allEntries) == 0 {
		return nil, nil
	}

	return search.FindTopK(queryVec, allEntries, k), nil
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

func findImages(folderPath string, recursive bool) ([]string, error) {
	var images []string

	if !recursive {
		entries, err := os.ReadDir(folderPath)
		if err != nil {
			return nil, err
		}

		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			ext := strings.ToLower(filepath.Ext(e.Name()))
			switch ext {
			case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
				images = append(images, filepath.Join(folderPath, e.Name()))
			}
		}

		return images, nil
	}

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
