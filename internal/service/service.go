// Package service provides the service orchestrator for LocalLens.
package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/description"
	"github.com/ramon-reichert/locallens/internal/service/embedding"
	"github.com/ramon-reichert/locallens/internal/service/index"
	"github.com/ramon-reichert/locallens/internal/service/search"
)

const (
	indexFileName        = ".locallens.index"
	describeImageTimeout = 2 * time.Minute
	embedTimeout         = 1 * time.Minute
)

// Service orchestrates indexing and search operations.
type Service struct {
	log       logger.Logger
	describer *description.Describer
	embedder  *embedding.Embedder

	// indexes caches per-folder indexes loaded from disk so repeat searches
	// avoid reading and deserializing .locallens.index files. Keyed by the
	// cleaned absolute folder path.
	mu      sync.RWMutex
	indexes map[string]*index.Index
}

// Config holds configuration for creating a Service.
type Config struct {
	Log         logger.Logger
	VisionPaths config.ModelFilePaths
	EmbedPaths  config.ModelFilePaths
	AppCfg      config.Config
}

// New creates a Service with the given configuration and loads the embedding
// model. The embedder is kept loaded for the lifetime of the Service because
// every Search call needs it to vectorize the query, and per-image embedding
// during indexing reuses the same loaded model.
func New(ctx context.Context, cfg Config) (*Service, error) {
	s := &Service{
		log: cfg.Log,
		describer: description.New(description.Config{
			Log:     cfg.Log,
			Paths:   cfg.VisionPaths,
			Vision:  cfg.AppCfg.Vision,
			Prompt:  cfg.AppCfg.Prompt,
			MaxSide: cfg.AppCfg.Image.MaxSide,
		}),
		embedder: embedding.New(embedding.Config{
			Log:   cfg.Log,
			Paths: cfg.EmbedPaths,
			Embed: cfg.AppCfg.Embed,
		}),
		indexes: make(map[string]*index.Index),
	}

	if err := s.embedder.Load(ctx); err != nil {
		return nil, fmt.Errorf("load embedder: %w", err)
	}

	return s, nil
}

// IndexProgressInfo describes the state of an in-progress indexing operation.
// Done counts images successfully described+embedded+saved so far in this run.
// Total counts only images that need new work — already-indexed images that
// were skipped on resume don't contribute to either Done or Total.
type IndexProgressInfo struct {
	Folder  string        // current folder being processed
	Current string        // path of the image just processed
	Done    int           // newly indexed images so far
	Total   int           // total new images to index across all folders in this call
	ETA     time.Duration // estimated remaining time, based on running average
}

// IndexProgress is invoked once after each image is fully indexed and saved.
// Pass nil to disable progress reporting.
type IndexProgress func(IndexProgressInfo)

// IndexFolder indexes the images directly inside folderPath (non-recursive).
// It loads the vision model, processes the folder, saves the .locallens.index
// file after every image, and unloads the vision model. Each call is a
// self-contained, crash-safe unit of work.
//
// If progress is non-nil, it is invoked after each image is saved with a
// running ETA. If ctx is cancelled, indexing returns ctx.Err() after the
// in-flight image; all images saved up to that point are durable.
//
// For recursive indexing across a directory tree, use IndexTree, which loads
// the vision model once and calls the same per-folder primitive for each
// subdirectory containing images.
func (s *Service) IndexFolder(ctx context.Context, folderPath string, progress IndexProgress) (int, error) {
	total, err := s.countNewImages([]string{folderPath})
	if err != nil {
		return 0, err
	}

	// Nothing new to index — skip loading the vision model entirely. This is
	// the common "user re-runs Index on an already-indexed folder" case, which
	// otherwise pays a 30s+ vision-model load/unload for no work.
	if total == 0 {
		idx, err := s.loadIndex(folderPath)
		if err != nil {
			return 0, err
		}
		s.log(ctx, "index folder", "folder", folderPath, "indexed images", idx.Len(), "new images", 0)
		return idx.Len(), nil
	}

	if err := s.describer.Load(ctx); err != nil {
		return 0, fmt.Errorf("load describer: %w", err)
	}
	defer func() {
		if err := s.describer.Unload(ctx); err != nil {
			s.log(ctx, "unload describer error", "error", err)
		}
	}()

	tracker := &indexProgressTracker{cb: progress, total: total, folder: folderPath}
	return s.indexFolder(ctx, folderPath, tracker)
}

// IndexTree walks folderPath recursively and indexes every subdirectory
// containing images. The vision model is loaded once before the walk and
// unloaded after all folders are processed, so the load/unload cost is paid
// once per user-initiated indexing action regardless of how many folders are
// touched. Progress and cancellation behave the same as IndexFolder, with
// Total spanning all subfolders.
func (s *Service) IndexTree(ctx context.Context, folderPath string, progress IndexProgress) (int, error) {
	folders, err := findImageFolders(folderPath)
	if err != nil {
		return 0, fmt.Errorf("find image folders: %w", err)
	}

	if len(folders) == 0 {
		s.log(ctx, "no images found", "folder", folderPath)
		return 0, nil
	}

	total, err := s.countNewImages(folders)
	if err != nil {
		return 0, err
	}

	// Sum the existing entries across all folders so we can return a useful
	// count even when nothing new needs work.
	indexed := 0
	for _, dir := range folders {
		idx, err := s.loadIndex(dir)
		if err != nil {
			return 0, err
		}
		indexed += idx.Len()
	}

	if total == 0 {
		s.log(ctx, "index tree", "folder", folderPath, "indexed images", indexed, "new images", 0)
		return indexed, nil
	}

	if err := s.describer.Load(ctx); err != nil {
		return indexed, fmt.Errorf("load describer: %w", err)
	}
	defer func() {
		if err := s.describer.Unload(ctx); err != nil {
			s.log(ctx, "unload describer error", "error", err)
		}
	}()

	tracker := &indexProgressTracker{cb: progress, total: total}

	// Re-tally from scratch so the returned count reflects only what was
	// processed in this call across the folders walked.
	indexed = 0
	var firstErr error
	for _, dir := range folders {
		if ctxErr := ctx.Err(); ctxErr != nil {
			firstErr = ctxErr
			break
		}
		tracker.folder = dir
		n, err := s.indexFolder(ctx, dir, tracker)
		indexed += n
		if err != nil && firstErr == nil {
			firstErr = err
			// On cancellation, stop walking further folders.
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				break
			}
		}
	}

	return indexed, firstErr
}

// countNewImages returns the total number of images across the given folders
// that are not yet present in their per-folder index. Used to compute the
// Total field of IndexProgressInfo before starting work.
func (s *Service) countNewImages(folders []string) (int, error) {
	total := 0
	for _, dir := range folders {
		images, err := findImagesIn(dir)
		if err != nil {
			return 0, fmt.Errorf("find images in %q: %w", dir, err)
		}
		idx, err := s.loadIndex(dir)
		if err != nil {
			return 0, fmt.Errorf("load index %q: %w", dir, err)
		}
		for _, img := range images {
			if _, exists := idx.Get(img); !exists {
				total++
			}
		}
	}
	return total, nil
}

// indexProgressTracker accumulates per-image timing for ETA computation and
// forwards progress to a user-supplied callback. Shared by IndexFolder (one
// folder) and IndexTree (many folders) so the caller sees a single unified
// counter regardless of how many folders are walked.
type indexProgressTracker struct {
	cb         IndexProgress
	folder     string
	total      int
	done       int
	sumElapsed time.Duration
}

func (t *indexProgressTracker) record(imgPath string, imgElapsed time.Duration) {
	if t == nil {
		return
	}
	t.done++
	t.sumElapsed += imgElapsed
	if t.cb == nil {
		return
	}
	var eta time.Duration
	if t.done > 0 && t.total > t.done {
		avg := t.sumElapsed / time.Duration(t.done)
		eta = avg * time.Duration(t.total-t.done)
	}
	t.cb(IndexProgressInfo{
		Folder:  t.folder,
		Current: imgPath,
		Done:    t.done,
		Total:   t.total,
		ETA:     eta,
	})
}

// indexFolder indexes the images directly inside folderPath. The vision model
// must already be loaded; the embedder is kept loaded by the Service for the
// app's lifetime. This is the single-folder primitive shared by IndexFolder
// and IndexTree. The tracker (may be nil) is updated after each saved image.
//
// If ctx is cancelled, the loop returns ctx.Err() at the start of the next
// iteration; images already saved remain durable.
func (s *Service) indexFolder(ctx context.Context, folderPath string, tracker *indexProgressTracker) (int, error) {
	images, err := findImagesIn(folderPath)
	if err != nil {
		return 0, fmt.Errorf("find images: %w", err)
	}

	if len(images) == 0 {
		s.log(ctx, "no images found", "folder", folderPath)
		return 0, nil
	}

	start := time.Now()
	s.log(ctx, "index folder", "folder", folderPath, "found images", len(images))

	idx, err := s.loadIndex(folderPath)
	if err != nil {
		return 0, fmt.Errorf("load index %q: %w", folderPath, err)
	}

	// Per-image describe → embed → add → save. Saving after each image makes
	// progress durable: a crash after image N leaves images 1..N persisted,
	// and the existing-entry skip below resumes from N+1 on restart.
	var sumTTFT, sumTPS, sumEmbedMS float64
	described := 0

	for _, imgPath := range images {
		if ctxErr := ctx.Err(); ctxErr != nil {
			s.log(ctx, "index folder cancelled", "folder", folderPath, "indexed images", idx.Len(), "described in run", described)
			return idx.Len(), ctxErr
		}

		if _, exists := idx.Get(imgPath); exists {
			s.log(ctx, "already indexed, skipping", "path", imgPath)
			continue
		}

		imgStart := time.Now()

		imgCtx, imgCancel := context.WithTimeout(ctx, describeImageTimeout)
		descResult, err := s.describer.Describe(imgCtx, imgPath)
		imgCancel()
		if err != nil {
			s.log(ctx, "describe error", "path", imgPath, "error", err)
			continue
		}

		s.log(ctx, "embed image", "path", imgPath)

		embedCtx, embedCancel := context.WithTimeout(ctx, embedTimeout)
		embedResult, err := s.embedder.Embed(embedCtx, descResult.Description)
		embedCancel()
		if err != nil {
			s.log(ctx, "embed error", "path", imgPath, "error", err)
			continue
		}

		idx.Add(index.Entry{
			Path:        imgPath,
			Description: descResult.Description,
			Embedding:   embedResult.Embedding,
		})

		if err := idx.Save(); err != nil {
			return idx.Len(), fmt.Errorf("save index: %w", err)
		}

		sumTTFT += descResult.TimeToFirstTokenMS
		sumTPS += descResult.TokensPerSecond
		sumEmbedMS += float64(embedResult.Elapsed.Milliseconds())
		described++

		tracker.record(imgPath, time.Since(imgStart))
	}

	total := idx.Len()

	// Log summary with timing breakdown from Kronk metrics.
	if described > 0 {
		n := float64(described)
		s.log(ctx, "index folder",
			"folder", folderPath,
			"indexed images", total,
			"described", described,
			"avg ttft ms", sumTTFT/n,
			"avg tok/s", sumTPS/n,
			"avg embed ms", sumEmbedMS/n,
			"elapsed time", time.Since(start))
	} else {
		s.log(ctx, "\n=============\nindex folder", "folder", folderPath, "indexed images", total, "elapsed time", time.Since(start))
	}

	return total, nil
}

// Search finds images similar to the query text in the given folder.
// The folder must have been indexed first.
// If recursive is true, it searches across all indexed subfolders.
func (s *Service) Search(ctx context.Context, folderPath string, query string, k int, recursive bool) ([]search.Result, error) {
	s.log(ctx, "search images", "folder", folderPath, "top k", k, "recursive", recursive, "query", query)

	embedCtx, embedCancel := context.WithTimeout(ctx, embedTimeout)
	embedResult, err := s.embedder.Embed(embedCtx, query)
	embedCancel()
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	queryVec := embedResult.Embedding

	var allEntries []search.Entry

	if recursive {
		// Walk all subdirectories and collect entries from each cached index.
		// Folders without an on-disk index file are skipped to avoid polluting
		// the cache with empty entries for un-indexed directories.
		filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
			if err != nil || !info.IsDir() {
				return nil
			}
			if _, statErr := os.Stat(indexPathFor(path)); os.IsNotExist(statErr) {
				return nil
			}

			idx, loadErr := s.loadIndex(path)
			if loadErr != nil {
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
		idx, err := s.loadIndex(folderPath)
		if err != nil {
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
	idx := index.New(indexPathFor(folderPath)) // TODO: Worth read from the cache?
	idx.Load()
	return idx.Len()
}

// IndexedPaths returns the set of image paths that have been indexed in a folder.
func (s *Service) IndexedPaths(folderPath string) map[string]bool {
	idx := index.New(indexPathFor(folderPath)) // TODO: Worth read from the cache?
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

// loadIndex returns the cached index for folderPath, loading it from disk on
// first access. The returned pointer is shared with the cache, so subsequent
// Add/Save calls on it are visible to later searches without any extra work.
// Folders that have no .locallens.index file yet still get a cached empty
// Index — this is the desired behavior for IndexFolder, which then populates
// and saves it. Search filters un-indexed folders out before calling this.
func (s *Service) loadIndex(folderPath string) (*index.Index, error) {
	key := filepath.Clean(folderPath)

	s.mu.RLock()
	idx, ok := s.indexes[key]
	s.mu.RUnlock()
	if ok {
		return idx, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if idx, ok := s.indexes[key]; ok {
		return idx, nil
	}

	idx = index.New(indexPathFor(key))
	if err := idx.Load(); err != nil {
		return nil, fmt.Errorf("load: %w", err)
	}
	s.indexes[key] = idx
	return idx, nil
}

func indexPathFor(folderPath string) string {
	return filepath.Join(folderPath, indexFileName)
}

// findImagesIn returns the image files directly inside folderPath (non-recursive).
func findImagesIn(folderPath string) ([]string, error) {
	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return nil, err
	}

	var images []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if isImageExt(filepath.Ext(e.Name())) {
			images = append(images, filepath.Join(folderPath, e.Name()))
		}
	}
	return images, nil
}

// findImageFolders walks rootPath and returns every directory that contains at
// least one image file directly inside it. Used by IndexTree to enumerate the
// folders that need a per-folder .locallens.index.
func findImageFolders(rootPath string) ([]string, error) {
	seen := make(map[string]bool)
	var folders []string

	err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if !isImageExt(filepath.Ext(path)) {
			return nil
		}
		dir := filepath.Dir(path)
		if !seen[dir] {
			seen[dir] = true
			folders = append(folders, dir)
		}
		return nil
	})

	return folders, err
}

func isImageExt(ext string) bool {
	switch strings.ToLower(ext) {
	case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
		return true
	}
	return false
}
