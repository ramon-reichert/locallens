// Package service provides the service orchestrator for LocalLens.
package service

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/categorization"
	"github.com/ramon-reichert/locallens/internal/service/description"
	"github.com/ramon-reichert/locallens/internal/service/embedding"
	"github.com/ramon-reichert/locallens/internal/service/index"
	"github.com/ramon-reichert/locallens/internal/service/search"
)

const (
	indexFileName        = ".locallens.index"
	describeImageTimeout = 2 * time.Minute
	categorizeTimeout    = 2 * time.Minute
	embedTimeout         = 1 * time.Minute
)

// Service orchestrates indexing and search operations.
type Service struct {
	log         logger.Logger
	describer   *description.Describer
	categorizer *categorization.Categorizer
	embedder    *embedding.Embedder

	// indexes caches per-folder indexes loaded from disk so repeat searches
	// avoid reading and deserializing .locallens.index files. Keyed by the
	// cleaned absolute folder path.
	mu      sync.RWMutex
	indexes map[string]*index.Index
}

// Config holds configuration for creating a Service.
type Config struct {
	Log             logger.Logger
	VisionPaths     config.ModelFilePaths
	CategorizePaths config.ModelFilePaths
	EmbedPaths      config.ModelFilePaths
	AppCfg          config.Config
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
		categorizer: categorization.New(categorization.Config{
			Log:    cfg.Log,
			Paths:  cfg.CategorizePaths,
			Engine: cfg.AppCfg.Categorize,
			Prompt: cfg.AppCfg.CategorizePrompt,
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
// Stage is one of:
//   - "describing": the image is about to be sent to the vision model. ETA is
//     not set (we don't yet know how long it will take). Done is the count of
//     images already fully indexed (i.e., it does not include this one).
//   - "categorized": the description was reshaped into search facets. Facets
//     carries the result so callers can display it. ETA is not set.
//   - "indexed": the image was fully described, categorized, embedded, and
//     saved. Done includes this image, and ETA is the estimated remaining time
//     based on a running average of per-image elapsed time.
//
// Already-indexed images that were skipped on resume don't trigger callbacks
// and don't contribute to either Done or Total.
type IndexProgressInfo struct {
	Stage     string                 // "describing", "categorized", "indexed", or "failed"
	Folder    string                 // current folder being processed
	Current   string                 // path of the image being processed
	Done      int                    // newly indexed images so far
	Failed    int                    // images that failed during this indexing run
	Processed int                    // images completed or failed during this indexing run
	Total     int                    // total new images to index across all folders in this call
	ETA       time.Duration          // estimated remaining time (only set when Stage == "indexed")
	Facets    *categorization.Facets // set only when Stage == "categorized"
	Error     string                 // failure reason when Stage == "failed"
}

// IndexProgress is invoked at multiple points per image: once when the image
// is about to be described (Stage="describing"), once after it has been
// categorized into facets (Stage="categorized"), and once after it has been
// fully indexed and saved (Stage="indexed"). Pass nil to disable reporting.
type IndexProgress func(IndexProgressInfo)

// IndexResult summarizes one indexing run.
type IndexResult struct {
	IndexedTotal int // total entries in the folder index after the run
	Added        int // newly described, embedded, and saved images in this run
	Failed       int // new images skipped after describe/embed errors in this run
	Total        int // new images considered in this run
}

// IndexFolder indexes the images directly inside folderPath (non-recursive).
// It loads the vision model, processes the folder, saves the .locallens.index
// file after every image, and unloads the vision model. Each call is a
// self-contained, crash-safe unit of work.
//
// If progress is non-nil, it is invoked after each image is saved with a
// running ETA. If ctx is cancelled, indexing returns ctx.Err() after the
// in-flight image; all images saved up to that point are durable.
func (s *Service) IndexFolder(ctx context.Context, folderPath string, progress IndexProgress) (IndexResult, error) {
	total, err := s.countNewImages([]string{folderPath})
	if err != nil {
		return IndexResult{}, err
	}

	// Nothing new to index — skip loading the vision model entirely.
	if total == 0 {
		idx, err := s.loadIndex(folderPath)
		if err != nil {
			return IndexResult{}, err
		}
		s.log(ctx, "index folder", "folder", folderPath, "indexed images", idx.Len(), "new images", 0)
		return IndexResult{IndexedTotal: idx.Len()}, nil
	}

	if err := s.describer.Load(ctx); err != nil {
		return IndexResult{}, fmt.Errorf("load describer: %w", err)
	}
	defer func() {
		if err := s.describer.Unload(ctx); err != nil {
			s.log(ctx, "unload describer error", "error", err)
		}
	}()

	if err := s.categorizer.Load(ctx); err != nil {
		return IndexResult{}, fmt.Errorf("load categorizer: %w", err)
	}
	defer func() {
		if err := s.categorizer.Unload(ctx); err != nil {
			s.log(ctx, "unload categorizer error", "error", err)
		}
	}()

	tracker := &indexProgressTracker{callback: progress, total: total, folder: folderPath}
	return s.indexFolder(ctx, folderPath, tracker)
}

// indexFolder indexes the images directly inside folderPath. The vision model
// must already be loaded; the embedder is kept loaded by the Service for the
// app's lifetime. This is the per-folder primitive called by IndexFolder.
// The tracker (may be nil) is updated before describe and after each saved
// image.
//
// If ctx is cancelled, the loop returns ctx.Err() at the start of the next
// iteration; images already saved remain durable.
func (s *Service) indexFolder(ctx context.Context, folderPath string, tracker *indexProgressTracker) (IndexResult, error) {
	images, err := findImagesIn(folderPath)
	if err != nil {
		return IndexResult{}, fmt.Errorf("find images: %w", err)
	}

	if len(images) == 0 {
		s.log(ctx, "no images found", "folder", folderPath)
		return IndexResult{}, nil
	}

	start := time.Now()
	s.log(ctx, "index folder", "folder", folderPath, "found images", len(images))

	idx, err := s.loadIndex(folderPath)
	if err != nil {
		return IndexResult{}, fmt.Errorf("load index %q: %w", folderPath, err)
	}

	// Per-image describe → categorize → embed → add → save. Saving after each image makes
	// progress durable: a crash after image N leaves images 1..N persisted,
	// and the existing-entry skip below resumes from N+1 on restart.
	var sumTTFT, sumTPS, sumEmbedMS float64
	described := 0
	failed := 0

	for _, imgPath := range images {
		if ctxErr := ctx.Err(); ctxErr != nil {
			s.log(ctx, "index folder cancelled", "folder", folderPath, "indexed images", idx.Len(), "described in run", described)
			return IndexResult{IndexedTotal: idx.Len(), Added: described, Failed: failed, Total: tracker.total}, ctxErr
		}

		if _, exists := idx.Get(imgPath); exists {
			s.log(ctx, "already indexed, skipping", "path", imgPath)
			continue
		}

		imgStart := time.Now()
		tracker.describing(imgPath)

		s.log(ctx, "\n::::::::::::")
		s.log(ctx, "describe image", "path", imgPath)

		imgCtx, imgCancel := context.WithTimeout(ctx, describeImageTimeout)
		descResult, err := s.describer.Describe(imgCtx, imgPath)
		imgCancel()
		if err != nil {
			s.log(ctx, "describe error", "path", imgPath, "error", err)
			failed++
			tracker.recordFailed(imgPath, time.Since(imgStart), err)
			continue
		}

		s.log(ctx, "categorize image", "path", imgPath)

		catCtx, catCancel := context.WithTimeout(ctx, categorizeTimeout)
		catResult, err := s.categorizer.Categorize(catCtx, descResult.Description)
		catCancel()
		if err != nil {
			s.log(ctx, "categorize error", "path", imgPath, "error", err)
			failed++
			tracker.recordFailed(imgPath, time.Since(imgStart), err)
			continue
		}

		tracker.categorized(imgPath, catResult.Facets)

		s.log(ctx, "embed image", "path", imgPath)

		embedCtx, embedCancel := context.WithTimeout(ctx, embedTimeout)
		embedResult, err := s.embedder.Embed(embedCtx, catResult.EmbedText)
		embedCancel()
		if err != nil {
			s.log(ctx, "embed error", "path", imgPath, "error", err)
			failed++
			tracker.recordFailed(imgPath, time.Since(imgStart), err)
			continue
		}

		s.log(ctx, "save index with new described image", "path", imgPath)

		idx.Add(index.Entry{
			Path:        imgPath,
			Description: descResult.Description,
			Embedding:   embedResult.Embedding,
		})

		if err := idx.Save(); err != nil {
			return IndexResult{IndexedTotal: idx.Len(), Added: described, Failed: failed, Total: tracker.total}, fmt.Errorf("save index: %w", err)
		}

		sumTTFT += descResult.TimeToFirstTokenMS
		sumTPS += descResult.TokensPerSecond
		sumEmbedMS += float64(embedResult.Elapsed.Milliseconds())
		described++

		tracker.indexed(imgPath, time.Since(imgStart))

		s.log(ctx, "::::::::::::")
	}

	total := idx.Len()
	result := IndexResult{IndexedTotal: total, Added: described, Failed: failed, Total: tracker.total}

	// Log summary with timing breakdown from Kronk metrics.
	if described > 0 {
		n := float64(described)
		elapsed := time.Since(start)
		s.log(ctx, "\n=============\nindex folder",
			"folder", folderPath,
			"\nindexed images", total,
			"\ndescribed", described,
			"\nfailed", failed,
			"\navg ttft ms", sumTTFT/n,
			"\navg tok/s", sumTPS/n,
			"\navg embed ms", sumEmbedMS/n,
			"\nelapsed time", elapsed,
			"\navg time/img", elapsed/time.Duration(n))
	} else {
		s.log(ctx, "\n=============\nindex folder", "folder", folderPath, "\nindexed images", total, "\ndescribed", described, "\nfailed", failed, "\nelapsed time", time.Since(start))
	}

	return result, nil
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
// forwards progress to a user-supplied callback. Created once per IndexFolder
// call with a fixed folder, total, and progress callback.
type indexProgressTracker struct {
	callback   IndexProgress
	folder     string
	total      int
	done       int
	failed     int
	sumElapsed time.Duration
}

func (t *indexProgressTracker) describing(imgPath string) {
	if t == nil || t.callback == nil {
		return
	}
	t.callback(IndexProgressInfo{
		Stage:     "describing",
		Folder:    t.folder,
		Current:   imgPath,
		Done:      t.done,
		Failed:    t.failed,
		Processed: t.done + t.failed,
		Total:     t.total,
	})
}

func (t *indexProgressTracker) categorized(imgPath string, facets categorization.Facets) {
	if t == nil || t.callback == nil {
		return
	}
	f := facets
	t.callback(IndexProgressInfo{
		Stage:     "categorized",
		Folder:    t.folder,
		Current:   imgPath,
		Done:      t.done,
		Failed:    t.failed,
		Processed: t.done + t.failed,
		Total:     t.total,
		Facets:    &f,
	})
}

func (t *indexProgressTracker) indexed(imgPath string, imgElapsed time.Duration) {
	if t == nil {
		return
	}
	t.done++
	t.sumElapsed += imgElapsed
	if t.callback == nil {
		return
	}
	var eta time.Duration
	if t.done > 0 && t.total > t.done {
		avg := t.sumElapsed / time.Duration(t.done)
		eta = avg * time.Duration(t.total-t.done)
	}
	t.callback(IndexProgressInfo{
		Stage:     "indexed",
		Folder:    t.folder,
		Current:   imgPath,
		Done:      t.done,
		Failed:    t.failed,
		Processed: t.done + t.failed,
		Total:     t.total,
		ETA:       eta,
	})
}

func (t *indexProgressTracker) recordFailed(imgPath string, imgElapsed time.Duration, err error) {
	if t == nil {
		return
	}
	t.failed++
	t.sumElapsed += imgElapsed
	if t.callback == nil {
		return
	}
	var eta time.Duration
	processed := t.done + t.failed
	if processed > 0 && t.total > processed {
		avg := t.sumElapsed / time.Duration(processed)
		eta = avg * time.Duration(t.total-processed)
	}
	t.callback(IndexProgressInfo{
		Stage:     "failed",
		Folder:    t.folder,
		Current:   imgPath,
		Done:      t.done,
		Failed:    t.failed,
		Processed: processed,
		Total:     t.total,
		ETA:       eta,
		Error:     err.Error(),
	})
}

// Search finds images similar to the query text in the given folder.
// The folder must have been indexed first.
func (s *Service) Search(ctx context.Context, folderPath string, query string, k int) ([]search.Result, error) {

	s.log(ctx, "\n::::::::::::")
	s.log(ctx, "search images", "folder", folderPath, "top k", k, "query", query)

	embedCtx, embedCancel := context.WithTimeout(ctx, embedTimeout)
	embedResult, err := s.embedder.Embed(embedCtx, query)
	embedCancel()
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	queryVec := embedResult.Embedding

	idx, err := s.loadIndex(folderPath)
	if err != nil {
		return nil, fmt.Errorf("load index: %w", err)
	}

	entries := idx.All()
	if len(entries) == 0 {
		return nil, nil
	}

	searchEntries := make([]search.Entry, 0, len(entries))
	for _, e := range entries {
		searchEntries = append(searchEntries, search.Entry{
			Path:        e.Path,
			Description: e.Description,
			Embedding:   e.Embedding,
		})
	}

	s.log(ctx, "::::::::::::")

	return search.FindTopK(queryVec, searchEntries, k), nil
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
	if err := s.categorizer.Unload(ctx); err != nil {
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

func isImageExt(ext string) bool {
	switch strings.ToLower(ext) {
	case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
		return true
	}
	return false
}
