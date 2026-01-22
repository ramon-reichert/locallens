// Package index provides vector storage for LocalLens.
package index

import (
	"sync"
)

// Entry represents an indexed image with its description and embedding.
type Entry struct {
	Path        string
	Description string
	Embedding   []float32
}

// Index stores image embeddings.
type Index struct {
	mu        sync.RWMutex
	entries   map[string]Entry // key: image path
	indexPath string
}

// New creates an Index with the given storage path.
func New(indexPath string) *Index {
	return &Index{
		entries:   make(map[string]Entry),
		indexPath: indexPath,
	}
}

// Add adds or updates an entry in the index.
func (idx *Index) Add(entry Entry) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.entries[entry.Path] = entry
}

// Get retrieves an entry by path.
func (idx *Index) Get(path string) (Entry, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	entry, ok := idx.entries[path]
	return entry, ok
}

// Remove deletes an entry from the index.
func (idx *Index) Remove(path string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	delete(idx.entries, path)
}

// Len returns the number of entries in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return len(idx.entries)
}

// All returns all entries as a slice for searching.
func (idx *Index) All() []Entry {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	entries := make([]Entry, 0, len(idx.entries))
	for _, entry := range idx.entries {
		entries = append(entries, entry)
	}
	return entries
}
