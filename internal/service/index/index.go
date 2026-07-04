// Package index provides vector storage for LocalLens.
package index

import (
	"sync"
)

// FacetEmbedding is the embedding of a single image facet (e.g. "objects").
type FacetEmbedding struct {
	Facet  string
	Vector []float32
}

// Entry represents an indexed image with its description and one embedding
// vector per non-empty facet.
type Entry struct {
	Path        string
	Description string
	Embeddings  []FacetEmbedding
}

// Index stores image embeddings.
type Index struct {
	mu        sync.Mutex
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
	idx.mu.Lock()
	defer idx.mu.Unlock()

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
	idx.mu.Lock()
	defer idx.mu.Unlock()

	return len(idx.entries)
}

// All returns all entries as a slice for searching.
func (idx *Index) All() []Entry {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	entries := make([]Entry, 0, len(idx.entries))
	for _, entry := range idx.entries {
		entries = append(entries, entry)
	}
	return entries
}
