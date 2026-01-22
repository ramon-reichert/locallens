package index

import (
	"encoding/gob"
	"fmt"
	"os"
)

// Save writes the index to disk using gob encoding.
func (idx *Index) Save() error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	f, err := os.Create(idx.indexPath)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	if err := enc.Encode(idx.entries); err != nil {
		return fmt.Errorf("encode: %w", err)
	}

	return nil
}

// Load reads the index from disk. If the file doesn't exist, the index remains empty.
func (idx *Index) Load() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, err := os.Stat(idx.indexPath); os.IsNotExist(err) {
		return nil
	}

	f, err := os.Open(idx.indexPath)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	var entries map[string]Entry
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&entries); err != nil {
		return fmt.Errorf("decode: %w", err)
	}

	idx.entries = entries
	return nil
}
