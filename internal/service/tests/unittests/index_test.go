package tests

import (
	"path/filepath"
	"testing"

	"github.com/ramon-reichert/locallens/internal/service/index"
)

func TestAddAndGet(t *testing.T) {
	idx := index.New("")

	idx.Add(index.Entry{
		Path:        "dog.jpg",
		Description: "A brown dog running through a green field",
		Embedding:   []float32{1, 0, 0, 0},
	})
	idx.Add(index.Entry{
		Path:        "cat.jpg",
		Description: "A white cat sleeping on a couch",
		Embedding:   []float32{0, 1, 0, 0},
	})

	if idx.Len() != 2 {
		t.Errorf("expected 2 entries, got %d", idx.Len())
	}

	entry, ok := idx.Get("dog.jpg")
	if !ok {
		t.Fatal("dog.jpg not found")
	}

	if entry.Description != "A brown dog running through a green field" {
		t.Errorf("unexpected description: %s", entry.Description)
	}

	_, ok = idx.Get("notfound.jpg")
	if ok {
		t.Error("expected not found")
	}
}

func TestRemove(t *testing.T) {
	idx := index.New("")

	idx.Add(index.Entry{Path: "a.jpg", Description: "Image A", Embedding: []float32{1, 0}})
	idx.Add(index.Entry{Path: "b.jpg", Description: "Image B", Embedding: []float32{0, 1}})

	if idx.Len() != 2 {
		t.Fatalf("expected 2 entries, got %d", idx.Len())
	}

	idx.Remove("a.jpg")

	if idx.Len() != 1 {
		t.Errorf("expected 1 entry after remove, got %d", idx.Len())
	}

	_, ok := idx.Get("a.jpg")
	if ok {
		t.Error("a.jpg should be removed")
	}
}

func TestAll(t *testing.T) {
	idx := index.New("")

	idx.Add(index.Entry{Path: "a.jpg", Description: "A", Embedding: []float32{1}})
	idx.Add(index.Entry{Path: "b.jpg", Description: "B", Embedding: []float32{2}})
	idx.Add(index.Entry{Path: "c.jpg", Description: "C", Embedding: []float32{3}})

	all := idx.All()
	if len(all) != 3 {
		t.Errorf("expected 3 entries, got %d", len(all))
	}
}

func TestSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test.index")

	idx := index.New(indexPath)
	idx.Add(index.Entry{
		Path:        "photo1.jpg",
		Description: "A sunset over the ocean",
		Embedding:   []float32{0.5, 0.5, 0, 0},
	})
	idx.Add(index.Entry{
		Path:        "photo2.jpg",
		Description: "Mountains covered in snow",
		Embedding:   []float32{0, 0.5, 0.5, 0},
	})

	if err := idx.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded := index.New(indexPath)
	if err := loaded.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.Len() != 2 {
		t.Errorf("expected 2 entries after load, got %d", loaded.Len())
	}

	entry, ok := loaded.Get("photo1.jpg")
	if !ok {
		t.Fatal("photo1.jpg not found after load")
	}

	if entry.Description != "A sunset over the ocean" {
		t.Errorf("unexpected description: %s", entry.Description)
	}
}

func TestLoadNonExistent(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "nonexistent.index")

	idx := index.New(indexPath)
	if err := idx.Load(); err != nil {
		t.Errorf("load non-existent should not error: %v", err)
	}

	if idx.Len() != 0 {
		t.Errorf("expected 0 entries, got %d", idx.Len())
	}
}
