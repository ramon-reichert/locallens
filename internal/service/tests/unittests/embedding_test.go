package unittests

import (
	"context"
	"testing"
	"time"

	"github.com/ramon-reichert/locallens/internal/service/embedding"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

func TestEmbed(t *testing.T) {
	testsboot.Boot()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	e := embedding.New(embedding.Config{
		Log:   testsboot.Log,
		Paths: testsboot.EmbedPaths,
	})
	defer e.Unload(ctx)

	if err := e.Load(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}

	text := "A brown dog running through a green field"

	vec, err := e.Embed(ctx, text)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}

	if len(vec) == 0 {
		t.Error("expected non-empty embedding vector")
	}

	if vec[0] == 0 && vec[len(vec)-1] == 0 {
		t.Error("expected embedding to have non-zero values")
	}

	t.Logf("embedding dimensions: %d", len(vec))
}

func TestEmbed_NotLoaded(t *testing.T) {
	testsboot.Boot()
	e := embedding.New(embedding.Config{
		Log: testsboot.Log,
	})

	_, err := e.Embed(context.Background(), "any text")
	if err != embedding.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}

func TestEmbed_EmptyText(t *testing.T) {
	testsboot.Boot()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	e := embedding.New(embedding.Config{
		Log:   testsboot.Log,
		Paths: testsboot.EmbedPaths,
	})
	defer e.Unload(ctx)

	if err := e.Load(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err := e.Embed(ctx, "")
	if err != embedding.ErrEmptyText {
		t.Errorf("expected ErrEmptyText, got %v", err)
	}
}
