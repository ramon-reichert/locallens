package embedding_test

import (
	"context"
	"testing"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/embedding"
)

func TestEmbed_NotLoaded(t *testing.T) {
	e := embedding.New(embedding.Config{
		Log: logger.Discard(),
	})

	_, err := e.Embed(context.Background(), "any text")
	if err != embedding.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}

func TestEmbedder_IsLoaded_InitiallyFalse(t *testing.T) {
	e := embedding.New(embedding.Config{
		Log: logger.Discard(),
	})

	if e.IsLoaded() {
		t.Error("expected IsLoaded() == false before Load()")
	}
}

func TestEmbedder_UnloadWithoutLoad(t *testing.T) {
	e := embedding.New(embedding.Config{
		Log: logger.Discard(),
	})

	if err := e.Unload(context.Background()); err != nil {
		t.Errorf("unload without load should not error: %v", err)
	}
}
