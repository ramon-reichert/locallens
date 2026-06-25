package description_test

import (
	"context"
	"testing"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/description"
)

func TestDescribe_NotLoaded(t *testing.T) {
	d := description.New(description.Config{
		Log: logger.Discard(),
	})

	_, err := d.Describe(context.Background(), "any.jpg")
	if err != description.ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}

func TestDescriber_IsLoaded_InitiallyFalse(t *testing.T) {
	d := description.New(description.Config{
		Log: logger.Discard(),
	})

	if d.IsLoaded() {
		t.Error("expected IsLoaded() == false before Load()")
	}
}

func TestDescriber_UnloadWithoutLoad(t *testing.T) {
	d := description.New(description.Config{
		Log: logger.Discard(),
	})

	if err := d.Unload(context.Background()); err != nil {
		t.Errorf("unload without load should not error: %v", err)
	}
}
