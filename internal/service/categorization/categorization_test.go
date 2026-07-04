package categorization

import (
	"context"
	"testing"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

func TestCategorize_NotLoaded(t *testing.T) {
	c := New(Config{Log: logger.Discard()})

	_, err := c.Categorize(context.Background(), "a red bird on a branch")
	if err != ErrModelNotLoaded {
		t.Errorf("expected ErrModelNotLoaded, got %v", err)
	}
}

func TestCategorizer_IsLoaded_InitiallyFalse(t *testing.T) {
	c := New(Config{Log: logger.Discard()})

	if c.IsLoaded() {
		t.Error("expected IsLoaded() == false before Load()")
	}
}

func TestCategorizer_UnloadWithoutLoad(t *testing.T) {
	c := New(Config{Log: logger.Discard()})

	if err := c.Unload(context.Background()); err != nil {
		t.Errorf("unload without load should not error: %v", err)
	}
}

func TestParseFacets(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		want    Facets
	}{
		{
			name:  "full object",
			input: `{"scene":"a parrot on a branch","objects":["parrot","bird","branch"],"actions":["perching"],"attributes":["yellow","green","bright"]}`,
			want: Facets{
				Scene:      "a parrot on a branch",
				Objects:    []string{"parrot", "bird", "branch"},
				Actions:    []string{"perching"},
				Attributes: []string{"yellow", "green", "bright"},
			},
		},
		{
			name:  "missing keys yield zero values",
			input: `{"scene":"a beach"}`,
			want:  Facets{Scene: "a beach"},
		},
		{
			name:    "malformed json",
			input:   `{"scene": [`,
			wantErr: true,
		},
		{
			name:    "not an object",
			input:   `"just a string"`,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseFacets([]byte(tc.input))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil (facets: %+v)", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !equalFacets(got, tc.want) {
				t.Errorf("parseFacets() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestFacets_EmbedText(t *testing.T) {
	f := Facets{
		Scene:      "a parrot on a branch",
		Objects:    []string{"parrot", "branch"},
		Actions:    []string{"perching"},
		Attributes: []string{"yellow", "green"},
	}

	want := "Scene: a parrot on a branch\n" +
		"Objects: parrot, branch\n" +
		"Actions: perching\n" +
		"Attributes: yellow, green"

	if got := f.EmbedText(); got != want {
		t.Errorf("EmbedText() =\n%q\nwant\n%q", got, want)
	}
}

func TestFacets_EmbedText_OmitsEmpty(t *testing.T) {
	f := Facets{
		Objects: []string{"parrot"},
	}

	want := "Objects: parrot"
	if got := f.EmbedText(); got != want {
		t.Errorf("EmbedText() = %q, want %q", got, want)
	}
}

func TestFacets_IsEmpty(t *testing.T) {
	if !(Facets{}).IsEmpty() {
		t.Error("zero-value Facets should be empty")
	}
	if (Facets{Actions: []string{"running"}}).IsEmpty() {
		t.Error("Facets with an action should not be empty")
	}
}

func equalFacets(a, b Facets) bool {
	return a.Scene == b.Scene &&
		equalSlice(a.Objects, b.Objects) &&
		equalSlice(a.Actions, b.Actions) &&
		equalSlice(a.Attributes, b.Attributes)
}

func equalSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
