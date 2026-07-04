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

func TestFacets_FacetTexts(t *testing.T) {
	f := Facets{
		Scene:      "a parrot on a branch",
		Objects:    []string{"parrot", "branch"},
		Actions:    []string{"perching"},
		Attributes: []string{"yellow", "green"},
	}

	want := []FacetText{
		{Name: "scene", Text: "a parrot on a branch"},
		{Name: "objects", Text: "parrot, branch"},
		{Name: "actions", Text: "perching"},
		{Name: "attributes", Text: "yellow, green"},
	}

	got := f.FacetTexts()
	if len(got) != len(want) {
		t.Fatalf("FacetTexts() returned %d entries, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("FacetTexts()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestFacets_FacetTexts_OmitsEmpty(t *testing.T) {
	f := Facets{
		Scene:   "   ", // whitespace-only, treated as empty
		Objects: []string{"parrot"},
	}

	got := f.FacetTexts()
	if len(got) != 1 {
		t.Fatalf("expected 1 non-empty facet, got %d: %+v", len(got), got)
	}
	if got[0] != (FacetText{Name: "objects", Text: "parrot"}) {
		t.Errorf("FacetTexts()[0] = %+v, want objects/parrot", got[0])
	}
}

func TestTrimScene(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		maxWords int
		want     string
	}{
		{
			name:     "under the cap is unchanged",
			in:       "a parrot on a branch",
			maxWords: 20,
			want:     "a parrot on a branch",
		},
		{
			name:     "long scene is cut to maxWords",
			in:       "A serene forest scene during late autumn or early winter, with tall slender trees everywhere around",
			maxWords: 6,
			want:     "A serene forest scene during late",
		},
		{
			name:     "trailing comma is dropped after the cut",
			in:       "A serene forest scene, with trees",
			maxWords: 4,
			want:     "A serene forest scene",
		},
		{
			name:     "zero disables trimming (only trims whitespace)",
			in:       "  a very long scene that keeps going and going  ",
			maxWords: 0,
			want:     "a very long scene that keeps going and going",
		},
		{
			name:     "exact word count is unchanged",
			in:       "one two three",
			maxWords: 3,
			want:     "one two three",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := trimScene(tc.in, tc.maxWords); got != tc.want {
				t.Errorf("trimScene(%q, %d) = %q, want %q", tc.in, tc.maxWords, got, tc.want)
			}
		})
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
