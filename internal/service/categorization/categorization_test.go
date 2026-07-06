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

func TestParseExpressions(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		want    Expressions
	}{
		{
			name:  "expressions object",
			input: `{"expressions":["bright yellow parrot","parrot perched on branch","dense tropical forest"]}`,
			want:  Expressions{"bright yellow parrot", "parrot perched on branch", "dense tropical forest"},
		},
		{
			name:  "trims blanks and caps at 15",
			input: `{"expressions":[" one "," ","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen"]}`,
			want:  Expressions{"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"},
		},
		{
			name:  "missing expressions yields empty",
			input: `{}`,
			want:  Expressions{},
		},
		{
			name:    "malformed json",
			input:   `{"expressions": [`,
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
			got, err := parseExpressions([]byte(tc.input))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil (expressions: %+v)", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !equalSlice(got, tc.want) {
				t.Errorf("parseExpressions() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestExpressions_IsEmpty(t *testing.T) {
	if !(Expressions{}).IsEmpty() {
		t.Error("zero-value Expressions should be empty")
	}
	if (Expressions{"running dog"}).IsEmpty() {
		t.Error("Expressions with a value should not be empty")
	}
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
