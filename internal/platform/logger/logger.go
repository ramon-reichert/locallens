// Package logger provides logging utilities for LocalLens.
package logger

import (
	"context"
	"fmt"
)

// Logger provides a function for logging messages with key-value pairs.
type Logger func(ctx context.Context, msg string, args ...any)

// New returns a standard logger that writes to stdout.
func New() Logger {
	return func(ctx context.Context, msg string, args ...any) {
		fmt.Printf("%s:", msg)
		for i := 0; i < len(args); i += 2 {
			if i+1 < len(args) {
				fmt.Printf(" %v[%v]", args[i], args[i+1])
			}
		}
		fmt.Println()
	}
}

// Discard returns a logger that discards all output.
func Discard() Logger {
	return func(ctx context.Context, msg string, args ...any) {}
}
