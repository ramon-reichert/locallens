// Package main provides the entry point for LocalLens, a local semantic
// image search application.
package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"net/http"
	"os"

	"github.com/ramon-reichert/locallens/internal/app"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/platform/web"
)

//go:embed static
var staticFiles embed.FS

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()
	log := logger.New()

	handlers := app.New(log)
	defer handlers.Close(ctx)

	mux := http.NewServeMux()

	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return fmt.Errorf("static fs: %w", err)
	}

	handlers.Register(mux, staticFS)

	srv := web.New(web.Config{
		Log:  log,
		Mux:  mux,
		Host: "localhost",
		Port: "8080",
	})

	return srv.ListenAndServe()
}
