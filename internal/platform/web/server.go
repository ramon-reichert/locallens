// Package web provides an HTTP server for the LocalLens browser UI.
package web

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"time"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
)

// Server wraps an HTTP server with graceful shutdown.
type Server struct {
	log  logger.Logger
	http *http.Server
	host string
	port string
}

// Config holds configuration for creating a Server.
type Config struct {
	Log  logger.Logger
	Mux  *http.ServeMux
	Host string
	Port string
}

// New creates a Server with the given configuration.
func New(cfg Config) *Server {
	return &Server{
		log:  cfg.Log,
		host: cfg.Host,
		port: cfg.Port,
		http: &http.Server{
			Addr:              net.JoinHostPort(cfg.Host, cfg.Port),
			Handler:           cfg.Mux,
			ReadHeaderTimeout: 5 * time.Second,
			IdleTimeout:       120 * time.Second,
		},
	}
}

// ListenAndServe starts the server and blocks until a shutdown signal is
// received (SIGINT or SIGTERM). It then gracefully shuts down the server.
func (s *Server) ListenAndServe() error {
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt)

	serverErrors := make(chan error, 1)

	go func() {
		s.log(context.Background(), "server started", "addr", s.http.Addr)
		serverErrors <- s.http.ListenAndServe()
	}()

	openBrowser("http://" + s.http.Addr)

	select {
	case err := <-serverErrors:
		if err != http.ErrServerClosed {
			return fmt.Errorf("server error: %w", err)
		}

	case sig := <-shutdown:
		s.log(context.Background(), "server shutting down", "signal", sig.String())

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := s.http.Shutdown(ctx); err != nil {
			s.http.Close()
			return fmt.Errorf("graceful shutdown failed: %w", err)
		}
	}

	return nil
}

// Addr returns the address the server is configured to listen on.
func (s *Server) Addr() string {
	return s.http.Addr
}

func openBrowser(url string) {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}

	cmd.Start()
}
