// Package app provides HTTP handlers for the LocalLens API.
package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service"
)

// SetupStatusInfo holds the current setup state returned to the UI.
type SetupStatusInfo struct {
	BasePath    string `json:"basePath"`
	DefaultPath string `json:"defaultPath"`
}

// SetupProgress reports a setup step to the caller.
type SetupProgress func(step, status string)

// Config holds dependencies for creating Handlers.
type Config struct {
	Log         logger.Logger
	Service     *service.Service // nil if setup not yet complete.
	SetupStatus func() SetupStatusInfo
	SetupRunner func(ctx context.Context, log logger.Logger, basePath string, progress SetupProgress) (*service.Service, error)
}

// Handlers holds dependencies for all HTTP handlers.
type Handlers struct {
	log         logger.Logger
	setupStatus func() SetupStatusInfo
	setupRunner func(ctx context.Context, log logger.Logger, basePath string, progress SetupProgress) (*service.Service, error)

	mu  sync.RWMutex
	svc *service.Service
}

// New creates Handlers with the given dependencies.
func New(cfg Config) *Handlers {
	return &Handlers{
		log:         cfg.Log,
		svc:         cfg.Service,
		setupStatus: cfg.SetupStatus,
		setupRunner: cfg.SetupRunner,
	}
}

// Register registers all API routes and static file serving on the given mux.
func (h *Handlers) Register(mux *http.ServeMux, staticFS fs.FS) {
	mux.HandleFunc("POST /api/index", h.handleIndex)
	mux.HandleFunc("GET /api/search", h.handleSearch)
	mux.HandleFunc("GET /api/browse", h.handleBrowse)
	mux.HandleFunc("GET /api/images", h.handleImage)
	mux.HandleFunc("GET /api/index-info", h.handleIndexInfo)
	mux.HandleFunc("POST /api/open", h.handleOpen)
	mux.HandleFunc("GET /api/setup/status", h.handleSetupStatus)
	mux.HandleFunc("POST /api/setup/run", h.handleSetupRun)

	mux.Handle("GET /", http.FileServerFS(staticFS))
}

// Close releases service resources if initialized.
func (h *Handlers) Close(ctx context.Context) {
	h.mu.RLock()
	svc := h.svc
	h.mu.RUnlock()

	if svc != nil {
		svc.Close(ctx)
	}
}

func (h *Handlers) requireService(w http.ResponseWriter) *service.Service {
	h.mu.RLock()
	svc := h.svc
	h.mu.RUnlock()

	if svc == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "setup_required",
		})
		return nil
	}

	return svc
}

// ---- API Handlers ----

func (h *Handlers) handleIndex(w http.ResponseWriter, r *http.Request) {
	svc := h.requireService(w)
	if svc == nil {
		return
	}

	var req struct {
		Folder    string `json:"folder"`
		Recursive bool   `json:"recursive"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Folder == "" {
		http.Error(w, "folder is required", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// send writes a single SSE event. Any field absent from event is omitted
	// from the JSON payload by the encoder.
	send := func(event map[string]any) {
		buf, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", buf)
		flusher.Flush()
	}

	progress := func(p service.IndexProgressInfo) {
		send(map[string]any{
			"type":    "progress",
			"folder":  p.Folder,
			"current": p.Current,
			"done":    p.Done,
			"total":   p.Total,
			"etaMs":   p.ETA.Milliseconds(),
		})
	}

	// Flush an initial event so the client's `await fetch(...)` resolves
	// immediately. Without this, the browser blocks until the first body byte
	// is sent, which can be 30+ seconds while the vision model loads — making
	// the UI look frozen.
	send(map[string]any{"type": "started", "folder": req.Folder, "recursive": req.Recursive})

	h.log(r.Context(), "handle index", "folder", req.Folder, "recursive", req.Recursive)

	// r.Context() is cancelled when the client closes the connection (e.g.,
	// the user hits Stop in the UI). The Service checks ctx.Done() between
	// images and returns context.Canceled once the in-flight image completes.
	var (
		count int
		err   error
	)
	if req.Recursive {
		count, err = svc.IndexTree(r.Context(), req.Folder, progress)
	} else {
		count, err = svc.IndexFolder(r.Context(), req.Folder, progress)
	}

	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		send(map[string]any{"type": "cancelled", "count": count})
	case err != nil:
		h.log(r.Context(), "index error", "folder", req.Folder, "error", err)
		send(map[string]any{"type": "error", "error": err.Error(), "count": count})
	default:
		send(map[string]any{"type": "done", "count": count})
	}
}

func (h *Handlers) handleSearch(w http.ResponseWriter, r *http.Request) {
	svc := h.requireService(w)
	if svc == nil {
		return
	}

	query := r.URL.Query().Get("q")
	folder := r.URL.Query().Get("folder")

	if query == "" || folder == "" {
		http.Error(w, "q and folder are required", http.StatusBadRequest)
		return
	}

	k := 20
	if ks := r.URL.Query().Get("k"); ks != "" {
		if v, err := strconv.Atoi(ks); err == nil && v > 0 {
			k = v
		}
	}

	recursive := r.URL.Query().Get("recursive") == "true"

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	results, err := svc.Search(ctx, folder, query, k, recursive)
	if err != nil {
		h.log(r.Context(), "search error", "query", query, "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, results)
}

func (h *Handlers) handleBrowse(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")

	var indexedPaths map[string]bool

	h.mu.RLock()
	svc := h.svc
	h.mu.RUnlock()

	if path != "" && svc != nil {
		indexedPaths = svc.IndexedPaths(path)
	}

	resp, err := browse(path, indexedPaths)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

func (h *Handlers) handleImage(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")
	if path == "" {
		http.Error(w, "path is required", http.StatusBadRequest)
		return
	}

	http.ServeFile(w, r, path)
}

func (h *Handlers) handleIndexInfo(w http.ResponseWriter, r *http.Request) {
	svc := h.requireService(w)
	if svc == nil {
		return
	}

	folder := r.URL.Query().Get("folder")
	if folder == "" {
		http.Error(w, "folder is required", http.StatusBadRequest)
		return
	}

	count := svc.IndexInfo(folder)
	writeJSON(w, http.StatusOK, map[string]int{"count": count})
}

func (h *Handlers) handleOpen(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Path == "" {
		http.Error(w, "path is required", http.StatusBadRequest)
		return
	}

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/C", "start", "", "explorer.exe", "/select,"+req.Path)
	case "darwin":
		cmd = exec.Command("open", "-R", req.Path)
	default:
		cmd = exec.Command("xdg-open", filepath.Dir(req.Path))
	}

	if err := cmd.Start(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// ---- Setup Handlers ----

func (h *Handlers) handleSetupStatus(w http.ResponseWriter, r *http.Request) {
	info := h.setupStatus()

	h.mu.RLock()
	ready := h.svc != nil
	h.mu.RUnlock()

	writeJSON(w, http.StatusOK, struct {
		Complete bool   `json:"complete"`
		SetupStatusInfo
	}{
		Complete:        ready,
		SetupStatusInfo: info,
	})
}

func (h *Handlers) handleSetupRun(w http.ResponseWriter, r *http.Request) {
	var req struct {
		BasePath string `json:"basePath"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.BasePath == "" {
		http.Error(w, "basePath is required", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	send := func(step, status string) {
		fmt.Fprintf(w, "data: {\"step\":%q,\"status\":%q}\n\n", step, status)
		flusher.Flush()
	}

	svc, err := h.setupRunner(r.Context(), h.log, req.BasePath, send)
	if err != nil {
		return
	}

	h.mu.Lock()
	h.svc = svc
	h.mu.Unlock()

	send("done", "complete")
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
