// Package app provides HTTP handlers for the LocalLens API.
package app

import (
	"context"
	"encoding/json"
	"io/fs"
	"net/http"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service"
)

// Handlers holds dependencies for all HTTP handlers.
type Handlers struct {
	log logger.Logger
	svc *service.Service
}

// New creates a Handlers with the given dependencies.
func New(log logger.Logger, svc *service.Service) *Handlers {
	return &Handlers{
		log: log,
		svc: svc,
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

	mux.Handle("GET /", http.FileServerFS(staticFS))
}

func (h *Handlers) handleIndex(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Folder string `json:"folder"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Folder == "" {
		http.Error(w, "folder is required", http.StatusBadRequest)
		return
	}

	count, err := h.svc.IndexFolder(r.Context(), req.Folder)
	if err != nil {
		h.log(r.Context(), "index error", "folder", req.Folder, "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]int{"count": count})
}

func (h *Handlers) handleSearch(w http.ResponseWriter, r *http.Request) {
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

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	results, err := h.svc.Search(ctx, folder, query, k)
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
	if path != "" {
		indexedPaths = h.svc.IndexedPaths(path)
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
	folder := r.URL.Query().Get("folder")
	if folder == "" {
		http.Error(w, "folder is required", http.StatusBadRequest)
		return
	}

	count := h.svc.IndexInfo(folder)
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
