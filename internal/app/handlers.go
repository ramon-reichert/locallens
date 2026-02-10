// Package app provides HTTP handlers for the LocalLens API.
package app

import (
	"encoding/json"
	"io/fs"
	"net/http"

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

	results, err := h.svc.Search(r.Context(), folder, query, 20)
	if err != nil {
		h.log(r.Context(), "search error", "query", query, "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, results)
}

func (h *Handlers) handleBrowse(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")

	resp, err := browse(path)
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
