package app

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// BrowseResponse contains folders and images at a given path.
type BrowseResponse struct {
	Parent  string        `json:"parent"`
	Current string        `json:"current"`
	Folders []FolderEntry `json:"folders"`
	Images  []ImageEntry  `json:"images"`
}

// FolderEntry represents a folder in the browse response.
type FolderEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// ImageEntry represents an image file with its indexed status.
type ImageEntry struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Indexed bool   `json:"indexed"`
}

var imageExts = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true,
	".gif": true, ".webp": true, ".bmp": true,
}

func browse(path string, indexedPaths map[string]bool) (*BrowseResponse, error) {
	if path == "" {
		return listDrives()
	}

	path = filepath.Clean(path)

	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	resp := BrowseResponse{
		Current: path,
		Parent:  filepath.Dir(path),
	}

	if resp.Parent == path {
		resp.Parent = ""
	}

	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}

		if e.IsDir() {
			resp.Folders = append(resp.Folders, FolderEntry{
				Name: e.Name(),
				Path: filepath.Join(path, e.Name()),
			})
			continue
		}

		ext := strings.ToLower(filepath.Ext(e.Name()))
		if imageExts[ext] {
			imgPath := filepath.Join(path, e.Name())
			resp.Images = append(resp.Images, ImageEntry{
				Name:    e.Name(),
				Path:    imgPath,
				Indexed: indexedPaths[imgPath],
			})
		}
	}

	sort.Slice(resp.Folders, func(i, j int) bool {
		return strings.ToLower(resp.Folders[i].Name) < strings.ToLower(resp.Folders[j].Name)
	})
	sort.Slice(resp.Images, func(i, j int) bool {
		return strings.ToLower(resp.Images[i].Name) < strings.ToLower(resp.Images[j].Name)
	})

	return &resp, nil
}

func listDrives() (*BrowseResponse, error) {
	resp := &BrowseResponse{Current: ""}

	if runtime.GOOS != "windows" {
		resp.Folders = []FolderEntry{{Name: "/", Path: "/"}}
		return resp, nil
	}

	for letter := 'A'; letter <= 'Z'; letter++ {
		drive := string(letter) + `:\`
		if _, err := os.Stat(drive); err == nil {
			resp.Folders = append(resp.Folders, FolderEntry{
				Name: drive,
				Path: drive,
			})
		}
	}

	return resp, nil
}
