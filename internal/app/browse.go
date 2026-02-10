package app

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// BrowseResponse contains folders and images at a given path.
type BrowseResponse struct {
	Parent  string        `json:"parent"`
	Current string        `json:"current"`
	Folders []FolderEntry `json:"folders"`
	Images  []string      `json:"images"`
}

// FolderEntry represents a folder in the browse response.
type FolderEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

var imageExts = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true,
	".gif": true, ".webp": true, ".bmp": true,
}

func browse(path string) (*BrowseResponse, error) {
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
			resp.Images = append(resp.Images, filepath.Join(path, e.Name()))
		}
	}

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
