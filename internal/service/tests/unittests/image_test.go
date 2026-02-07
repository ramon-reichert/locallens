package unittests

import (
	"bytes"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	"github.com/ramon-reichert/locallens/internal/service/image"
)

const testdataDir = "../testdata"

func TestResize_JPEG(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "forest.jpg")

	data, err := image.Resize(imgPath, image.DefaultMaxSide)
	if err != nil {
		t.Fatalf("resize: %v", err)
	}

	if len(data) == 0 {
		t.Error("expected non-empty output")
	}

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}

	bounds := img.Bounds()
	if bounds.Dx() > image.DefaultMaxSide && bounds.Dy() > image.DefaultMaxSide {
		t.Errorf("expected max side <= %d, got %dx%d", image.DefaultMaxSide, bounds.Dx(), bounds.Dy())
	}
}

func TestResize_PreservesAspectRatio(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "parrot.jpg")

	data, err := image.Resize(imgPath, 200)
	if err != nil {
		t.Fatalf("resize: %v", err)
	}

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}

	bounds := img.Bounds()
	if bounds.Dx() > 200 && bounds.Dy() > 200 {
		t.Errorf("expected at least one side <= 200, got %dx%d", bounds.Dx(), bounds.Dy())
	}
}

func TestResize_CustomMaxSide(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "vietnam.jpg")

	maxSide := 128

	data, err := image.Resize(imgPath, maxSide)
	if err != nil {
		t.Fatalf("resize: %v", err)
	}

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}

	bounds := img.Bounds()
	if bounds.Dx() > maxSide && bounds.Dy() > maxSide {
		t.Errorf("expected max side <= %d, got %dx%d", maxSide, bounds.Dx(), bounds.Dy())
	}
}

func TestResize_OutputIsJPEG(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "lighthouse.jpg")

	data, err := image.Resize(imgPath, image.DefaultMaxSide)
	if err != nil {
		t.Fatalf("resize: %v", err)
	}

	_, err = jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Errorf("output is not valid JPEG: %v", err)
	}
}

func TestResize_FileNotFound(t *testing.T) {
	_, err := image.Resize("nonexistent.jpg", image.DefaultMaxSide)
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestResize_InvalidImage(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "invalid_*.jpg")
	if err != nil {
		t.Fatalf("create temp: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	tmpFile.WriteString("not an image")
	tmpFile.Close()

	_, err = image.Resize(tmpFile.Name(), image.DefaultMaxSide)
	if err == nil {
		t.Error("expected error for invalid image")
	}
}
