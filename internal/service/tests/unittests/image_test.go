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
	imgPath := filepath.Join(testdataDir, "353kpx_63kb.jpg")
	if _, err := os.Stat(imgPath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}

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

	t.Logf("output: %dx%d, %d bytes", bounds.Dx(), bounds.Dy(), len(data))
}

func TestResize_PNG(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "12kpx_22kb.png")
	if _, err := os.Stat(imgPath); os.IsNotExist(err) {
		t.Fatalf("test image not found")
	}

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
	t.Logf("output: %dx%d, %d bytes", bounds.Dx(), bounds.Dy(), len(data))
}

func TestResize_SmallImage(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "12kpx_22kb.png")
	if _, err := os.Stat(imgPath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}

	data, err := image.Resize(imgPath, 2000)
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
	t.Logf("small image output: %dx%d, %d bytes", bounds.Dx(), bounds.Dy(), len(data))
}

func TestResize_CustomMaxSide(t *testing.T) {
	imgPath := filepath.Join(testdataDir, "353kpx_63kb.jpg")
	if _, err := os.Stat(imgPath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}

	maxSide := 200

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

	t.Logf("custom maxSide output: %dx%d, %d bytes", bounds.Dx(), bounds.Dy(), len(data))
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
