// Package image provides image processing utilities for vision inference.
package image

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif" // register GIF decoder
	"image/jpeg"
	_ "image/png" // register PNG decoder

	"os"

	_ "golang.org/x/image/bmp"  // register BMP decoder
	_ "golang.org/x/image/webp" // register WebP decoder

	"golang.org/x/image/draw"
)

const (
	DefaultMaxSide = 384
	DefaultQuality = 90
)

// Resize loads an image, resizes it for vision inference, and returns JPEG bytes.
// If the image is already small enough, it still re-encodes to JPEG for consistency.
func Resize(srcPath string, maxSide int) ([]byte, error) {
	img, err := load(srcPath)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	fmt.Printf("before resize: %dx%d", w, h)
	fmt.Println()

	// Scale down if needed
	if w > maxSide || h > maxSide {
		w, h = scaleDimensions(w, h, maxSide)
		dst := image.NewRGBA(image.Rect(0, 0, w, h))
		draw.CatmullRom.Scale(dst, dst.Bounds(), img, bounds, draw.Over, nil)
		img = dst
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: DefaultQuality}); err != nil {
		return nil, fmt.Errorf("encode jpeg: %w", err)
	}

	fmt.Printf("output: %dx%d, %d bytes", w, h, len(buf.Bytes()))
	fmt.Println()

	return buf.Bytes(), nil
}

func load(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	return img, nil
}

func scaleDimensions(w, h, maxSide int) (int, int) {
	if w >= h {
		newW := maxSide
		newH := int(float64(h) * float64(maxSide) / float64(w))
		return newW, newH
	}

	newH := maxSide
	newW := int(float64(w) * float64(maxSide) / float64(h))
	return newW, newH
}
