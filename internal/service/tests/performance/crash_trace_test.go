package performance

import (
	"bytes"
	"context"
	"fmt"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/service/image"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

// TestCrashTrace reproduces the "slice bounds out of range [8388190:1]" panic.
//
// Run with GOTRACEBACK=crash to capture the full stack trace:
//
//	set GOTRACEBACK=crash
//	go test -v -run TestCrashTrace -timeout 30m ./internal/service/tests/performance/ 2>crash_trace.txt
//
// The crash_trace.txt file will contain the exact goroutine stack showing
// which line in kronk/yzma produced the panic, even though recover() catches it.
//
// Strategy: maximize KV cache pressure by using large images with a small
// context window and aggressive cache quantization. The error typically
// appears after 1-2 successful descriptions, when the model's internal state
// becomes corrupted under memory pressure.
func TestCrashTrace(t *testing.T) {
	// Force GOTRACEBACK=crash so the runtime prints the full stack before
	// recover() catches the panic. This is the key to getting the exact line.
	os.Setenv("GOTRACEBACK", "crash")

	testsboot.Boot()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	testdataDir := "../testdata"
	images, err := filepath.Glob(filepath.Join(testdataDir, "*.*"))
	if err != nil || len(images) == 0 {
		t.Skip("no test images found")
	}

	P := config.VisionPrompt{
		SystemPrompt: "You extract image keywords for semantic search.",
		UserPrompt:   "Describe this image in detail. Include: objects, people, background, colors, actions, visible text and overall context. Be descriptive and precise.",
		MaxTokens:    300,
		Temperature:  0.1,
	}

	// Configs designed to maximize KV cache pressure.
	// From most aggressive (most likely to crash) to least.
	stressConfigs := []struct {
		name          string
		contextWindow int
		nBatch        int
		nUBatch       int
		cacheK        model.GGMLType
		cacheV        model.GGMLType
		maxSides      []int
	}{
		// Tiny context + Q4_0 cache + large images = maximum pressure
		{"stress-q4-tiny", 768, 512, 512, model.GGMLTypeQ4_0, model.GGMLTypeQ4_0, []int{768}},
		// Small context + Q4_0 cache
		//	{"stress-q4-small", 4096, 1024, 512, model.GGMLTypeQ4_0, model.GGMLTypeQ4_0, []int{512, 384}},
		// Tiny context + Q8_0 cache + large images
		//	{"stress-q8-tiny", 2048, 1024, 512, model.GGMLTypeQ8_0, model.GGMLTypeQ8_0, []int{512, 384, 256}},
		// App defaults but with larger images
		//	{"stress-app", 8192, 2048, 1024, model.GGMLTypeQ8_0, model.GGMLTypeQ8_0, []int{768, 512, 384}},
	}

	for _, sc := range stressConfigs {
		t.Run(sc.name, func(t *testing.T) {
			fmt.Printf("\n%s\n=== Stress config: %s (ctx=%d, Q_K=%d, Q_V=%d) ===\n%s\n",
				strings.Repeat("=", 80), sc.name, sc.contextWindow, sc.cacheK, sc.cacheV, strings.Repeat("=", 80))

			krn, err := kronksdk.New(
				model.WithModelFiles(testsboot.VisionPaths.ModelFiles),
				model.WithProjFile(testsboot.VisionPaths.ProjFile),
				model.WithContextWindow(sc.contextWindow),
				model.WithNBatch(sc.nBatch),
				model.WithNUBatch(sc.nUBatch),
				model.WithCacheTypeK(sc.cacheK),
				model.WithCacheTypeV(sc.cacheV),
			)
			if err != nil {
				t.Logf("config %s failed to load: %v", sc.name, err)
				return
			}

			for _, maxSide := range sc.maxSides {
				// Run ALL images at this size, sequentially, on the same model
				// instance. The error typically appears after a good first run.
				for runNum, imgPath := range images {
					imgName := filepath.Base(imgPath)
					fmt.Printf("\n--- [%s] Run %d: %s maxSide=%d ---\n", sc.name, runNum+1, imgName, maxSide)

					imageData, err := image.Resize(imgPath, maxSide)
					if err != nil {
						fmt.Printf("  resize error: %v\n", err)
						continue
					}

					resized, _ := jpeg.Decode(bytes.NewReader(imageData))
					if resized != nil {
						fmt.Printf("  image: %dx%d (%d bytes)\n", resized.Bounds().Dx(), resized.Bounds().Dy(), len(imageData))
					}

					inferCtx, inferCancel := context.WithTimeout(ctx, 2*time.Minute)

					messages := []model.D{
						{"role": "system", "content": P.SystemPrompt},
						{"role": "user", "content": imageData},
						{"role": "user", "content": P.UserPrompt},
					}

					data := model.D{
						"messages":    messages,
						"temperature": P.Temperature,
						"max_tokens":  P.MaxTokens,
					}

					start := time.Now()
					resp, chatErr := krn.Chat(inferCtx, data)
					elapsed := time.Since(start)
					inferCancel()

					if chatErr != nil {
						fmt.Printf("  CHAT ERROR: %v\n", chatErr)

						// Check if this is the slice bounds panic we're looking for.
						if strings.Contains(chatErr.Error(), "slice bounds out of range") {
							fmt.Printf("\n\n*** FOUND THE BUG ***\n")
							fmt.Printf("  Config: %s | Image: %s | MaxSide: %d\n", sc.name, imgName, maxSide)
							fmt.Printf("  Error: %v\n", chatErr)
							fmt.Printf("  Check crash_trace.txt (stderr) for the full stack trace.\n\n")
							t.Fatalf("Reproduced slice bounds panic: %v", chatErr)
						}
						continue
					}

					// Check FinishReason for errors that didn't come through chatErr.
					if len(resp.Choices) > 0 && resp.Choices[0].FinishReason() == model.FinishReasonError {
						errMsg := ""
						if resp.Choices[0].Delta != nil {
							errMsg = resp.Choices[0].Delta.Content
						} else if resp.Choices[0].Message != nil {
							errMsg = resp.Choices[0].Message.Content
						}
						fmt.Printf("  DECODE ERROR: %s\n", errMsg)

						if strings.Contains(errMsg, "slice bounds out of range") {
							fmt.Printf("\n\n*** FOUND THE BUG (via FinishReason) ***\n")
							fmt.Printf("  Config: %s | Image: %s | MaxSide: %d\n", sc.name, imgName, maxSide)
							fmt.Printf("  Error: %s\n", errMsg)
							fmt.Printf("  Check crash_trace.txt (stderr) for the full stack trace.\n\n")
							t.Fatalf("Reproduced slice bounds panic: %s", errMsg)
						}
						continue
					}

					desc := ""
					if len(resp.Choices) > 0 && resp.Choices[0].Message != nil {
						desc = resp.Choices[0].Message.Content
					}
					fmt.Printf("  OK (%dms, %d tok): %.80s...\n", elapsed.Milliseconds(), resp.Usage.OutputTokens, desc)
				}
			}

			krn.Unload(ctx)
		})
	}
}
