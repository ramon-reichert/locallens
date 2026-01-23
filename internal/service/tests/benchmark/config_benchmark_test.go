package benchmark

import (
	"bytes"
	"context"
	"fmt"
	stdimage "image"
	_ "image/gif"
	"image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/webp"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"

	"github.com/ramon-reichert/locallens/internal/platform/kronk"
	"github.com/ramon-reichert/locallens/internal/platform/logger"
	"github.com/ramon-reichert/locallens/internal/service/image"
)

const (
	DefaultRepetitions    = 3
	HighVarianceThreshold = 0.3 // 30% coefficient of variation
)

type ConfigVariant struct {
	Name          string
	ContextWindow int
	NBatch        int
	NUBatch       int
	MaxTokens     int
	Temperature   float64
}

type ImageInfo struct {
	OriginalW int
	OriginalH int
	ResizedW  int
	ResizedH  int
	Bytes     int
}

type ModelMetrics struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	TokensPerSecond  float64
}

type RunResult struct {
	Run         int
	Elapsed     time.Duration
	OutputLen   int
	Description string
	Metrics     ModelMetrics
	Error       string
}

type AggregatedResult struct {
	Config       string
	Image        string
	Prompt       string
	ImageInfo    ImageInfo
	Runs         []RunResult
	Stats        Stats
	HighVariance bool
}

type Stats struct {
	MeanTime    float64
	StdDevTime  float64
	MinTime     time.Duration
	MaxTime     time.Duration
	MeanTokens  float64
	MeanTPS     float64
	MeanOutLen  float64
	SuccessRate float64
}

var defaultConfigs = []ConfigVariant{
	{"small", 1536, 256, 0, 80, 0.2},
	{"medium", 4096, 1024, 1024, 120, 0.2},
	{"large", 8192, 2048, 2048, 120, 0.2},
}

var defaultPrompt = `Describe the image for semantic search.
List visible objects, counts, attributes, actions, colors, and setting.
Use short phrases. Output a single comma-separated list.`

func TestConfigBenchmark(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Minute)
	defer cancel()

	log := logger.New()
	repetitions := DefaultRepetitions

	if err := kronk.InstallDependencies(ctx, log); err != nil {
		t.Fatalf("install: %v", err)
	}

	paths, err := kronk.DownloadModels(ctx, log)
	if err != nil {
		t.Fatalf("download: %v", err)
	}

	if err := kronksdk.Init(); err != nil {
		t.Fatalf("init: %v", err)
	}

	testdataDir := "../testdata"
	images, err := filepath.Glob(filepath.Join(testdataDir, "*.*"))
	if err != nil || len(images) == 0 {
		t.Skip("no test images found")
	}

	var results []AggregatedResult

	for _, cfg := range defaultConfigs {
		t.Logf("\n=== Config: %s (ctx=%d, batch=%d, ubatch=%d) | %d repetitions ===",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, repetitions)

		krn, err := kronksdk.New(model.Config{
			ModelFiles:    paths.Vision.ModelFiles,
			ProjFile:      paths.Vision.ProjFile,
			ContextWindow: cfg.ContextWindow,
			NBatch:        cfg.NBatch,
			NUBatch:       cfg.NUBatch,
			CacheTypeK:    model.GGMLTypeQ8_0,
			CacheTypeV:    model.GGMLTypeQ8_0,
		})
		if err != nil {
			t.Logf("config %s failed to load: %v", cfg.Name, err)
			continue
		}

		for _, imgPath := range images {
			aggResult := runWithRepetitions(ctx, krn, imgPath, defaultPrompt, cfg, repetitions)
			results = append(results, aggResult)

			status := "OK"
			if aggResult.HighVariance {
				status = "HIGH VARIANCE"
			}

			t.Logf("  %s: mean %.0fms (±%.0f) | %d/%d success | %s",
				filepath.Base(imgPath),
				aggResult.Stats.MeanTime,
				aggResult.Stats.StdDevTime,
				int(aggResult.Stats.SuccessRate*float64(repetitions)),
				repetitions,
				status)
		}

		krn.Unload(ctx)
	}

	printAggregatedResults(t, results)
	printConfigSummary(t, results)
	saveAggregatedCSV(t, results)
}

func runWithRepetitions(ctx context.Context, krn *kronksdk.Kronk, imgPath, prompt string, cfg ConfigVariant, reps int) AggregatedResult {
	result := AggregatedResult{
		Config: cfg.Name,
		Image:  filepath.Base(imgPath),
		Prompt: prompt,
		Runs:   make([]RunResult, 0, reps),
	}

	// Get image info once
	origInfo, err := getImageInfo(imgPath)
	if err != nil {
		result.Runs = append(result.Runs, RunResult{Run: 1, Error: fmt.Sprintf("get info: %v", err)})
		return result
	}

	imageData, err := image.Resize(imgPath, image.DefaultMaxSide)
	if err != nil {
		result.Runs = append(result.Runs, RunResult{Run: 1, Error: fmt.Sprintf("resize: %v", err)})
		return result
	}

	resizedImg, err := jpeg.Decode(bytes.NewReader(imageData))
	if err != nil {
		result.Runs = append(result.Runs, RunResult{Run: 1, Error: fmt.Sprintf("decode: %v", err)})
		return result
	}

	result.ImageInfo = ImageInfo{
		OriginalW: origInfo.W,
		OriginalH: origInfo.H,
		ResizedW:  resizedImg.Bounds().Dx(),
		ResizedH:  resizedImg.Bounds().Dy(),
		Bytes:     len(imageData),
	}

	// Run multiple times
	for i := 0; i < reps; i++ {
		run := runSingleInference(ctx, krn, imageData, prompt, cfg, i+1)
		result.Runs = append(result.Runs, run)
	}

	// Calculate stats
	result.Stats = calculateStats(result.Runs)
	result.HighVariance = isHighVariance(result.Stats)

	return result
}

func runSingleInference(ctx context.Context, krn *kronksdk.Kronk, imageData []byte, prompt string, cfg ConfigVariant, runNum int) RunResult {
	run := RunResult{Run: runNum}

	data := model.D{
		"messages":    model.RawMediaMessage(prompt, imageData),
		"temperature": cfg.Temperature,
		"max_tokens":  cfg.MaxTokens,
	}

	start := time.Now()
	resp, err := krn.Chat(ctx, data)
	run.Elapsed = time.Since(start)

	if err != nil {
		run.Error = fmt.Sprintf("chat: %v", err)
		return run
	}

	run.Description = resp.Choice[0].Message.Content
	run.OutputLen = len(run.Description)
	run.Metrics = ModelMetrics{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
		TokensPerSecond:  resp.Usage.TokensPerSecond,
	}

	return run
}

func calculateStats(runs []RunResult) Stats {
	var stats Stats
	var times []float64
	var tokens, tps, outLen []float64
	successCount := 0

	for _, r := range runs {
		if r.Error == "" {
			successCount++
			times = append(times, float64(r.Elapsed.Milliseconds()))
			tokens = append(tokens, float64(r.Metrics.TotalTokens))
			tps = append(tps, r.Metrics.TokensPerSecond)
			outLen = append(outLen, float64(r.OutputLen))
		}
	}

	stats.SuccessRate = float64(successCount) / float64(len(runs))

	if len(times) == 0 {
		return stats
	}

	stats.MeanTime = mean(times)
	stats.StdDevTime = stddev(times)
	stats.MinTime = time.Duration(min(times)) * time.Millisecond
	stats.MaxTime = time.Duration(max(times)) * time.Millisecond
	stats.MeanTokens = mean(tokens)
	stats.MeanTPS = mean(tps)
	stats.MeanOutLen = mean(outLen)

	return stats
}

func isHighVariance(stats Stats) bool {
	if stats.MeanTime == 0 {
		return false
	}
	cv := stats.StdDevTime / stats.MeanTime
	return cv > HighVarianceThreshold
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func stddev(vals []float64) float64 {
	if len(vals) < 2 {
		return 0
	}
	m := mean(vals)
	sum := 0.0
	for _, v := range vals {
		sum += (v - m) * (v - m)
	}
	return math.Sqrt(sum / float64(len(vals)-1))
}

func min(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := vals[0]
	for _, v := range vals[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func max(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

type imgInfo struct {
	W, H int
}

func getImageInfo(path string) (imgInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return imgInfo{}, err
	}
	defer file.Close()

	img, _, err := stdimage.DecodeConfig(file)
	if err != nil {
		return imgInfo{}, err
	}

	return imgInfo{W: img.Width, H: img.Height}, nil
}

func printAggregatedResults(t *testing.T, results []AggregatedResult) {
	t.Log("\n" + strings.Repeat("=", 150))
	t.Log("DETAILED RESULTS")
	t.Log(strings.Repeat("=", 150))
	t.Logf("%-8s | %-20s | %-10s | %10s | %10s | %10s | %6s | %6s | %5s | %s",
		"Config", "Image", "Size", "Mean(ms)", "StdDev", "Min-Max", "Tokens", "TPS", "Succ", "Variance")
	t.Log(strings.Repeat("-", 150))

	for _, r := range results {
		size := fmt.Sprintf("%dx%d", r.ImageInfo.ResizedW, r.ImageInfo.ResizedH)
		minMax := fmt.Sprintf("%d-%d", r.Stats.MinTime.Milliseconds(), r.Stats.MaxTime.Milliseconds())
		successPct := fmt.Sprintf("%.0f%%", r.Stats.SuccessRate*100)

		variance := "OK"
		if r.HighVariance {
			variance = "HIGH!"
		}

		t.Logf("%-8s | %-20s | %-10s | %10.0f | %10.0f | %10s | %6.0f | %6.1f | %5s | %s",
			r.Config, r.Image, size,
			r.Stats.MeanTime, r.Stats.StdDevTime, minMax,
			r.Stats.MeanTokens, r.Stats.MeanTPS, successPct, variance)
	}
}

func printConfigSummary(t *testing.T, results []AggregatedResult) {
	t.Log("\n" + strings.Repeat("=", 100))
	t.Log("SUMMARY BY CONFIG")
	t.Log(strings.Repeat("=", 100))

	configStats := make(map[string]struct {
		times        []float64
		tokens       []float64
		tps          []float64
		highVarCount int
		totalImages  int
	})

	for _, r := range results {
		stats := configStats[r.Config]
		stats.totalImages++
		if r.Stats.SuccessRate > 0 {
			stats.times = append(stats.times, r.Stats.MeanTime)
			stats.tokens = append(stats.tokens, r.Stats.MeanTokens)
			stats.tps = append(stats.tps, r.Stats.MeanTPS)
		}
		if r.HighVariance {
			stats.highVarCount++
		}
		configStats[r.Config] = stats
	}

	// Sort by config name
	var configNames []string
	for name := range configStats {
		configNames = append(configNames, name)
	}
	sort.Strings(configNames)

	for _, name := range configNames {
		stats := configStats[name]
		t.Logf("%-8s: avg time %.0fms | avg tokens %.0f | avg TPS %.1f | high variance: %d/%d images",
			name,
			mean(stats.times),
			mean(stats.tokens),
			mean(stats.tps),
			stats.highVarCount, stats.totalImages)
	}

	t.Log(strings.Repeat("=", 100))
}

func saveAggregatedCSV(t *testing.T, results []AggregatedResult) {
	filename := fmt.Sprintf("benchmark_results_%s.csv", time.Now().Format("20060102_150405"))

	var sb strings.Builder
	sb.WriteString("config,image,orig_w,orig_h,resized_w,resized_h,bytes,")
	sb.WriteString("mean_ms,stddev_ms,min_ms,max_ms,")
	sb.WriteString("mean_tokens,mean_tps,mean_outlen,")
	sb.WriteString("success_rate,high_variance,")
	sb.WriteString("prompt,sample_description\n")

	for _, r := range results {
		sampleDesc := ""
		for _, run := range r.Runs {
			if run.Error == "" && run.Description != "" {
				sampleDesc = run.Description
				break
			}
		}
		sampleDesc = strings.ReplaceAll(sampleDesc, "\"", "'")
		sampleDesc = strings.ReplaceAll(sampleDesc, "\n", " ")
		prompt := strings.ReplaceAll(r.Prompt, "\"", "'")
		prompt = strings.ReplaceAll(prompt, "\n", " ")

		sb.WriteString(fmt.Sprintf("%s,%s,%d,%d,%d,%d,%d,",
			r.Config, r.Image,
			r.ImageInfo.OriginalW, r.ImageInfo.OriginalH,
			r.ImageInfo.ResizedW, r.ImageInfo.ResizedH, r.ImageInfo.Bytes))
		sb.WriteString(fmt.Sprintf("%.0f,%.0f,%d,%d,",
			r.Stats.MeanTime, r.Stats.StdDevTime,
			r.Stats.MinTime.Milliseconds(), r.Stats.MaxTime.Milliseconds()))
		sb.WriteString(fmt.Sprintf("%.0f,%.1f,%.0f,",
			r.Stats.MeanTokens, r.Stats.MeanTPS, r.Stats.MeanOutLen))
		sb.WriteString(fmt.Sprintf("%.2f,%t,", r.Stats.SuccessRate, r.HighVariance))
		sb.WriteString(fmt.Sprintf("\"%s\",\"%s\"\n", prompt, sampleDesc))
	}

	if err := os.WriteFile(filename, []byte(sb.String()), 0644); err != nil {
		t.Logf("failed to save CSV: %v", err)
		return
	}

	t.Logf("\nResults saved to: %s", filename)
}
