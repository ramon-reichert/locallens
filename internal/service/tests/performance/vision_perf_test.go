package performance

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

	"github.com/ramon-reichert/locallens/internal/service/description"
	"github.com/ramon-reichert/locallens/internal/service/image"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

const (
	DefaultRepetitions = 3
)

var defaultMaxSizes = []int{256, 384, 512}

var defaultConfigs = []ConfigVariant{
	{"small", 1536, 256, 0, 80, 0.2},
	{"medium", 4096, 1024, 1024, 120, 0.2},
	{"large", 8192, 2048, 2048, 120, 0.2},
}

var prompt = description.Prompt

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
	Config    string
	MaxSize   int
	Image     string
	Prompt    string
	ImageInfo ImageInfo
	Runs      []RunResult
	Stats     Stats
}

type Stats struct {
	MeanTime    float64
	StdDevTime  float64
	MinTime     time.Duration
	MaxTime     time.Duration
	MeanInTok   float64
	MeanOutTok  float64
	MeanTPS     float64
	MeanOutLen  float64
	SuccessRate float64
	VarianceCV  float64
}

type BenchmarkInfo struct {
	ModelFile  string
	ProjFile   string
	CacheTypeK string
	CacheTypeV string
	Prompt     string
	Configs    []ConfigVariant
	MaxSizes   []int
}

func TestVisionPerformance(t *testing.T) {
	testsboot.Boot()

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Minute)
	defer cancel()

	repetitions := DefaultRepetitions

	testdataDir := "../testdata"
	images, err := filepath.Glob(filepath.Join(testdataDir, "*.*"))
	if err != nil || len(images) == 0 {
		t.Skip("no test images found")
	}

	info := BenchmarkInfo{
		ModelFile:  filepath.Base(testsboot.VisionPaths.ModelFiles[0]),
		ProjFile:   filepath.Base(testsboot.VisionPaths.ProjFile),
		CacheTypeK: "Q8_0",
		CacheTypeV: "Q8_0",
		Prompt:     prompt,
		Configs:    defaultConfigs,
		MaxSizes:   defaultMaxSizes,
	}

	var results []AggregatedResult

	for _, cfg := range defaultConfigs {
		fmt.Print(strings.Repeat("=", 100))
		fmt.Printf("\n\n    === Config: %s (ctx=%d, Nbatch=%d, NUbatch=%d, maxTok=%d, temp=%v) | %d repetitions ===\n\n",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, cfg.MaxTokens, cfg.Temperature, repetitions)

		krn, err := kronksdk.New(model.Config{
			ModelFiles:    testsboot.VisionPaths.ModelFiles,
			ProjFile:      testsboot.VisionPaths.ProjFile,
			ContextWindow: cfg.ContextWindow,
			NBatch:        cfg.NBatch,
			NUBatch:       cfg.NUBatch,
			CacheTypeK:    model.GGMLTypeQ8_0,
			CacheTypeV:    model.GGMLTypeQ8_0,
		})
		if err != nil {
			fmt.Printf("config %s failed to load: %v", cfg.Name, err)
			continue
		}

		for _, maxSize := range defaultMaxSizes {

			for _, imgPath := range images {
				aggResult := runWithRepetitions(ctx, krn, imgPath, prompt, cfg, maxSize, repetitions)
				results = append(results, aggResult)

				fmt.Printf("     >>>> maxSize: %d === %s: avgTime %.0fms | timeVar %.0f%% | %d/%d success\n\n\n",
					maxSize,
					filepath.Base(imgPath),
					aggResult.Stats.MeanTime,
					aggResult.Stats.VarianceCV*100,
					int(aggResult.Stats.SuccessRate*float64(repetitions)),
					repetitions)
			}
		}

		krn.Unload(ctx)
	}

	printConfigs(info)
	printConfigSummary(results)
	printGroupedResults(results)
	saveCSV(info, results)
}

func runWithRepetitions(ctx context.Context, krn *kronksdk.Kronk, imgPath, prompt string, cfg ConfigVariant, maxSize, reps int) AggregatedResult {
	result := AggregatedResult{
		Config:  cfg.Name,
		MaxSize: maxSize,
		Image:   filepath.Base(imgPath),
		Prompt:  prompt,
		Runs:    make([]RunResult, 0, reps),
	}

	// Get image info once
	origInfo, err := getImageInfo(imgPath)
	if err != nil {
		result.Runs = append(result.Runs, RunResult{Run: 1, Error: fmt.Sprintf("get info: %v", err)})
		return result
	}

	imageData, err := image.Resize(imgPath, maxSize)
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

	fmt.Printf("=> Run %d: %d ms | inTok=%d outTok=%d | Tok/s=%.1f",
		run.Run, run.Elapsed.Milliseconds(),
		run.Metrics.PromptTokens, run.Metrics.CompletionTokens,
		run.Metrics.TokensPerSecond)
	fmt.Printf("| Description: %s\n\n", run.Description)

	return run
}

func calculateStats(runs []RunResult) Stats {
	var stats Stats
	var times []float64
	var inTok, outTok, tps, outLen []float64
	successCount := 0

	for _, r := range runs {
		if r.Error == "" {
			successCount++
			times = append(times, float64(r.Elapsed.Milliseconds()))
			inTok = append(inTok, float64(r.Metrics.PromptTokens))
			outTok = append(outTok, float64(r.Metrics.CompletionTokens))
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
	stats.MeanInTok = mean(inTok)
	stats.MeanOutTok = mean(outTok)
	stats.MeanTPS = mean(tps)
	stats.MeanOutLen = mean(outLen)

	if stats.MeanTime > 0 {
		stats.VarianceCV = stats.StdDevTime / stats.MeanTime
	}

	return stats
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

func printConfigs(info BenchmarkInfo) {
	fmt.Print("\n" + strings.Repeat("=", 100))
	fmt.Print("CONFIGS")
	fmt.Print(strings.Repeat("=", 100))

	fmt.Printf("Model:      %s", info.ModelFile)
	fmt.Printf("MMProj:     %s", info.ProjFile)
	fmt.Printf("CacheTypeK: %s", info.CacheTypeK)
	fmt.Printf("CacheTypeV: %s", info.CacheTypeV)
	fmt.Printf("MaxSizes:   %v", info.MaxSizes)
	fmt.Print("")
	fmt.Printf("Prompt: %s", strings.ReplaceAll(info.Prompt, "\n", " "))
	fmt.Print("")

	fmt.Printf("%-8s | %6s | %6s | %7s | %6s | %5s",
		"Name", "CtxWin", "NBatch", "NUBatch", "MaxTok", "Temp")
	fmt.Print(strings.Repeat("-", 60))
	for _, cfg := range info.Configs {
		fmt.Printf("%-8s | %6d | %6d | %7d | %6d | %.1f",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, cfg.MaxTokens, cfg.Temperature)
	}
}

func printGroupedResults(results []AggregatedResult) {
	fmt.Print("\n" + strings.Repeat("=", 100))
	fmt.Print("GROUPED RESULTS")
	fmt.Print(strings.Repeat("=", 100))
	fmt.Printf("%-8s | %4s | %-15s | %10s | %5s | %6s | %6s | %5s | %5s",
		"Config", "Max", "Image", "AvgTime(ms)", "TimeVar%", "InTok", "OutTok", "Tok/s", "Succ")
	fmt.Print(strings.Repeat("-", 100))

	for _, r := range results {
		successPct := fmt.Sprintf("%.0f%%", r.Stats.SuccessRate*100)
		cvPct := fmt.Sprintf("%.0f%%", r.Stats.VarianceCV*100)

		fmt.Printf("%-8s | %4d | %-15s | %10.0f | %5s | %6.0f | %6.0f | %5.1f | %5s",
			r.Config, r.MaxSize, r.Image,
			r.Stats.MeanTime, cvPct, r.Stats.MeanInTok, r.Stats.MeanOutTok,
			r.Stats.MeanTPS, successPct)
	}
}

func printConfigSummary(results []AggregatedResult) {
	fmt.Print("\n" + strings.Repeat("=", 100))
	fmt.Print("SUMMARY BY CONFIG + MAXSIZE")
	fmt.Print(strings.Repeat("=", 100))

	type key struct {
		config  string
		maxSize int
	}

	configStats := make(map[key]struct {
		times  []float64
		inTok  []float64
		outTok []float64
		tps    []float64
		cvSum  float64
		count  int
	})

	for _, r := range results {
		k := key{r.Config, r.MaxSize}
		stats := configStats[k]
		stats.count++
		if r.Stats.SuccessRate > 0 {
			stats.times = append(stats.times, r.Stats.MeanTime)
			stats.inTok = append(stats.inTok, r.Stats.MeanInTok)
			stats.outTok = append(stats.outTok, r.Stats.MeanOutTok)
			stats.tps = append(stats.tps, r.Stats.MeanTPS)
			stats.cvSum += r.Stats.VarianceCV
		}
		configStats[k] = stats
	}

	var keys []key
	for k := range configStats {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].config != keys[j].config {
			return keys[i].config < keys[j].config
		}
		return keys[i].maxSize < keys[j].maxSize
	})

	for _, k := range keys {
		stats := configStats[k]
		avgCV := 0.0
		if stats.count > 0 {
			avgCV = stats.cvSum / float64(stats.count) * 100
		}
		fmt.Printf("%-8s @%3d: avgTime %.0fms | avgTimeVar %.0f%% | inTok %.0f | outTok %.0f | Tok/s %.1f",
			k.config, k.maxSize,
			mean(stats.times),
			avgCV,
			mean(stats.inTok),
			mean(stats.outTok),
			mean(stats.tps))
	}

	fmt.Print(strings.Repeat("=", 100))
}

func saveCSV(info BenchmarkInfo, results []AggregatedResult) {
	timestamp := time.Now().Format("20060102_150405")

	prompt := strings.ReplaceAll(info.Prompt, "\"", "'")
	prompt = strings.ReplaceAll(prompt, "\n", " ")

	// Save grouped results
	groupedFile := fmt.Sprintf("benchmark_grouped_%s.csv", timestamp)
	var sb strings.Builder
	sb.WriteString("model,mmproj,cache_k,cache_v,prompt,")
	sb.WriteString("config,ctx_win,nbatch,nubatch,max_tok,temp,")
	sb.WriteString("max_size,image,orig_w,orig_h,resized_w,resized_h,bytes,")
	sb.WriteString("mean_ms,cv_pct,in_tok,out_tok,tps,success_rate\n")

	for _, r := range results {
		var cfg ConfigVariant
		for _, c := range info.Configs {
			if c.Name == r.Config {
				cfg = c
				break
			}
		}

		sb.WriteString(fmt.Sprintf("\"%s\",\"%s\",%s,%s,\"%s\",",
			info.ModelFile, info.ProjFile, info.CacheTypeK, info.CacheTypeV, prompt))
		sb.WriteString(fmt.Sprintf("%s,%d,%d,%d,%d,%.1f,",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, cfg.MaxTokens, cfg.Temperature))
		sb.WriteString(fmt.Sprintf("%d,%s,%d,%d,%d,%d,%d,",
			r.MaxSize, r.Image,
			r.ImageInfo.OriginalW, r.ImageInfo.OriginalH,
			r.ImageInfo.ResizedW, r.ImageInfo.ResizedH, r.ImageInfo.Bytes))
		sb.WriteString(fmt.Sprintf("%.0f,%.0f,%.0f,%.0f,%.1f,%.2f\n",
			r.Stats.MeanTime, r.Stats.VarianceCV*100,
			r.Stats.MeanInTok, r.Stats.MeanOutTok, r.Stats.MeanTPS, r.Stats.SuccessRate))
	}

	if err := os.WriteFile(groupedFile, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("failed to save grouped CSV: %v", err)
	} else {
		fmt.Printf("\nGrouped results saved to: %s", groupedFile)
	}

	// Save individual results
	individualFile := fmt.Sprintf("benchmark_individual_%s.csv", timestamp)
	sb.Reset()
	sb.WriteString("config,max_size,image,run,elapsed_ms,in_tok,out_tok,tps,error,description\n")

	for _, r := range results {
		for _, run := range r.Runs {
			desc := strings.ReplaceAll(run.Description, "\"", "'")
			desc = strings.ReplaceAll(desc, "\n", " ")
			errStr := strings.ReplaceAll(run.Error, "\"", "'")

			sb.WriteString(fmt.Sprintf("%s,%d,%s,%d,%d,%d,%d,%.1f,\"%s\",\"%s\"\n",
				r.Config, r.MaxSize, r.Image, run.Run,
				run.Elapsed.Milliseconds(),
				run.Metrics.PromptTokens, run.Metrics.CompletionTokens,
				run.Metrics.TokensPerSecond,
				errStr, desc))
		}
	}

	if err := os.WriteFile(individualFile, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("failed to save individual CSV: %v", err)
	} else {
		fmt.Printf("Individual results saved to: %s", individualFile)
	}
}
