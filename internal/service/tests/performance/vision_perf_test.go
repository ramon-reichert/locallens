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
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/webp"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"

	"github.com/ramon-reichert/locallens/internal/platform/sysmon"
	"github.com/ramon-reichert/locallens/internal/service/description"
	"github.com/ramon-reichert/locallens/internal/service/image"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

var decodeErrorRegex = regexp.MustCompile(`(?:non-zero|code):\s*(-?\d+)`)

const (
	DefaultRepetitions = 2

	// Pressure detection thresholds
	ThresholdMsPerTokenMultiplier = 2.0 // Flag if MsPerToken > baseline * this
	ThresholdPageFaults           = 100 // Flag if page faults delta > this
	ThresholdLowRAM_MB            = 500 // Flag if available RAM < this (MB)
	ThresholdMinOutputLen         = 20  // Flag if output length < this
)

var defaultMaxSizes = []int{384} //64, 128, 256, 384, 512

var defaultConfigs = []ConfigVariant{
	{"small", 1024, 8, 8, model.GGMLTypeQ8_0, model.GGMLTypeQ8_0},
}

var P = description.P

type ConfigVariant struct {
	Name          string
	ContextWindow int
	NBatch        int
	NUBatch       int
	CacheTypeK    model.GGMLType
	CacheTypeV    model.GGMLType
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
	MsPerToken       float64
}

type SystemMetrics struct {
	AvailableRAM_MB uint64
	PageFaultsDelta uint64
}

type PressureFlags struct {
	SlowToken       bool
	HighFaults      bool
	LowRAM          bool
	Truncated       bool
	DecodeErrorCode int // 0=none, 1=KV cache full, 2=aborted, -1=invalid batch, <-1=fatal
}

func (f PressureFlags) Any() bool {
	return f.SlowToken || f.HighFaults || f.LowRAM || f.Truncated || f.DecodeErrorCode != 0
}

// TODO: Return defined error types in sdk\kronk\model\model.go:635 to prevent parsing the error string
func (f PressureFlags) DecodeErrorString() string {
	switch f.DecodeErrorCode {
	case 0:
		return ""
	case 1:
		return "[DECODE:1-KV_CACHE_FULL]"
	case 2:
		return "[DECODE:2-ABORTED]"
	case -1:
		return "[DECODE:-1-INVALID_BATCH]"
	default:
		return fmt.Sprintf("[DECODE:%d-FATAL]", f.DecodeErrorCode)
	}
}

func extractDecodeErrorCode(errMsg string) int {
	matches := decodeErrorRegex.FindStringSubmatch(errMsg)
	if len(matches) < 2 {
		return 0
	}
	code, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0
	}
	return code
}

type RunResult struct {
	Run         int
	Elapsed     time.Duration
	OutputLen   int
	Description string
	Metrics     ModelMetrics
	System      SystemMetrics
	Flags       PressureFlags
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

	// Memory pressure stats
	MeanMsPerToken   float64
	MaxPageFaults    uint64
	MinAvailableRAM  uint64
	PressureRunCount int
}

type BenchmarkInfo struct {
	ModelFile string
	ProjFile  string
	Prompt    string
	Configs   []ConfigVariant
	MaxSizes  []int
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

	fullPrompt := P.SystemPrompt + "\n\n" + P.UserPrompt + "\n"

	info := BenchmarkInfo{
		ModelFile: filepath.Base(testsboot.VisionPaths.ModelFiles[0]),
		ProjFile:  filepath.Base(testsboot.VisionPaths.ProjFile),
		Prompt:    fullPrompt,
		Configs:   defaultConfigs,
		MaxSizes:  defaultMaxSizes,
	}

	var results []AggregatedResult

	for _, cfg := range defaultConfigs {
		fmt.Print(strings.Repeat("=", 100))
		fmt.Printf("\n\n    === Config: %s (ctx=%d, Nbatch=%d, NUbatch=%d) | maxTok=%d, temp=%.1f | %d reps ===\n\n",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, P.MaxTokens, P.Temperature, repetitions)

		krn, err := kronksdk.New(model.Config{
			ModelFiles:    testsboot.VisionPaths.ModelFiles,
			ProjFile:      testsboot.VisionPaths.ProjFile,
			ContextWindow: cfg.ContextWindow,
			NBatch:        cfg.NBatch,
			NUBatch:       cfg.NUBatch,
			CacheTypeK:    cfg.CacheTypeK,
			CacheTypeV:    cfg.CacheTypeV,
		})
		if err != nil {
			fmt.Printf("config %s failed to load: %v\n", cfg.Name, err)
			continue
		}

		for _, maxSize := range defaultMaxSizes {

			for _, imgPath := range images {
				aggResult := runWithRepetitions(ctx, krn, imgPath, cfg, maxSize, repetitions)
				results = append(results, aggResult)

				fmt.Printf("     >>>> %s: maxSize=%d config=%s || avgTime=%.0fms | timeVar=%.0f%% | %d/%d success\n\n\n",
					filepath.Base(imgPath),
					maxSize,
					cfg.Name,
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
	printPressureSummary(results)
	saveCSV(info, results)
}

func runWithRepetitions(ctx context.Context, krn *kronksdk.Kronk, imgPath string, cfg ConfigVariant, maxSize, reps int) AggregatedResult {
	fullPrompt := P.SystemPrompt + "\n\n" + P.UserPrompt + "\n"

	result := AggregatedResult{
		Config:  cfg.Name,
		MaxSize: maxSize,
		Image:   filepath.Base(imgPath),
		Prompt:  fullPrompt,
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
	// TODO: This is bad relationship! Should find a normal ratio between Tok/s and ms/tok. Not use a baseline!
	// First run establishes baseline MsPerToken (use 0 for first run)
	var baselineMsPerToken float64
	for i := 0; i < reps; i++ {
		run := runSingleInference(ctx, krn, imageData, i+1, baselineMsPerToken)

		// Set baseline from first successful run
		if i == 0 && run.Error == "" && run.Metrics.MsPerToken > 0 {
			baselineMsPerToken = run.Metrics.MsPerToken
		}

		fmt.Printf("=> Run %d: %s | maxSize=%d | Config: %s (ctx=%d, Nbatch=%d, NUbatch=%d) | maxTok=%d, temp=%.1f",
			run.Run,
			filepath.Base(imgPath),
			maxSize,
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, P.MaxTokens, P.Temperature)

		fmt.Printf("\n   Results: time=%d ms | inTok=%d outTok=%d | Tok/s=%.1f | ms/tok=%.1f",
			run.Elapsed.Milliseconds(),
			run.Metrics.PromptTokens,
			run.Metrics.CompletionTokens,
			run.Metrics.TokensPerSecond,
			run.Metrics.MsPerToken)

		// Print system metrics
		fmt.Printf("\n   Memory: avlbRAM=%dMB pgFaults=%d",
			run.System.AvailableRAM_MB,
			run.System.PageFaultsDelta)

		// Print pressure flags if any
		if run.Flags.Any() {
			fmt.Print(" | FLAGS:")
			if run.Flags.SlowToken {
				fmt.Print(" [SlowToks]")
			}
			if run.Flags.HighFaults {
				fmt.Print(" [HiFaults]")
			}
			if run.Flags.LowRAM {
				fmt.Print(" [LowRAM]")
			}
			if run.Flags.Truncated {
				fmt.Print(" [TructOut]")
			}
			if run.Flags.DecodeErrorCode != 0 {
				fmt.Print(" " + run.Flags.DecodeErrorString())
			}
		}

		fmt.Printf("\n   Description: %s\n\n", run.Description)
		if run.Error != "" {
			fmt.Printf("   Error: %s\n", run.Error)
		}
		fmt.Println()

		result.Runs = append(result.Runs, run)

		// Pause between runs to fresh CPU
		if i < reps-1 {
			time.Sleep(5 * time.Second)
		}
	}

	// Calculate stats
	result.Stats = calculateStats(result.Runs)

	return result
}

func runSingleInference(ctx context.Context, krn *kronksdk.Kronk, imageData []byte, runNum int, baselineMsPerToken float64) RunResult {
	run := RunResult{Run: runNum}

	// Capture system state before inference
	snapBefore := sysmon.Capture()
	run.System.AvailableRAM_MB = snapBefore.AvailableRAM_MB

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
	resp, err := krn.Chat(ctx, data)
	run.Elapsed = time.Since(start)

	// Capture system state after inference
	snapAfter := sysmon.Capture()
	if snapAfter.PageFaults >= snapBefore.PageFaults {
		run.System.PageFaultsDelta = snapAfter.PageFaults - snapBefore.PageFaults
	}

	if err != nil {
		run.Error = fmt.Sprintf("chat: %v", err)
		run.Flags.DecodeErrorCode = extractDecodeErrorCode(err.Error())
		return run
	}

	run.Description = resp.Choice[0].Message.Content

	// TODO: Shoud be measured by tokens, no characters.
	run.OutputLen = len(run.Description)

	// Calculate MsPerToken
	msPerToken := 0.0
	if resp.Usage.CompletionTokens > 0 {
		msPerToken = float64(run.Elapsed.Milliseconds()) / float64(resp.Usage.CompletionTokens)
	}

	run.Metrics = ModelMetrics{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
		TokensPerSecond:  resp.Usage.TokensPerSecond,
		MsPerToken:       msPerToken,
	}

	// Set pressure flags
	if baselineMsPerToken > 0 && msPerToken > baselineMsPerToken*ThresholdMsPerTokenMultiplier {
		run.Flags.SlowToken = true
	}
	if run.System.PageFaultsDelta > ThresholdPageFaults {
		run.Flags.HighFaults = true
	}
	if run.System.AvailableRAM_MB > 0 && run.System.AvailableRAM_MB < ThresholdLowRAM_MB {
		run.Flags.LowRAM = true
	}
	if run.OutputLen < ThresholdMinOutputLen {
		run.Flags.Truncated = true
	}

	return run
}

func calculateStats(runs []RunResult) Stats {
	var stats Stats
	var times []float64
	var inTok, outTok, tps, outLen []float64
	var msPerToken []float64
	successCount := 0

	// Initialize min RAM to max uint64 for proper min tracking
	stats.MinAvailableRAM = ^uint64(0)

	for _, r := range runs {
		// Track pressure metrics for all runs (including errors)
		if r.System.PageFaultsDelta > stats.MaxPageFaults {
			stats.MaxPageFaults = r.System.PageFaultsDelta
		}
		if r.System.AvailableRAM_MB > 0 && r.System.AvailableRAM_MB < stats.MinAvailableRAM {
			stats.MinAvailableRAM = r.System.AvailableRAM_MB
		}
		if r.Flags.Any() {
			stats.PressureRunCount++
		}

		if r.Error == "" {
			successCount++
			times = append(times, float64(r.Elapsed.Milliseconds()))
			inTok = append(inTok, float64(r.Metrics.PromptTokens))
			outTok = append(outTok, float64(r.Metrics.CompletionTokens))
			tps = append(tps, r.Metrics.TokensPerSecond)
			outLen = append(outLen, float64(r.OutputLen))
			msPerToken = append(msPerToken, r.Metrics.MsPerToken)
		}
	}

	// Reset MinAvailableRAM if no measurements taken
	if stats.MinAvailableRAM == ^uint64(0) {
		stats.MinAvailableRAM = 0
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
	stats.MeanMsPerToken = mean(msPerToken)

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
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("CONFIGS")
	fmt.Println(strings.Repeat("=", 100))

	fmt.Printf("Model:       %s\n", info.ModelFile)
	fmt.Printf("MMProj:      %s\n", info.ProjFile)
	fmt.Printf("MaxSizes:    %v\n", info.MaxSizes)
	fmt.Printf("MaxTokens:   %d\n", P.MaxTokens)
	fmt.Printf("Temperature: %.1f\n", P.Temperature)
	fmt.Println()
	fmt.Printf("Prompt: %s\n", info.Prompt)
	fmt.Println()

	fmt.Printf("%-10s | %6s | %6s | %7s | %8s | %8s\n",
		"Name", "CtxWin", "NBatch", "NUBatch", "CacheK", "CacheV")
	fmt.Println(strings.Repeat("-", 70))
	for _, cfg := range info.Configs {
		fmt.Printf("%-10s | %6d | %6d | %7d | %8s | %8s\n",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch,
			cacheTypeName(cfg.CacheTypeK), cacheTypeName(cfg.CacheTypeV))
	}
}

func cacheTypeName(t model.GGMLType) string {
	switch t {
	case model.GGMLTypeF16:
		return "F16"
	case model.GGMLTypeQ8_0:
		return "Q8_0"
	case model.GGMLTypeQ4_0:
		return "Q4_0"
	case model.GGMLTypeQ4_1:
		return "Q4_1"
	case model.GGMLTypeBF16:
		return "BF16"
	default:
		return fmt.Sprintf("%d", t)
	}
}

func printGroupedResults(results []AggregatedResult) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("GROUPED RESULTS")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Printf("%-8s | %4s | %-15s | %11s | %8s | %6s | %6s | %5s | %5s | %8s\n",
		"Config", "Max", "Image", "AvgTime(ms)", "TimeVar%", "InTok", "OutTok", "Tok/s", "Succ", "Pressure")
	fmt.Println(strings.Repeat("-", 110))

	for _, r := range results {
		successPct := fmt.Sprintf("%.0f%%", r.Stats.SuccessRate*100)
		cvPct := fmt.Sprintf("%.0f%%", r.Stats.VarianceCV*100)
		pressureStr := fmt.Sprintf("%d runs", r.Stats.PressureRunCount)

		fmt.Printf("%-8s | %4d | %-15s | %11.0f | %8s | %6.0f | %6.0f | %5.1f | %5s | %8s\n",
			r.Config, r.MaxSize, r.Image,
			r.Stats.MeanTime, cvPct, r.Stats.MeanInTok, r.Stats.MeanOutTok,
			r.Stats.MeanTPS, successPct, pressureStr)
	}
}

func printPressureSummary(results []AggregatedResult) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("MEMORY PRESSURE SUMMARY")
	fmt.Println(strings.Repeat("=", 100))

	totalRuns := 0
	pressureRuns := 0
	slowTokenRuns := 0
	highFaultsRuns := 0
	lowRAMRuns := 0
	truncatedRuns := 0
	var minRAM uint64 = ^uint64(0)
	var maxFaults uint64 = 0

	for _, r := range results {
		for _, run := range r.Runs {
			totalRuns++
			if run.Flags.Any() {
				pressureRuns++
			}
			if run.Flags.SlowToken {
				slowTokenRuns++
			}
			if run.Flags.HighFaults {
				highFaultsRuns++
			}
			if run.Flags.LowRAM {
				lowRAMRuns++
			}
			if run.Flags.Truncated {
				truncatedRuns++
			}
			if run.System.AvailableRAM_MB > 0 && run.System.AvailableRAM_MB < minRAM {
				minRAM = run.System.AvailableRAM_MB
			}
			if run.System.PageFaultsDelta > maxFaults {
				maxFaults = run.System.PageFaultsDelta
			}
		}
	}

	if minRAM == ^uint64(0) {
		minRAM = 0
	}

	pressurePct := 0.0
	if totalRuns > 0 {
		pressurePct = float64(pressureRuns) / float64(totalRuns) * 100
	}

	fmt.Printf("Total runs:           %d\n", totalRuns)
	fmt.Printf("Runs with pressure:   %d (%.1f%%)\n", pressureRuns, pressurePct)
	fmt.Printf("  - Slow token:       %d\n", slowTokenRuns)
	fmt.Printf("  - High page faults: %d\n", highFaultsRuns)
	fmt.Printf("  - Low RAM:          %d\n", lowRAMRuns)
	fmt.Printf("  - Truncated output: %d\n", truncatedRuns)
	fmt.Printf("Min available RAM:    %d MB\n", minRAM)
	fmt.Printf("Max page faults:      %d\n", maxFaults)
	fmt.Println(strings.Repeat("=", 100))
}

func printConfigSummary(results []AggregatedResult) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("SUMMARY BY CONFIG + MAXSIZE")
	fmt.Println(strings.Repeat("=", 100))

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
		fmt.Printf("%-8s @%3d: avgTime %6.0fms | avgTimeVar %3.0f%% | inTok %4.0f | outTok %3.0f | Tok/s %4.1f\n",
			k.config, k.maxSize,
			mean(stats.times),
			avgCV,
			mean(stats.inTok),
			mean(stats.outTok),
			mean(stats.tps))
	}

	fmt.Println(strings.Repeat("=", 100))
}

func saveCSV(info BenchmarkInfo, results []AggregatedResult) {
	timestamp := time.Now().Format("20060102_150405")

	prompt := strings.ReplaceAll(info.Prompt, "\"", "'")
	prompt = strings.ReplaceAll(prompt, "\n", " ")

	// Save grouped results
	groupedFile := fmt.Sprintf("results/performVis_grp_%s.csv", timestamp)
	var sb strings.Builder
	sb.WriteString("model,mmproj,cache_k,cache_v,max_tok,temp,prompt,")
	sb.WriteString("config,ctx_win,nbatch,nubatch,")
	sb.WriteString("max_size,image,orig_w,orig_h,resized_w,resized_h,bytes,")
	sb.WriteString("mean_ms,cv_pct,in_tok,out_tok,tps,success_rate,")
	sb.WriteString("mean_ms_per_tok,min_ram_mb,max_page_faults,pressure_runs\n")

	for _, r := range results {
		var cfg ConfigVariant
		for _, c := range info.Configs {
			if c.Name == r.Config {
				cfg = c
				break
			}
		}

		sb.WriteString(fmt.Sprintf("\"%s\",\"%s\",%s,%s,%d,%.1f,\"%s\",",
			info.ModelFile, info.ProjFile,
			cacheTypeName(cfg.CacheTypeK), cacheTypeName(cfg.CacheTypeV),
			P.MaxTokens, P.Temperature, prompt))
		sb.WriteString(fmt.Sprintf("%s,%d,%d,%d,",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch))
		sb.WriteString(fmt.Sprintf("%d,%s,%d,%d,%d,%d,%d,",
			r.MaxSize, r.Image,
			r.ImageInfo.OriginalW, r.ImageInfo.OriginalH,
			r.ImageInfo.ResizedW, r.ImageInfo.ResizedH, r.ImageInfo.Bytes))
		sb.WriteString(fmt.Sprintf("%.0f,%.0f,%.0f,%.0f,%.1f,%.2f,",
			r.Stats.MeanTime, r.Stats.VarianceCV*100,
			r.Stats.MeanInTok, r.Stats.MeanOutTok, r.Stats.MeanTPS, r.Stats.SuccessRate))
		sb.WriteString(fmt.Sprintf("%.1f,%d,%d,%d\n",
			r.Stats.MeanMsPerToken, r.Stats.MinAvailableRAM, r.Stats.MaxPageFaults, r.Stats.PressureRunCount))
	}

	if err := os.WriteFile(groupedFile, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("failed to save grouped CSV: %v\n", err)
	} else {
		fmt.Printf("\nGrouped results saved to: %s\n", groupedFile)
	}

	// Save individual results
	individualFile := fmt.Sprintf("results/performVis_ind_%s.csv", timestamp)
	sb.Reset()
	sb.WriteString("config,max_size,image,run,elapsed_ms,in_tok,out_tok,tps,ms_per_tok,")
	sb.WriteString("avail_ram_mb,page_faults,flag_slow,flag_faults,flag_ram,flag_trunc,")
	sb.WriteString("description,error\n")

	for _, r := range results {
		for _, run := range r.Runs {
			desc := strings.ReplaceAll(run.Description, "\"", "'")
			desc = strings.ReplaceAll(desc, "\n", " ")
			errStr := strings.ReplaceAll(run.Error, "\"", "'")

			sb.WriteString(fmt.Sprintf("%s,%d,%s,%d,%d,%d,%d,%.1f,%.1f,",
				r.Config, r.MaxSize, r.Image, run.Run,
				run.Elapsed.Milliseconds(),
				run.Metrics.PromptTokens, run.Metrics.CompletionTokens,
				run.Metrics.TokensPerSecond, run.Metrics.MsPerToken))
			sb.WriteString(fmt.Sprintf("%d,%d,%t,%t,%t,%t,",
				run.System.AvailableRAM_MB, run.System.PageFaultsDelta,
				run.Flags.SlowToken, run.Flags.HighFaults, run.Flags.LowRAM, run.Flags.Truncated))
			sb.WriteString(fmt.Sprintf("\"%s\",\"%s\"\n", desc, errStr))
		}
	}

	if err := os.WriteFile(individualFile, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("failed to save individual CSV: %v\n", err)
	} else {
		fmt.Printf("Individual results saved to: %s\n", individualFile)
	}
}
