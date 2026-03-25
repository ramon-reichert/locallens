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
	"strconv"
	"strings"
	"testing"
	"time"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/webp"

	kronksdk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/devices"

	"github.com/ramon-reichert/locallens/internal/platform/config"
	"github.com/ramon-reichert/locallens/internal/platform/sysmon"
	"github.com/ramon-reichert/locallens/internal/service/image"
	"github.com/ramon-reichert/locallens/internal/service/tests/testsboot"
)

const (
	DefaultRepetitions = 1

	// Pressure detection thresholds
	ThresholdSlowTPS      = 10.0   // Flag if Kronk's TokensPerSecond < this (at 10 tps, 300 tokens = 30s)
	ThresholdPageFaults   = 500000 // Flag if page faults delta > this (soft faults from mmap are normal)
	ThresholdLowRAM_MB    = 500    // Flag if available RAM < this (MB)
	ThresholdMinOutputTok = 20     // Flag if output tokens < this
)

var defaultMaxSizes = []int{64, 256} //64, 128, 256, 384, 512

// defaultConfigs returns test config variants. The first entry always matches
// the app defaults so one test row represents real production behavior.
func defaultConfigs() []ConfigVariant {
	v := testsboot.Cfg.Vision
	return []ConfigVariant{
		{"app", v.ContextWindow, v.NBatch, v.NUBatch, model.GGMLType(config.ParseGGMLType(v.CacheTypeK)), model.GGMLType(config.ParseGGMLType(v.CacheTypeV))},
		//	{"small", 2048, 1024, 512, model.GGMLTypeQ8_0, model.GGMLTypeQ8_0},
	}
}

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
	PromptTokens       int
	ReasoningTokens    int
	CompletionTokens   int
	OutputTokens       int
	TotalTokens        int
	TokensPerSecond    float64
	TimeToFirstTokenMS float64
	GenerationTimeMS   float64 // Elapsed - TTFT: pure token generation time
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
	DecodeErrorMsg  string
}

func (f PressureFlags) Any() bool {
	return f.SlowToken || f.HighFaults || f.LowRAM || f.Truncated || f.DecodeErrorCode != 0
}

type RunResult struct {
	Run         int
	Elapsed     time.Duration
	OutputToks  int
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
	SuccessRate float64
	VarianceCV  float64

	// Kronk timing breakdown
	MeanTTFT_MS   float64
	MeanGenTimeMS float64

	// Memory pressure stats
	MaxPageFaults    uint64
	MinAvailableRAM  uint64
	PressureRunCount int
}

type ModelMemory struct {
	ConfigName string
	VRAMTotal  int64
	SlotMemory int64
}

type HardwareInfo struct {
	GPUName      string
	GPUType      string
	GPUTotalMB   uint64
	GPUFreeMB    uint64
	HasGPU       bool
	SystemRAM_MB uint64
	GPUOffload   bool
}

type BenchmarkInfo struct {
	ModelFile   string
	ProjFile    string
	Prompt      string
	MaxTokens   int
	Temperature float64
	Configs     []ConfigVariant
	MaxSizes    []int
	Memory      []ModelMemory
	Hardware    HardwareInfo
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

	P := testsboot.Cfg.Prompt
	fullPrompt := P.SystemPrompt + "\n\n" + P.UserPrompt + "\n"
	configs := defaultConfigs()

	hw := detectHardware()

	info := BenchmarkInfo{
		ModelFile:   filepath.Base(testsboot.VisionPaths.ModelFiles[0]),
		ProjFile:    filepath.Base(testsboot.VisionPaths.ProjFile),
		Prompt:      fullPrompt,
		MaxTokens:   P.MaxTokens,
		Temperature: P.Temperature,
		Configs:     configs,
		MaxSizes:    defaultMaxSizes,
		Hardware:    hw,
	}

	var results []AggregatedResult

	for _, cfg := range configs {
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

		mi := krn.ModelInfo()

		slotMem := mi.SlotMemory
		if slotMem == 0 {
			slotMem = estimateSlotMemory(mi, cfg)
		}

		info.Memory = append(info.Memory, ModelMemory{
			ConfigName: cfg.Name,
			VRAMTotal:  mi.VRAMTotal,
			SlotMemory: slotMem,
		})

		fmt.Printf("    Model: %s | Type: %s | VRAM: %.1f MB | KV Slots: %.1f MB\n\n",
			mi.ID, mi.Type, float64(mi.VRAMTotal)/(1024*1024), float64(slotMem)/(1024*1024))

		// Warmup: run a tiny inference to trigger lazy initialization (projector
		// loading, GPU buffer allocation, etc.) so the first real measurement is
		// not penalized by one-time startup costs.
		warmupImage, warmupErr := image.Resize(images[0], 64)
		if warmupErr == nil {
			fmt.Printf("    Warmup run...")
			warmupStart := time.Now()
			warmupData := model.D{
				"messages": []model.D{
					{"role": "user", "content": warmupImage},
					{"role": "user", "content": "hi"},
				},
				"temperature": 0.0,
				"max_tokens":  1,
			}
			krn.Chat(ctx, warmupData)
			fmt.Printf(" done (%dms)\n\n", time.Since(warmupStart).Milliseconds())
		}

		for _, maxSize := range defaultMaxSizes {

			for _, imgPath := range images {
				aggResult := runWithRepetitions(ctx, krn, imgPath, cfg, P, maxSize, repetitions)
				results = append(results, aggResult)

				fmt.Printf("     >>>> %s: maxSize=%d config=%s || avgTime=%.0fms | avgTTFT=%.0fms | timeVar=%.0f%% | %d/%d success\n\n\n",
					filepath.Base(imgPath),
					maxSize,
					cfg.Name,
					aggResult.Stats.MeanTime,
					aggResult.Stats.MeanTTFT_MS,
					aggResult.Stats.VarianceCV*100,
					int(aggResult.Stats.SuccessRate*float64(repetitions)),
					repetitions)
			}
		}

		krn.Unload(ctx)

	}

	printHardware(info.Hardware)
	printConfigs(info)
	printConfigSummary(results)
	printGroupedResults(results)
	printPressureSummary(results)
	saveCSV(info, results)
}

func runWithRepetitions(ctx context.Context, krn *kronksdk.Kronk, imgPath string, cfg ConfigVariant, P config.VisionPrompt, maxSize, reps int) AggregatedResult {
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

	for i := 0; i < reps; i++ {
		run := runSingleInference(ctx, krn, imageData, i+1, P)

		fmt.Printf("=> Run %d: %s | maxSize=%d | Config: %s (ctx=%d, Nbatch=%d, NUbatch=%d) | maxTok=%d, temp=%.1f",
			run.Run,
			filepath.Base(imgPath),
			maxSize,
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, P.MaxTokens, P.Temperature)

		fmt.Printf("\n   Timing:  total=%dms | ttft=%.0fms | gen=%.0fms",
			run.Elapsed.Milliseconds(),
			run.Metrics.TimeToFirstTokenMS,
			run.Metrics.GenerationTimeMS)

		fmt.Printf("\n   Tokens:  in=%d out=%d (reason=%d compl=%d) | Tok/s=%.1f",
			run.Metrics.PromptTokens,
			run.Metrics.OutputTokens,
			run.Metrics.ReasoningTokens,
			run.Metrics.CompletionTokens,
			run.Metrics.TokensPerSecond)

		fmt.Printf("\n   Memory:  avlbRAM=%dMB pgFaults=%d",
			run.System.AvailableRAM_MB,
			run.System.PageFaultsDelta)

		if run.Flags.Any() {
			fmt.Print(" | FLAGS:")
			if run.Flags.SlowToken {
				fmt.Print(" [SlowTPS]")
			}
			if run.Flags.HighFaults {
				fmt.Print(" [HiFaults]")
			}
			if run.Flags.LowRAM {
				fmt.Print(" [LowRAM]")
			}
			if run.Flags.Truncated {
				fmt.Print(" [LowOutTok]")
			}
			if run.Flags.DecodeErrorCode != 0 {
				fmt.Printf(" [DECODE:%d-%s]", run.Flags.DecodeErrorCode, run.Flags.DecodeErrorMsg)
			}
		}

		fmt.Printf("\n   Description: %s\n\n", run.Description)
		if run.Error != "" {
			fmt.Printf("   Error: %s\n", run.Error)
		}
		fmt.Println()

		result.Runs = append(result.Runs, run)

		if i < reps-1 {
			time.Sleep(5 * time.Second)
		}
	}

	// Calculate stats
	result.Stats = calculateStats(result.Runs)

	return result
}

func runSingleInference(ctx context.Context, krn *kronksdk.Kronk, imageData []byte, runNum int, P config.VisionPrompt) RunResult {
	run := RunResult{Run: runNum}

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

	snapAfter := sysmon.Capture()
	if snapAfter.PageFaults >= snapBefore.PageFaults {
		run.System.PageFaultsDelta = snapAfter.PageFaults - snapBefore.PageFaults
	}

	if err != nil {
		run.Error = fmt.Sprintf("chat: %v", err)
	}

	if len(resp.Choices) > 0 && resp.Choices[0].FinishReason() == model.FinishReasonError {
		errMsg := ""
		if resp.Choices[0].Delta != nil {
			errMsg = resp.Choices[0].Delta.Content
		} else if resp.Choices[0].Message != nil {
			errMsg = resp.Choices[0].Message.Content
		}
		run.Error = fmt.Sprintf("decode error: %s", errMsg)
		run.Flags.DecodeErrorMsg = errMsg
		switch {
		case strings.Contains(errMsg, "context window is full"):
			run.Flags.DecodeErrorCode = 1
		case strings.Contains(errMsg, "cancelled"):
			run.Flags.DecodeErrorCode = 2
		case strings.Contains(errMsg, "input could not be processed"):
			run.Flags.DecodeErrorCode = -1
		case strings.Contains(errMsg, "internal error"):
			run.Flags.DecodeErrorCode = -2
		default:
			run.Flags.DecodeErrorCode = -99
		}
	}

	if len(resp.Choices) > 0 && resp.Choices[0].Message != nil {
		run.Description = resp.Choices[0].Message.Content
	}

	u := resp.Usage
	genTimeMS := float64(run.Elapsed.Milliseconds()) - u.TimeToFirstTokenMS

	run.OutputToks = u.OutputTokens
	run.Metrics = ModelMetrics{
		PromptTokens:       u.PromptTokens,
		ReasoningTokens:    u.ReasoningTokens,
		CompletionTokens:   u.CompletionTokens,
		OutputTokens:       u.OutputTokens,
		TotalTokens:        u.TotalTokens,
		TokensPerSecond:    u.TokensPerSecond,
		TimeToFirstTokenMS: u.TimeToFirstTokenMS,
		GenerationTimeMS:   genTimeMS,
	}

	// Pressure flags using Kronk's measured TokensPerSecond (pure generation, excludes prefill).
	if u.TokensPerSecond > 0 && u.TokensPerSecond < ThresholdSlowTPS {
		run.Flags.SlowToken = true
	}
	if run.System.PageFaultsDelta > ThresholdPageFaults {
		run.Flags.HighFaults = true
	}
	if run.System.AvailableRAM_MB > 0 && run.System.AvailableRAM_MB < ThresholdLowRAM_MB {
		run.Flags.LowRAM = true
	}
	if u.OutputTokens < ThresholdMinOutputTok {
		run.Flags.Truncated = true
	}

	return run
}

func calculateStats(runs []RunResult) Stats {
	var stats Stats
	var times []float64
	var inTok, outTok, tps []float64
	var ttft, genTime []float64
	successCount := 0

	stats.MinAvailableRAM = ^uint64(0)

	for _, r := range runs {
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
			outTok = append(outTok, float64(r.Metrics.OutputTokens))
			tps = append(tps, r.Metrics.TokensPerSecond)
			ttft = append(ttft, r.Metrics.TimeToFirstTokenMS)
			genTime = append(genTime, r.Metrics.GenerationTimeMS)
		}
	}

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
	stats.MeanTTFT_MS = mean(ttft)
	stats.MeanGenTimeMS = mean(genTime)

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

func detectHardware() HardwareInfo {
	devs := devices.List()

	systemRAM := devs.SystemRAMBytes / (1024 * 1024)
	if systemRAM == 0 {
		systemRAM = sysmon.Capture().AvailableRAM_MB
	}

	hw := HardwareInfo{
		SystemRAM_MB: systemRAM,
		GPUOffload:   devs.SupportsGPUOffload,
	}

	for _, d := range devs.Devices {
		if strings.HasPrefix(d.Type, "gpu_") {
			hw.HasGPU = true
			hw.GPUName = d.Name
			hw.GPUType = d.Type
			hw.GPUTotalMB = d.TotalBytes / (1024 * 1024)
			hw.GPUFreeMB = d.FreeBytes / (1024 * 1024)
			break
		}
	}

	return hw
}

func printHardware(hw HardwareInfo) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("HARDWARE")
	fmt.Println(strings.Repeat("=", 100))

	if hw.HasGPU {
		fmt.Printf("GPU:         %s (%s) | %d MB total | %d MB free\n", hw.GPUName, hw.GPUType, hw.GPUTotalMB, hw.GPUFreeMB)
	} else {
		fmt.Printf("GPU:         none\n")
	}
	fmt.Printf("System RAM:  %d MB\n", hw.SystemRAM_MB)
	fmt.Printf("GPU Offload: %v\n", hw.GPUOffload)
}

func printConfigs(info BenchmarkInfo) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("CONFIGS")
	fmt.Println(strings.Repeat("=", 100))

	fmt.Printf("Model:       %s\n", info.ModelFile)
	fmt.Printf("MMProj:      %s\n", info.ProjFile)
	fmt.Printf("MaxSizes:    %v\n", info.MaxSizes)
	fmt.Printf("MaxTokens:   %d\n", info.MaxTokens)
	fmt.Printf("Temperature: %.1f\n", info.Temperature)
	fmt.Println()
	fmt.Printf("Prompt: %s\n", info.Prompt)
	fmt.Println()

	hw := info.Hardware
	useLabel := "GPU Use%"
	if !hw.HasGPU {
		useLabel = "RAM Use%"
	}

	fmt.Printf("%-10s | %6s | %6s | %7s | %8s | %8s | %10s | %10s | %8s\n",
		"Name", "CtxWin", "NBatch", "NUBatch", "CacheK", "CacheV", "VRAM(MB)", "KVSlot(MB)", useLabel)
	fmt.Println(strings.Repeat("-", 100))
	for _, cfg := range info.Configs {
		var vramMB, slotMB float64
		for _, mem := range info.Memory {
			if mem.ConfigName == cfg.Name {
				vramMB = float64(mem.VRAMTotal) / (1024 * 1024) // VRAMTotal = llama.ModelSize + SlotMemory
				slotMB = float64(mem.SlotMemory) / (1024 * 1024)
				break
			}
		}

		var usePct float64
		if hw.HasGPU && hw.GPUTotalMB > 0 {
			usePct = vramMB / float64(hw.GPUTotalMB) * 100
		} else if hw.SystemRAM_MB > 0 {
			usePct = vramMB / float64(hw.SystemRAM_MB) * 100
		}

		fmt.Printf("%-10s | %6d | %6d | %7d | %8s | %8s | %10.1f | %10.1f | %7.1f%%\n",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch,
			cacheTypeName(cfg.CacheTypeK), cacheTypeName(cfg.CacheTypeV),
			vramMB, slotMB, usePct)
	}
}

// estimateSlotMemory computes KV cache slot memory when Kronk's ModelInfo
// returns 0 (e.g. because the GGUF metadata lacks attention.key_length or
// attention.value_length).
// Formula: nSeqMax * contextWindow * blockCount * headCountKV * (keyLen + valLen) * bytesPerElement
func estimateSlotMemory(mi model.ModelInfo, cfg ConfigVariant) int64 {
	arch := mi.Metadata["general.architecture"]
	if arch == "" {
		return 0
	}

	parseInt := func(key string) (int64, bool) {
		v, ok := mi.Metadata[key]
		if !ok {
			return 0, false
		}
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return 0, false
		}
		return n, true
	}

	blockCount, ok := parseInt(arch + ".block_count")
	if !ok {
		return 0
	}

	headCountKV, ok := parseInt(arch + ".attention.head_count_kv")
	if !ok {
		headCountKV, ok = parseInt(arch + ".attention.head_count")
		if !ok {
			return 0
		}
	}

	headCount, _ := parseInt(arch + ".attention.head_count")

	// Derive key/value lengths from embedding_length / head_count when the
	// explicit metadata keys are absent.
	keyLen, okK := parseInt(arch + ".attention.key_length")
	valLen, okV := parseInt(arch + ".attention.value_length")
	if (!okK || !okV) && headCount > 0 {
		embLen, okE := parseInt(arch + ".embedding_length")
		if okE {
			dim := embLen / headCount
			if !okK {
				keyLen = dim
			}
			if !okV {
				valLen = dim
			}
		} else {
			return 0
		}
	}
	if keyLen == 0 || valLen == 0 {
		return 0
	}

	bytesPerElement := cacheGGMLBytes(cfg.CacheTypeK, cfg.CacheTypeV)
	contextWindow := int64(cfg.ContextWindow)

	kvPerTokenPerLayer := headCountKV * (keyLen + valLen) * bytesPerElement
	kvPerSlot := contextWindow * blockCount * kvPerTokenPerLayer

	return kvPerSlot
}

func cacheGGMLBytes(typeK, typeV model.GGMLType) int64 {
	ggmlBytes := func(t model.GGMLType) int64 {
		switch t {
		case model.GGMLTypeF16, model.GGMLTypeBF16:
			return 2
		case model.GGMLTypeQ8_0:
			return 1
		case model.GGMLTypeQ4_0, model.GGMLTypeQ4_1:
			return 1
		default:
			return 2
		}
	}
	bk := ggmlBytes(typeK)
	bv := ggmlBytes(typeV)
	if bk > bv {
		return bk
	}
	return bv
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
	fmt.Println("\n" + strings.Repeat("=", 130))
	fmt.Println("GROUPED RESULTS")
	fmt.Println(strings.Repeat("=", 130))
	fmt.Printf("%-8s | %4s | %-15s | %11s | %9s | %9s | %8s | %6s | %6s | %5s | %5s | %8s\n",
		"Config", "Max", "Image", "AvgTime(ms)", "TTFT(ms)", "GenTime", "TimeVar%", "InTok", "OutTok", "Tok/s", "Succ", "Pressure")
	fmt.Println(strings.Repeat("-", 130))

	for _, r := range results {
		successPct := fmt.Sprintf("%.0f%%", r.Stats.SuccessRate*100)
		cvPct := fmt.Sprintf("%.0f%%", r.Stats.VarianceCV*100)
		pressureStr := fmt.Sprintf("%d runs", r.Stats.PressureRunCount)

		fmt.Printf("%-8s | %4d | %-15s | %11.0f | %9.0f | %9.0f | %8s | %6.0f | %6.0f | %5.1f | %5s | %8s\n",
			r.Config, r.MaxSize, r.Image,
			r.Stats.MeanTime, r.Stats.MeanTTFT_MS, r.Stats.MeanGenTimeMS,
			cvPct, r.Stats.MeanInTok, r.Stats.MeanOutTok,
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
		ttft   []float64
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
			stats.ttft = append(stats.ttft, r.Stats.MeanTTFT_MS)
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

		// When repetitions > 1, avgCV is the average of per-image CVs.
		// When repetitions == 1, per-image CV is always 0, so we compute
		// the CV across all images in this config+maxSize group instead.
		avgCV := 0.0
		if stats.count > 0 && stats.cvSum > 0 {
			avgCV = stats.cvSum / float64(stats.count) * 100
		} else if len(stats.times) > 1 {
			sd := stddev(stats.times)
			m := mean(stats.times)
			if m > 0 {
				avgCV = sd / m * 100
			}
		}

		fmt.Printf("%-8s @%3d: avgTime %6.0fms | ttft %6.0fms | avgTimeVar %3.0f%% | inTok %4.0f | outTok %3.0f | Tok/s %4.1f\n",
			k.config, k.maxSize,
			mean(stats.times),
			mean(stats.ttft),
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

	resultsDir := filepath.Join("results", "vision")
	os.MkdirAll(resultsDir, 0755)

	// Save grouped results
	groupedFile := filepath.Join(resultsDir, fmt.Sprintf("performVis_grp_%s.csv", timestamp))
	var sb strings.Builder

	hw := info.Hardware
	sb.WriteString(fmt.Sprintf("# hardware: gpu=%s gpu_type=%s gpu_total_mb=%d gpu_free_mb=%d system_ram_mb=%d gpu_offload=%v\n",
		hw.GPUName, hw.GPUType, hw.GPUTotalMB, hw.GPUFreeMB, hw.SystemRAM_MB, hw.GPUOffload))

	sb.WriteString("model,mmproj,cache_k,cache_v,max_tok,temp,prompt,")
	sb.WriteString("config,ctx_win,nbatch,nubatch,vram_mb,kv_slot_mb,")
	sb.WriteString("max_size,image,orig_w,orig_h,resized_w,resized_h,bytes,")
	sb.WriteString("mean_ms,mean_ttft_ms,mean_gen_ms,cv_pct,in_tok,out_tok,tps,success_rate,")
	sb.WriteString("min_ram_mb,max_page_faults,pressure_runs\n")

	for _, r := range results {
		var cfg ConfigVariant
		var vramMB, slotMB float64
		for _, c := range info.Configs {
			if c.Name == r.Config {
				cfg = c
				break
			}
		}
		for _, mem := range info.Memory {
			if mem.ConfigName == r.Config {
				vramMB = float64(mem.VRAMTotal) / (1024 * 1024)
				slotMB = float64(mem.SlotMemory) / (1024 * 1024)
				break
			}
		}

		sb.WriteString(fmt.Sprintf("\"%s\",\"%s\",%s,%s,%d,%.1f,\"%s\",",
			info.ModelFile, info.ProjFile,
			cacheTypeName(cfg.CacheTypeK), cacheTypeName(cfg.CacheTypeV),
			info.MaxTokens, info.Temperature, prompt))
		sb.WriteString(fmt.Sprintf("%s,%d,%d,%d,%.1f,%.1f,",
			cfg.Name, cfg.ContextWindow, cfg.NBatch, cfg.NUBatch, vramMB, slotMB))
		sb.WriteString(fmt.Sprintf("%d,%s,%d,%d,%d,%d,%d,",
			r.MaxSize, r.Image,
			r.ImageInfo.OriginalW, r.ImageInfo.OriginalH,
			r.ImageInfo.ResizedW, r.ImageInfo.ResizedH, r.ImageInfo.Bytes))
		sb.WriteString(fmt.Sprintf("%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.2f,",
			r.Stats.MeanTime, r.Stats.MeanTTFT_MS, r.Stats.MeanGenTimeMS,
			r.Stats.VarianceCV*100,
			r.Stats.MeanInTok, r.Stats.MeanOutTok, r.Stats.MeanTPS, r.Stats.SuccessRate))
		sb.WriteString(fmt.Sprintf("%d,%d,%d\n",
			r.Stats.MinAvailableRAM, r.Stats.MaxPageFaults, r.Stats.PressureRunCount))
	}

	if err := os.WriteFile(groupedFile, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("failed to save grouped CSV: %v\n", err)
	} else {
		fmt.Printf("\nGrouped results saved to: %s\n", groupedFile)
	}

	// Save individual results
	individualFile := filepath.Join(resultsDir, fmt.Sprintf("performVis_ind_%s.csv", timestamp))
	sb.Reset()
	sb.WriteString("config,max_size,image,run,elapsed_ms,ttft_ms,gen_ms,")
	sb.WriteString("in_tok,reason_tok,compl_tok,out_tok,tps,")
	sb.WriteString("avail_ram_mb,page_faults,flag_slow_tps,flag_faults,flag_ram,flag_low_out_tok,")
	sb.WriteString("description,error\n")

	for _, r := range results {
		for _, run := range r.Runs {
			desc := strings.ReplaceAll(run.Description, "\"", "'")
			desc = strings.ReplaceAll(desc, "\n", " ")
			errStr := strings.ReplaceAll(run.Error, "\"", "'")

			sb.WriteString(fmt.Sprintf("%s,%d,%s,%d,%d,%.0f,%.0f,",
				r.Config, r.MaxSize, r.Image, run.Run,
				run.Elapsed.Milliseconds(),
				run.Metrics.TimeToFirstTokenMS, run.Metrics.GenerationTimeMS))
			sb.WriteString(fmt.Sprintf("%d,%d,%d,%d,%.1f,",
				run.Metrics.PromptTokens, run.Metrics.ReasoningTokens,
				run.Metrics.CompletionTokens, run.Metrics.OutputTokens,
				run.Metrics.TokensPerSecond))
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
