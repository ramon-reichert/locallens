include .env

VISION_MODEL_URL = https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf
VISION_PROJ_URL  = https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf
EMBED_MODEL_URL  = https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf

# Build the app without a console window (production mode).
build:
	CGO_ENABLED=0 go build -ldflags "-H windowsgui" -o locallens.exe ./cmd/locallens/

# Build the app with console output visible for debugging.
build-logs:
	CGO_ENABLED=0 go build -o locallens.exe ./cmd/locallens/

# Download llama.cpp libraries and pull the vision and embedding GGUF models.
setup:
	kronk libs --local
	kronk model pull --local $(VISION_MODEL_URL) $(VISION_PROJ_URL)
	kronk model pull --local $(EMBED_MODEL_URL)

# Run fast unit tests (no model loading).
test-unit:
	CGO_ENABLED=0 go test -timeout 15s -v -count=1 ./internal/service/tests/unittests/...

# Run service-level integration tests (loads models, indexes and searches images).
test-service:
	CGO_ENABLED=0 go test -timeout 30m -v -count=1 ./internal/service/tests

# Run vision model performance benchmarks across config and image size variants.
test-performance-vision:
	CGO_ENABLED=0 go test -timeout 60m -v -run TestVisionPerformance ./internal/service/tests/performance/...

# Run embedding similarity tests to evaluate search quality.
test-performance-similarity:
	CGO_ENABLED=0 go test -v -run TestSimilarity ./internal/service/tests/performance/...
