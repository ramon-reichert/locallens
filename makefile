run:
	CGO_ENABLED=0 go run -mod=vendor ./cmd/locallens/main.go

# Build the app without a console window (production mode).
build:
	CGO_ENABLED=0 go build -mod=vendor -ldflags "-H windowsgui" -o locallens.exe ./cmd/locallens/

# Build the app with console output visible for debugging.
build-logs:
	CGO_ENABLED=0 go build -mod=vendor -o locallens.exe ./cmd/locallens/

# Download llama.cpp libraries and pull the vision and embedding GGUF models.
# All config (model URLs, engine params, prompt) lives in internal/platform/config/config.go.
setup:
	CGO_ENABLED=0 go run -mod=vendor ./cmd/setup/

# Run fast unit tests (no model loading).
test-unit:
	CGO_ENABLED=0 go test -mod=vendor -timeout 15s -v -count=1 ./internal/...

# Run service-level integration tests (loads models, indexes and searches images).
test-service:
	CGO_ENABLED=0 go test -mod=vendor -tags "integration" -timeout 30m -v -count=1 ./internal/service/tests

# Run vision model performance benchmarks across config and image size variants.
test-performance-vision:
	CGO_ENABLED=0 go test -mod=vendor -tags "integration" -timeout 60m -v -run TestVisionPerformance ./internal/service/tests/performance/...

# Run embedding similarity tests to evaluate search quality.
test-performance-similarity:
	CGO_ENABLED=0 go test -mod=vendor -tags "integration" -v -run TestSimilarity ./internal/service/tests/performance/...
