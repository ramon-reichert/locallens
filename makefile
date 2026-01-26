test-performance:
	CGO_ENABLED=0 go test -timeout 60m -v -run TestVisionPerformance ./internal/service/tests/performance/...

test-unit:
	CGO_ENABLED=0 go test -v -count=1 ./internal/service/tests/unittests/...

test-service:
	CGO_ENABLED=0 go test -v -count=1 ./internal/service/tests/service_test.go