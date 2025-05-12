.PHONY: batch-http batch-grpc stress

batch-http:
	@uv run src/job/batch_http.py --path prompts.example.json


batch-grpc:
	@echo "not implemented"


# Start port forwarding in the background and run stress test
stress:
	@echo "Starting port forwarding..."
	@kubectl port-forward -n model-serving svc/tensorrt-service 8000:8000 & echo $$! > .port-forward.pid
	@sleep 2  # Give port forwarding a moment to establish
	@echo "Starting stress test..."
	@uv run locust -f src/job/stress.py
	@make stress-stop
