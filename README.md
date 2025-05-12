# TensorRT-LLM Model User

A Python client library for interacting with TensorRT-LLM model servers. This package provides a convenient interface to make inference requests to TensorRT-LLM models served via Triton Inference Server.

## Features

- Easy-to-use client interface for TensorRT-LLM model inference
- Support for various model architectures through the Triton client
- Built with modern Python practices and type hints
- Comprehensive error handling and logging

## Requirements

- Python >= 3.12
- Dependencies:
  - loguru >= 0.7.3
  - numpy >= 2.2.5
  - pydantic >= 2.11.4
  - requests >= 2.32.3

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tensorrt-llm-model-user.git
cd tensorrt-llm-model-user
```

2. Create and activate a virtual environment:
```bash
uv sync --frozen
```

## Usage

```python
uv run main.py
```

## License

See [LICENSE](LICENSE) file for details.
