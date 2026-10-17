# LLM Service

A FastAPI starter for integrating an existing open-weight LLM. It separates HTTP/API concerns from model inference and supports an always-runnable stub mode plus optional local Transformers inference.

## Quick start
Python 3.10+:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit http://127.0.0.1:8000/docs. Default `MODEL_BACKEND=stub` is a deterministic demo, **not an LLM**, so tests run without model downloads.

## Run an actual model
```bash
pip install -r requirements-model.txt
export MODEL_BACKEND=transformers
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
The first run downloads model weights and can require multiple GB of RAM/disk. CPU works but may be slow; a compatible GPU is recommended. Read the model card/license before use. Model inference is optional so CI does not download weights.

## Docker
```bash
docker compose up --build
```
Default is stub mode. To serve a real model, build an image with `requirements-model.txt`, configure the model ID/cache and resources, and test on your target hardware.

## Endpoints
- `GET /healthz` liveness
- `GET /readyz` provider readiness
- `POST /v1/chat/completions` chat request
- `GET /docs` interactive API documentation

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Explain causal attention"}],"max_tokens":100}'
```

## Node.js integration guide

See [Node.js LLM Integration: Architecture, Setup, and Runbook](docs/nodejs-llm-integration.md) for the request lifecycle, architecture diagrams, installation, Express API example, response handling, local service integration, troubleshooting, and production hardening.

## Configuration
`MODEL_BACKEND=stub|transformers`, `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct`, `MODEL_DEVICE=auto|cpu`, `MAX_INPUT_CHARS=12000`.

## Production gap checklist
This is a production-oriented learning starter, not production-ready as-is. Before exposing it publicly add authentication/authorization, rate and concurrency limits, secrets management, structured metrics/tracing, load and security tests, model warmup, GPU capacity planning, timeouts/cancellation, audit/retention policy, and network restrictions. Never pass secrets or private data to a model without authorization. RAG and agents are separate capabilities, not requirements for basic inference.
