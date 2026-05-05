"""Serving backends. Two paths:

- hf:   FastAPI + HF transformers, model-agnostic, slow.
- vllm: patched vLLM with continuous batching. Currently supports Qwen3.
        See .claude/skills/extend-vllm-fast-path.md to add an architecture.
"""
