# Reference Dockerfile for the vLLM fast path.
#
# Build context expects:
#   ./_steering.py
#   ./qwen3.py            (the patched model file matching your vLLM version)
#   ./vectors/*_chosen.npy
#
# Build:
#   docker build -t emotion-steering-vllm:0.1 .
#
# Run (single L4):
#   docker run -d --name emotion-steering --gpus all -p 8000:8000 \
#     -e STEERING_LAYERS=20,21,22 \
#     -e STEERING_EMOTIONS=anger,joy,sadness,disgust,fear,surprise \
#     -e STEERING_HIDDEN_DIM=4096 \
#     emotion-steering-vllm:0.1 \
#     --model Qwen/Qwen3-8B --served-model-name qwen3-8b \
#     --dtype bfloat16 --max-model-len 8192 \
#     --gpu-memory-utilization 0.90 --max-num-seqs 32 \
#     --enforce-eager --host 0.0.0.0 --port 8000

FROM vllm/vllm-openai:v0.20.0

# Steering module + Qwen3 model patch.
COPY _steering.py /usr/local/lib/python3.12/dist-packages/vllm/_steering.py
COPY qwen3.py     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3.py

# Bake vectors at the default load path.
COPY vectors/ /opt/vllm-vectors/

# Hook so the monkey-patches install in any vLLM-importing process,
# including the engine subprocess.
RUN echo "" >> /usr/local/lib/python3.12/dist-packages/vllm/__init__.py \
 && echo "# emotion-steering: steering extension" >> /usr/local/lib/python3.12/dist-packages/vllm/__init__.py \
 && echo "try:" >> /usr/local/lib/python3.12/dist-packages/vllm/__init__.py \
 && echo "    from vllm import _steering as _steering  # noqa: F401" >> /usr/local/lib/python3.12/dist-packages/vllm/__init__.py \
 && echo "except Exception as _e:" >> /usr/local/lib/python3.12/dist-packages/vllm/__init__.py \
 && echo "    import logging as _l; _l.getLogger('vllm').exception('steering init failed: %s', _e)" >> /usr/local/lib/python3.12/dist-packages/vllm/__init__.py
