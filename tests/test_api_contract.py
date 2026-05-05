"""Test the FastAPI contract of the HF backend without loading a real model.

We monkey-patch HFSteeringRunner so the test doesn't need GPU/transformers
weights, then use FastAPI's TestClient to hit each endpoint.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from emotion_steering.serve import hf as serve_hf_mod
from emotion_steering.vectors import load_bundle

EXAMPLE_BUNDLE = Path(__file__).resolve().parent.parent / "examples" / "qwen3-8b-ekman6"


class FakeRunner:
    """Stub that satisfies build_app() without actually loading a model."""

    def __init__(self, bundle, model_name="qwen3-8b"):
        self.bundle = bundle
        self.model_name = model_name

    async def chat(self, req):
        # Echo the steering spec so the test can assert it was received.
        spec = req.steering or (req.vllm_xargs or {}).get("steering")
        return serve_hf_mod.ChatResponse(
            id="chatcmpl-test",
            created=0,
            model=req.model,
            choices=[
                serve_hf_mod.ChatChoice(
                    index=0,
                    message=serve_hf_mod.ChatMessage(
                        role="assistant",
                        content=f"steering={spec}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=serve_hf_mod.ChatUsage(
                prompt_tokens=10, completion_tokens=4, total_tokens=14,
            ),
        )


@pytest.fixture
def client():
    if not EXAMPLE_BUNDLE.exists():
        pytest.skip("example bundle not present")
    bundle = load_bundle(EXAMPLE_BUNDLE)
    runner = FakeRunner(bundle)
    app = serve_hf_mod.build_app(runner, api_key=None)
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["data"][0]["id"] == "qwen3-8b"


def test_emotions_endpoint(client):
    r = client.get("/v1/emotions")
    assert r.status_code == 200
    body = r.json()
    assert body["emotions"] == ["anger", "joy", "sadness", "disgust", "fear", "surprise"]
    assert body["id_map"] == {
        "anger": 0, "joy": 1, "sadness": 2, "disgust": 3, "fear": 4, "surprise": 5,
    }
    assert body["chosen_layers"] == [20, 21, 22]
    assert body["model"] == "Qwen/Qwen3-8B"


def test_chat_completion_passes_steering_through_top_level(client):
    r = client.post("/v1/chat/completions", json={
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "steering": [1, 1.5],
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "steering=[1, 1.5]" in content


def test_chat_completion_passes_steering_through_vllm_xargs(client):
    r = client.post("/v1/chat/completions", json={
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "vllm_xargs": {"steering": [3, 1.0]},
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "steering=[3, 1.0]" in content


def test_auth_blocks_when_key_required():
    if not EXAMPLE_BUNDLE.exists():
        pytest.skip("example bundle not present")
    bundle = load_bundle(EXAMPLE_BUNDLE)
    app = serve_hf_mod.build_app(FakeRunner(bundle), api_key="secret")
    c = TestClient(app)
    assert c.get("/v1/emotions").status_code == 401
    r = c.get("/v1/emotions", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
