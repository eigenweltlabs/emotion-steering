"""emotion-steering CLI: extract, test, serve."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="emotion-steering",
    help="Extract and serve activation-based emotion steering vectors (Konen et al. 2024) for any HF causal LM.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _exit_on_cache_permission(stage: str, exc: Exception) -> None:
    """Turn common HF cache permission failures into an actionable CLI message."""
    if not isinstance(exc, PermissionError) and "Permission denied" not in str(exc):
        raise exc
    console.print(f"[red]{stage} failed because a Hugging Face cache path is not writable.[/red]")
    console.print(
        "Set HF_HOME, HF_HUB_CACHE, or HF_DATASETS_CACHE to a writable directory, "
        "for example: [bold]export HF_HOME=$HOME/.cache/huggingface[/bold]"
    )
    console.print(f"[dim]underlying error: {exc}[/dim]")
    raise typer.Exit(1)


def _metadata_count_summary(md: dict[str, Any]) -> str:
    if "n_train" in md or "n_val" in md:
        return f"{md.get('n_train', 'n/a')} / {md.get('n_val', 'n/a')}"

    groups: list[str] = []
    for key in sorted(md):
        if not key.startswith("n_train_"):
            continue
        suffix = key.removeprefix("n_train_")
        val_key = f"n_val_{suffix}"
        groups.append(f"{suffix.replace('_', '/')}={md.get(key)} / {md.get(val_key, 'n/a')}")
    return "; ".join(groups) if groups else "n/a"


def _auc_tables(md: dict[str, Any], emotions: list[str]) -> list[tuple[str, list[str], Any]]:
    if md.get("auc_matrix") is not None:
        return [("Per-layer validation ROC-AUC (unitless)", emotions, md["auc_matrix"])]

    out: list[tuple[str, list[str], Any]] = []
    for key in sorted(md):
        if key.startswith("auc_matrix_"):
            cols = key.removeprefix("auc_matrix_").split("_")
            out.append((f"Validation ROC-AUC (unitless): {'/'.join(cols)}", cols, md[key]))
    return out


def _chosen_auc_summary(md: dict[str, Any], tables: list[tuple[str, list[str], Any]]) -> str:
    if md.get("mean_micro_auc_chosen") is not None:
        return f"{float(md['mean_micro_auc_chosen']):.3f}"

    search_layers = md.get("search_layers") or []
    chosen_layers = md.get("chosen_layers") or []
    if not search_layers or not chosen_layers or not tables:
        return "n/a"

    import numpy as np

    row_idxs = [i for i, li in enumerate(search_layers) if li in set(chosen_layers)]
    if not row_idxs:
        return "n/a"

    summaries: list[str] = []
    for _, cols, matrix in tables:
        arr = np.array(matrix, dtype=float)
        if arr.ndim != 2 or not row_idxs:
            continue
        summaries.append(f"{'/'.join(cols)}={arr[row_idxs].mean():.3f}")
    return "; ".join(summaries) if summaries else "n/a"


@app.command()
def extract(
    model: str = typer.Option(
        "Qwen/Qwen3-8B", "--model", "-m",
        help="HuggingFace causal-LM repo id.",
    ),
    emotions: str = typer.Option(
        "anger,joy,sadness,disgust,fear,surprise", "--emotions", "-e",
        help="Comma-separated target emotion names. Must exist as keys in the Ekman map.",
    ),
    output: Path = typer.Option(
        Path("./vectors"), "--output", "-o",
        help="Directory to write vectors + metadata.json.",
    ),
    search_layers: str | None = typer.Option(
        None, "--search-layers",
        help="Legacy explicit comma-separated layer indices to sweep.",
    ),
    layers: str | None = typer.Option(
        None, "--layers", "-l",
        help=(
            "Layer selector: early, mid, late, all, integer ids, or comma-separated "
            "mixes such as early,mid,late or 4,20,32."
        ),
    ),
    layer: int | None = typer.Option(
        None, "--layer",
        help="Single zero-indexed decoder layer. Forces --window 1.",
    ),
    window: int = typer.Option(3, "--window", help="Size of contiguous chosen layer window."),
    batch_size: int = typer.Option(16, "--batch-size"),
    max_length: int = typer.Option(256, "--max-length"),
    test_size: float = typer.Option(0.2, "--test-size", help="Validation split fraction."),
    seed: int = typer.Option(42, "--seed"),
    dtype: str = typer.Option("float16", "--dtype", help="float16 | bfloat16 | float32"),
    device: str = typer.Option("cuda", "--device"),
    save_full_sweep: bool = typer.Option(
        True, "--save-full-sweep/--no-save-full-sweep",
        help="Also save vectors for every searched layer, not just the chosen window.",
    ),
):
    """Extract activation-based emotion vectors (Konen Eq. 5) and save AUC validation report."""
    import torch

    from .dataset import EKMAN_MAP, load_goemotions_ekman, split_train_val
    from .extract import (
        CaptureConfig, build_vectors, capture_activations,
        default_search_layers, load_model_for_extraction, parse_layer_selector,
    )
    from .probe import best_window, probe_all_layers
    from .vectors import save_bundle

    targets = [e.strip() for e in emotions.split(",") if e.strip()]
    bad = set(targets) - set(EKMAN_MAP)
    if bad:
        console.print(f"[red]unknown emotions {sorted(bad)}; known: {sorted(EKMAN_MAP)}")
        raise typer.Exit(1)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype not in dtype_map:
        console.print(f"[red]unknown dtype {dtype}; pick from {list(dtype_map)}")
        raise typer.Exit(1)
    torch_dtype = dtype_map[dtype]

    console.rule(f"[bold]extract[/bold] model={model} emotions={targets}")

    console.print("[1/5] loading dataset (GoEmotions)...")
    try:
        records = load_goemotions_ekman(targets, seed=seed)
    except Exception as exc:
        _exit_on_cache_permission("Dataset load", exc)
    train_recs, val_recs = split_train_val(records, test_size=test_size, seed=seed)
    console.print(f"  train={len(train_recs)} | val={len(val_recs)} | balanced per class")

    console.print(f"[2/5] loading model ({dtype}, device={device})...")
    try:
        tokenizer, hf_model = load_model_for_extraction(model, dtype=torch_dtype, device=device)
    except Exception as exc:
        _exit_on_cache_permission("Model load", exc)
    num_layers = hf_model.config.num_hidden_layers
    hidden = hf_model.config.hidden_size

    selectors = [x is not None for x in (layer, layers, search_layers)]
    if sum(selectors) > 1:
        console.print("[red]use only one of --layer, --layers, or --search-layers[/red]")
        raise typer.Exit(1)
    if layer is not None:
        sl = parse_layer_selector(str(layer), num_layers)
        window = 1
    elif layers:
        try:
            sl = parse_layer_selector(layers, num_layers)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
    elif search_layers:
        try:
            sl = parse_layer_selector(search_layers, num_layers)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
    else:
        sl = default_search_layers(num_layers)
    if window > len(sl):
        console.print(f"[red]--window {window} is larger than selected layer count {len(sl)}[/red]")
        raise typer.Exit(1)
    console.print(f"  layers={num_layers} hidden={hidden} search_layers={sl}")

    cfg = CaptureConfig(
        search_layers=sl, capture_batch=batch_size,
        max_length=max_length, dtype=torch_dtype,
    )

    console.print("[3/5] capturing train activations...")
    train_acts = capture_activations(
        hf_model, tokenizer, train_recs, cfg,
        label="train", device=device, log_fn=console.print,
    )
    console.print("       capturing val activations...")
    val_acts = capture_activations(
        hf_model, tokenizer, val_recs, cfg,
        label="val", device=device, log_fn=console.print,
    )

    train_labels = [r.label for r in train_recs]
    val_labels = [r.label for r in val_recs]

    console.print("[4/5] building vectors (Konen Eq. 5: mean(class) - mean(rest))...")
    vec_dict = build_vectors(train_acts, train_labels, classes=targets)

    console.print("       running per-layer AUC probe...")
    t0 = time.time()
    auc = probe_all_layers(
        train_acts, train_labels, val_acts, val_labels,
        classes=targets, device=device,
    )
    console.print(f"       probe wall: {time.time() - t0:.0f}s")

    try:
        start_idx, mean_auc = best_window(auc, window=window, layer_ids=sl)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    chosen_layer_idxs = [sl[start_idx + k] for k in range(window)]
    console.print(
        f"       best {window}-layer window: {chosen_layer_idxs} "
        f"(mean validation ROC-AUC = {mean_auc:.3f}, unitless)"
    )

    # Print AUC table.
    tbl = Table(title="Per-layer validation ROC-AUC (unitless; 0.5 chance, 1.0 perfect)")
    tbl.add_column("layer", justify="right")
    for e in targets:
        tbl.add_column(e, justify="right")
    tbl.add_column("mean", justify="right", style="bold")
    for li_idx, li in enumerate(sl):
        row = [str(li)]
        for ei in range(len(targets)):
            row.append(f"{auc[li_idx, ei]:.3f}")
        row.append(f"{auc[li_idx].mean():.3f}")
        tbl.add_row(*row)
    console.print(tbl)

    console.print("[5/5] saving vectors + metadata...")
    chosen = {e: vec_dict[e][start_idx : start_idx + window] for e in targets}
    full_sweep = vec_dict if save_full_sweep else None

    metadata = {
        "model": model,
        "emotions": targets,
        "num_layers": num_layers,
        "hidden": hidden,
        "search_layers": sl,
        "chosen_layers": chosen_layer_idxs,
        "best_start_idx_in_search": start_idx,
        "auc_matrix": auc.tolist(),
        "mean_micro_auc_chosen": mean_auc,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "convention_capture": "last_token_post_block_residual_stream",
        "convention_inject": "post_block_residual_stream",
        "contrast": "internal: v_e = mean(e) - mean(rest of target emotions)",
        "normalization": "none",
        "seed": seed,
        "dtype": dtype,
        "extractor_version": "emotion-steering 0.1.0",
    }

    save_bundle(output, chosen=chosen, full_sweep=full_sweep, metadata=metadata)
    console.print(f"[green]wrote {output}[/green]")


@app.command()
def test(
    vectors: Path = typer.Argument(..., help="Path to a vector bundle (directory)."),
    show_norms: bool = typer.Option(True, "--norms/--no-norms"),
):
    """Show AUC + norm summary for a saved bundle (no GPU needed)."""
    from .vectors import load_bundle

    bundle = load_bundle(vectors)
    md = bundle.metadata
    console.rule(f"[bold]test[/bold] {vectors}")
    console.print(f"model:           {md.get('model')}")
    console.print(f"emotions:        {bundle.emotions}")
    console.print(f"chosen layers:   {bundle.chosen_layers}")
    console.print(f"hidden dim:      {md.get('hidden')}")
    console.print(f"n_train / n_val: {_metadata_count_summary(md)}")

    tables = _auc_tables(md, bundle.emotions)
    console.print(
        "mean chosen ROC-AUC: "
        f"{_chosen_auc_summary(md, tables)} [dim](unitless; 0.5 chance, 1.0 perfect)[/dim]"
    )

    sl = md.get("search_layers", [])
    for title, cols, auc in tables:
        if not sl:
            continue
        tbl = Table(title=title)
        tbl.add_column("layer", justify="right")
        for e in cols:
            tbl.add_column(e, justify="right")
        tbl.add_column("mean", justify="right", style="bold")
        import numpy as np
        a = np.array(auc)
        for li_idx, li in enumerate(sl):
            if li_idx >= len(a):
                break
            row = [str(li)]
            for ei in range(len(cols)):
                row.append(f"{a[li_idx, ei]:.3f}")
            row.append(f"{a[li_idx].mean():.3f}")
            tbl.add_row(*row)
        console.print(tbl)

    if show_norms:
        nt = Table(title="L2 vector norms at chosen layers (hidden-state units)")
        nt.add_column("emotion")
        for li in bundle.chosen_layers:
            nt.add_column(f"L{li}", justify="right")
        for e in bundle.emotions:
            row = [e]
            import numpy as np
            for k in range(len(bundle.chosen_layers)):
                row.append(f"{float(np.linalg.norm(bundle.chosen[e][k])):.2f}")
            nt.add_row(*row)
        console.print(nt)


@app.command(name="test-http")
def test_http(
    base_url: str = typer.Option(..., "--base-url", "-u", help="Server root, e.g. http://localhost:8000"),
    api_key: str | None = typer.Option(None, "--api-key", "-k", envvar="EMOTION_STEERING_API_KEY"),
    model: str = typer.Option("model", "--model", help="Model id served by the endpoint."),
    prompt: str = typer.Option(
        "Please continue the sentence: \"Tonight i had a dream about...\"",
        "--prompt",
    ),
    max_tokens: int = typer.Option(120, "--max-tokens"),
    alpha: float = typer.Option(1.5, "--alpha"),
    no_thinking: bool = typer.Option(True, "--no-thinking/--thinking"),
):
    """Hit a running endpoint and print continuations for every emotion."""
    import json as _json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Discover the emotion list from /v1/emotions
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/emotions", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        meta = _json.loads(r.read())
    emotions = meta["emotions"]
    id_map = meta.get("id_map", {e: i for i, e in enumerate(emotions)})

    def call(spec):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if no_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if spec is not None:
            body["vllm_xargs"] = {"steering": spec}
        data = _json.dumps(body).encode()
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        req2 = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req2, timeout=300) as r2:
            d = _json.loads(r2.read())
        return d["choices"][0]["message"]["content"]

    specs = [None] + [[id_map[e], alpha] for e in emotions]
    labels = ["baseline"] + emotions
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        outs = list(ex.map(call, specs))

    for label, text in zip(labels, outs):
        console.rule(f"[bold]{label}[/bold]")
        console.print(text.strip())


@app.command()
def serve(
    vectors: Path = typer.Option(..., "--vectors", "-v", help="Path to a vector bundle directory."),
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model id to serve."),
    backend: str = typer.Option(
        "auto", "--backend", "-b",
        help="auto | hf | vllm. 'auto' picks vllm for Qwen3, hf otherwise.",
    ),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="EMOTION_STEERING_API_KEY"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    max_model_len: int = typer.Option(8192, "--max-model-len"),
    gpu_memory_utilization: float = typer.Option(0.9, "--gpu-memory-utilization"),
    max_num_seqs: int = typer.Option(32, "--max-num-seqs"),
):
    """Serve emotion-steered chat completions on an OpenAI-compatible endpoint."""
    from .vectors import load_bundle

    bundle = load_bundle(vectors)
    chosen_arch = _resolve_backend(backend, bundle.model, model)

    if chosen_arch == "vllm":
        console.print(
            f"[green]using vLLM fast path for {model}[/green] "
            "(continuous batching, ~365 tok/s on L4)"
        )
        from .serve.vllm import serve_vllm
        serve_vllm(
            bundle=bundle,
            model=model,
            host=host, port=port,
            api_key=api_key,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
    else:
        console.print(
            f"[yellow]using HF transformers slow path for {model}[/yellow]\n"
            "  This works for any architecture but does not have continuous batching.\n"
            "  To add a vLLM fast path for this model, see:\n"
            "    .claude/skills/extend-vllm-fast-path.md (in this repo)"
        )
        from .serve.hf import serve_hf
        serve_hf(
            bundle=bundle, model=model,
            host=host, port=port, api_key=api_key, dtype=dtype,
        )


def _resolve_backend(backend: str, bundle_model: str, serve_model: str) -> str:
    if backend == "hf":
        return "hf"
    if backend == "vllm":
        return "vllm"
    if backend != "auto":
        raise typer.BadParameter(f"backend must be auto|hf|vllm, got {backend!r}")
    # auto: vLLM for archs the bundled patches support (Qwen3, Gemma3)
    needle = serve_model.lower()
    if any(k in needle for k in ("qwen3", "qwen-3", "gemma-3", "gemma3", "mistral")):
        try:
            import vllm  # noqa: F401
        except ImportError:
            console.print(
                "[yellow]vllm not installed; install with `pip install emotion-steering[vllm]` "
                "to use the fast path. Falling back to HF.[/yellow]"
            )
            return "hf"
        return "vllm"
    return "hf"


@app.command()
def info():
    """Show package version + supported model architectures (vLLM fast path)."""
    from . import __version__
    console.print(f"emotion-steering version: {__version__}")
    console.print("vLLM fast path supports: Qwen3 (qwen3.py monkey-patch).")
    console.print(
        "All other models use the HF transformers slow path. "
        "See .claude/skills/extend-vllm-fast-path.md to add a new architecture."
    )


if __name__ == "__main__":
    app()
