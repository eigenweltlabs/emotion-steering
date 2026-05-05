"""Vector bundle: load/save/inspect a directory of emotion vectors + metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VectorBundle:
    """A complete artifact directory: per-emotion .npy files + metadata.json."""

    path: Path
    metadata: dict
    chosen: dict[str, np.ndarray]      # emotion -> [n_chosen_layers, hidden]
    full_sweep: dict[str, np.ndarray] | None  # emotion -> [n_search_layers, hidden]

    @property
    def emotions(self) -> list[str]:
        return list(self.metadata["emotions"])

    @property
    def chosen_layers(self) -> list[int]:
        return list(self.metadata["chosen_layers"])

    @property
    def model(self) -> str:
        return str(self.metadata["model"])

    @property
    def hidden(self) -> int:
        return int(self.metadata["hidden"])

    def stack_chosen(self, dtype=np.float32) -> np.ndarray:
        """[n_emotions, n_chosen_layers, hidden] tensor in fixed emotion order."""
        return np.stack([self.chosen[e].astype(dtype) for e in self.emotions], axis=0)


def save_bundle(
    out_dir: str | Path,
    chosen: dict[str, np.ndarray],
    full_sweep: dict[str, np.ndarray] | None,
    metadata: dict,
) -> Path:
    """Write {emotion}_chosen.npy, {emotion}_full_sweep.npy, metadata.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for e, arr in chosen.items():
        np.save(out_dir / f"{e}_chosen.npy", arr.astype(np.float32))
    if full_sweep is not None:
        for e, arr in full_sweep.items():
            np.save(out_dir / f"{e}_full_sweep.npy", arr.astype(np.float32))

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


def load_bundle(path: str | Path) -> VectorBundle:
    path = Path(path)
    md_path = path / "metadata.json"
    if not md_path.exists():
        raise FileNotFoundError(f"no metadata.json at {md_path}")
    metadata = json.loads(md_path.read_text())

    emotions = metadata["emotions"]
    chosen: dict[str, np.ndarray] = {}
    full_sweep: dict[str, np.ndarray] = {}
    for e in emotions:
        c_path = path / f"{e}_chosen.npy"
        if not c_path.exists():
            raise FileNotFoundError(f"missing {c_path}")
        chosen[e] = np.load(c_path)
        f_path = path / f"{e}_full_sweep.npy"
        if f_path.exists():
            full_sweep[e] = np.load(f_path)

    return VectorBundle(
        path=path,
        metadata=metadata,
        chosen=chosen,
        full_sweep=full_sweep or None,
    )
