"""Per-layer one-vs-rest AUC probe for vector quality reporting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def probe_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: str | torch.device = "cuda",
) -> float:
    """LBFGS logistic regression probe (GPU-accelerated). Returns ROC-AUC on val.

    Equivalent to sklearn LogisticRegression(C=1.0) on standardized features,
    up to optimizer tolerance, but ~30x faster at 4096-dim.
    """
    tx = torch.from_numpy(train_x).float().to(device)
    vx = torch.from_numpy(val_x).float().to(device)
    ty = torch.from_numpy(train_y).float().to(device)

    mu = tx.mean(dim=0, keepdim=True)
    sd = tx.std(dim=0, keepdim=True).clamp(min=1e-6)
    tx = (tx - mu) / sd
    vx = (vx - mu) / sd

    n, d = tx.shape
    w = torch.zeros(d, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    weight_decay = 1.0 / n

    opt = torch.optim.LBFGS(
        [w, b], lr=1.0, max_iter=100, tolerance_grad=1e-6,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = tx @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, ty) + 0.5 * weight_decay * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        probs = torch.sigmoid(vx @ w + b).cpu().numpy()
    return float(roc_auc_score(val_y, probs))


def probe_all_layers(
    train_acts: np.ndarray,
    train_labels: list[str],
    val_acts: np.ndarray,
    val_labels: list[str],
    classes: list[str],
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Probe each (layer, class) combination. Returns AUC matrix [n_layers, n_classes]."""
    train_y_all = np.array(train_labels)
    val_y_all = np.array(val_labels)
    n_layers = train_acts.shape[1]
    auc = np.zeros((n_layers, len(classes)), dtype=np.float64)
    for li in range(n_layers):
        for ei, e in enumerate(classes):
            ytr = (train_y_all == e).astype(np.float32)
            yvl = (val_y_all == e).astype(np.float32)
            auc[li, ei] = probe_one(
                train_acts[:, li, :], ytr, val_acts[:, li, :], yvl, device=device
            )
    return auc


def best_window(auc_matrix: np.ndarray, window: int = 3) -> tuple[int, float]:
    """Return (start_index, mean_micro_auc) of the best contiguous layer window."""
    micro = auc_matrix.mean(axis=1)
    if window > len(micro):
        raise ValueError(f"window={window} > n_layers={len(micro)}")
    means = np.array([micro[i : i + window].mean() for i in range(len(micro) - window + 1)])
    start = int(np.argmax(means))
    return start, float(means[start])
