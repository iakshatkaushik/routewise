"""Neural network pieces (syllabus unit: Artificial Neural Networks).

`train_mlp` trains a feed-forward network with PyTorch on the GPU (CPU fallback).
`NumpyMLP` holds the trained weights and runs the forward pass with NumPy only, so the web
service does not need PyTorch installed.
"""
from __future__ import annotations

import numpy as np


class NumpyMLP:
    """Inference-only MLP: Linear -> activation -> ... -> Linear -> sigmoid."""

    def __init__(self, weights, biases, activation="relu", mean=None, std=None):
        self.weights = [w.astype(np.float32) for w in weights]
        self.biases = [b.astype(np.float32) for b in biases]
        self.activation = activation
        self.mean = None if mean is None else mean.astype(np.float32)
        self.std = None if std is None else std.astype(np.float32)

    def _act(self, h):
        if self.activation == "relu":
            return np.maximum(h, 0)
        if self.activation == "tanh":
            return np.tanh(h)
        if self.activation == "gelu":
            return 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
        return 1 / (1 + np.exp(-h))

    def predict_proba(self, X):
        h = np.asarray(X, dtype=np.float32)
        if self.mean is not None:
            h = (h - self.mean) / self.std
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = self._act(h)
        p = 1 / (1 + np.exp(-np.clip(h[:, 0], -30, 30)))
        return np.column_stack([1 - p, p])


def torch_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_mlp(X, y, X_val, y_val, hidden=(512,), dropout=0.2, lr=1e-3, weight_decay=1e-4,
              activation="relu", max_epochs=40, patience=5, batch_size=512, seed=42, fixed_epochs=None):
    """Train with early stopping on validation ROC-AUC. Returns (NumpyMLP, history, best_epoch).

    With `fixed_epochs`, trains for exactly that many epochs (used to refit on all data).
    """
    import torch
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch_device()
    mean, std = X.mean(0), X.std(0) + 1e-6
    norm = lambda A: torch.tensor((A - mean) / std, dtype=torch.float32, device=dev)  # noqa: E731
    Xt, yt = norm(X), torch.tensor(y, dtype=torch.float32, device=dev)
    Xv = norm(X_val) if X_val is not None else None
    acts = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "gelu": torch.nn.GELU, "logistic": torch.nn.Sigmoid}

    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), acts[activation](), torch.nn.Dropout(dropout)]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    net = torch.nn.Sequential(*layers).to(dev)
    pos_weight = torch.tensor([float((1 - y.mean()) / y.mean())], dtype=torch.float32, device=dev)  # balance the 13.5% minority class
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = fixed_epochs or max_epochs
    # same learning-rate schedule in both modes, so a refit for `fixed_epochs` matches the
    # early-stopped run up to that epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    def snapshot():
        lin = [m for m in net if isinstance(m, torch.nn.Linear)]
        return NumpyMLP([m.weight.detach().cpu().numpy().T for m in lin],
                        [m.bias.detach().cpu().numpy() for m in lin],
                        "logistic" if activation == "logistic" else activation, mean, std)

    best, best_auc, best_epoch, bad, history = None, -1.0, 0, 0, []
    n = len(Xt)
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if fixed_epochs:
            continue
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv).squeeze(1)).cpu().numpy()
        auc = roc_auc_score(y_val, pv)
        history.append(round(float(auc), 4))
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, bad, best = auc, epoch, 0, snapshot()
        else:
            bad += 1
            if bad >= patience:
                break
    if fixed_epochs:
        return snapshot(), history, fixed_epochs
    return best, history, best_epoch
