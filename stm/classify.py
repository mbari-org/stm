# stm, Apache-2.0 license
# Filename: classify.py
# Description: Classify sounds using Perch2 embeddings
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Perch2 constants - these match the exported ONNX graph -- see perch_v2.onnx)
PERCH2_EMBEDDING_DIM = 1536
PERCH2_SAMPLE_RATE = 32000
PERCH2_WINDOW_SECONDS = 5.0  # fixed input window baked into the ONNX graph
PERCH2_WINDOW_SAMPLES = int(PERCH2_SAMPLE_RATE * PERCH2_WINDOW_SECONDS)  # 160000
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEAK_NEG_WEIGHT = 0.05
EMBEDDING_CACHE_METADATA = "embedding_cache_metadata.json"


def bce_loss(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    is_labeled_mask: torch.Tensor,
    weak_neg_weight: float,
) -> torch.Tensor:
    """Binary cross entropy loss from logits with weak negative weights."""
    y_true = y_true.to(dtype=logits.dtype)
    log_p = F.logsigmoid(logits)
    log_not_p = F.logsigmoid(-logits)
    raw_bce = -y_true * log_p - (1.0 - y_true) * log_not_p
    is_labeled_mask = is_labeled_mask.to(dtype=logits.dtype)
    weights = (1.0 - is_labeled_mask) * weak_neg_weight + is_labeled_mask
    return torch.mean(raw_bce * weights)


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int = PERCH2_EMBEDDING_DIM, num_classes: int = 1):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.dense(embeddings)


@dataclass
class LinearModel:
    model: LinearProbe
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[..., torch.Tensor] = bce_loss
    classes: list[str] = field(default_factory=list)
    weak_neg_weight: float = DEFAULT_WEAK_NEG_WEIGHT

    def train_step(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        is_labeled_mask: torch.Tensor | None = None,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(embeddings)
        if is_labeled_mask is None:
            is_labeled_mask = torch.ones_like(targets)
        loss = self.loss_fn(targets, logits, is_labeled_mask, self.weak_neg_weight)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return torch.sigmoid(self.model(embeddings))

    @torch.no_grad()
    def predict_labels(self, embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
        """Argmax class index for each row of Perch embeddings."""
        x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (N, D), got {tuple(x.shape)}")
        in_features = self.model.dense.in_features
        if x.shape[1] != in_features:
            raise ValueError(
                f"linear probe expects D={in_features} Perch embeddings, got D={x.shape[1]}. "
                "Use a FeatureBlock from extract() on a single TimeGrid "
                "(not pooled or combined mean+std features)."
            )
        device = next(self.model.parameters()).device
        return self.predict_proba(x.to(device)).argmax(dim=1).cpu().numpy()

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> LinearModel:
        """Load a checkpoint written by :meth:`save`."""
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload["state_dict"]
        num_classes, embedding_dim = state["dense.weight"].shape
        loaded = build_model(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            device=device,
        )
        loaded.model.load_state_dict(state)
        loaded.classes = [str(c) for c in payload.get("classes", [])]
        print(f"Loaded {path} classes={loaded.classes} D={embedding_dim}", flush=True)
        return loaded

    def save(
        self,
        path: str | Path,
        classes: Sequence[str],
        preprocess: dict | None = None,
    ) -> None:
        """Write a torch checkpoint: ``state_dict``, ``classes``, optional ``preprocess``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": self.model.state_dict(), "classes": list(classes)}
        if preprocess is not None:
            payload["preprocess"] = dict(preprocess)
        print(f"Saving model to {path}")
        torch.save(payload, path)


def build_model(
    num_classes: int,
    embedding_dim: int = PERCH2_EMBEDDING_DIM,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str = "cpu",
    weak_neg_weight: float = DEFAULT_WEAK_NEG_WEIGHT,
) -> LinearModel:
    model = LinearProbe(embedding_dim=embedding_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return LinearModel(
        model=model,
        optimizer=optimizer,
        loss_fn=bce_loss,
        weak_neg_weight=weak_neg_weight,
    )
