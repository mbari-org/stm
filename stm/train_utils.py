# stm, Apache-2.0 license
# Filename: train_utils.py
# Description: Label encoding and feature packing for linear probes
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def labels_to_one_hot(labels):
    """Convert string labels to a one-hot float tensor.
    Returns ``(one_hot_labels, class_to_idx)``.
    """
    unique_labels = sorted(set(labels))
    class_to_idx = {label: i for i, label in enumerate(unique_labels)}
    return one_hot_with_vocab(labels, unique_labels), class_to_idx


def one_hot_with_vocab(
    values: Sequence[str],
    options: Sequence[str],
) -> torch.Tensor:
    """One-hot encode *values* using a fixed *options* vocabulary."""
    if not options:
        raise ValueError("options must be non-empty")
    if len(set(options)) != len(options):
        raise ValueError(f"duplicate options: {list(options)}")
    class_to_idx = {label: i for i, label in enumerate(options)}
    missing = set(values) - set(class_to_idx)
    if missing:
        raise KeyError(f"Labels not in options: {sorted(missing)}")
    indices = torch.tensor([class_to_idx[label] for label in values], dtype=torch.long)
    return F.one_hot(indices, num_classes=len(options)).float()


def packed_feature_dim(
    embedding_dim: int,
    vocabs: Mapping[str, Sequence[str]],
) -> int:
    """Width of :func:`pack_probe_features` given embedding size and one-hot vocabs."""
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
    return int(embedding_dim) + sum(len(options) for options in vocabs.values())


def pack_probe_features(
    embeddings: np.ndarray | torch.Tensor,
    categoricals: Mapping[str, Sequence[str]],
    vocabs: Mapping[str, Sequence[str]],
) -> torch.Tensor:
    """Concatenate a foundation embedding with one-hot categorical inputs.

    *vocabs* insertion order is the packed feature order after the embedding.
    Pass an empty *vocabs* mapping to use the embeddings alone.
    """
    x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D), got {tuple(x.shape)}")
    parts = [x]
    n = x.shape[0]
    for name, options in vocabs.items():
        if name not in categoricals:
            raise KeyError(f"missing categorical column {name!r}")
        values = list(categoricals[name])
        if len(values) != n:
            raise ValueError(
                f"{name} has {len(values)} values, expected {n} to match embeddings"
            )
        parts.append(one_hot_with_vocab(values, options))
    return torch.cat(parts, dim=1)


def encode_multihead_targets(
    labels: Mapping[str, Sequence[str]],
    category_options: Mapping[str, Sequence[str]],
) -> torch.Tensor:
    """Integer targets of shape ``(N, n_categories)`` in *category_options* order."""
    if not category_options:
        raise ValueError("category_options must be non-empty")
    columns: list[torch.Tensor] = []
    n: int | None = None
    for name, options in category_options.items():
        if name not in labels:
            raise KeyError(f"missing label column {name!r}")
        values = list(labels[name])
        if n is None:
            n = len(values)
        elif len(values) != n:
            raise ValueError(f"{name} has {len(values)} labels, expected {n}")
        idx = {opt: i for i, opt in enumerate(options)}
        if len(idx) != len(options):
            raise ValueError(f"duplicate options in {name!r}: {list(options)}")
        missing = set(values) - set(idx)
        if missing:
            raise KeyError(f"{name} labels not in options: {sorted(missing)}")
        columns.append(torch.tensor([idx[v] for v in values], dtype=torch.long))
    return torch.stack(columns, dim=1)
