# stm, Apache-2.0 license
# Filename: train_utils.py
# Description: parses annotation file created by the Raven software
import torch
import torch.nn.functional as F

def labels_to_one_hot(labels):
    """Convert string labels to a one-hot float tensor.
    Returns ``(one_hot_labels, class_to_idx)``.
    """
    unique_labels = sorted(set(labels))
    class_to_idx = {label: i for i, label in enumerate(unique_labels)}

    missing = set(labels) - set(class_to_idx)
    if missing:
        raise KeyError(f"Labels not in class_to_idx: {sorted(missing)}")

    indices = torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long)
    one_hot = F.one_hot(indices, num_classes=len(class_to_idx)).float()
    return one_hot, class_to_idx