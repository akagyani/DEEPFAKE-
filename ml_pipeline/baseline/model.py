"""Small, replaceable frame-classification baseline used by train.py and infer.py."""
from __future__ import annotations


def build_model():
    """Create a ResNet-18 classifier from scratch; no weights are downloaded."""
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
