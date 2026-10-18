"""Late-fusion image/audio network. Weights are always explicit; no downloads."""
from __future__ import annotations


def build_model(pretrained_visual: bool = False):
    from torch import nn
    from torchvision.models import resnet18

    # pretrained_visual defaults to False so the model can be initialized offline.
    weights = "DEFAULT" if pretrained_visual else None
    visual = resnet18(weights=weights)
    visual_dim = visual.fc.in_features
    visual.fc = nn.Identity()

    audio = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 128), nn.ReLU(inplace=True),
    )
    class AudioVisualDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = visual
            self.audio = audio
            self.visual_head = nn.Linear(visual_dim, 1)
            self.audio_head = nn.Linear(128, 1)
            self.fusion_head = nn.Sequential(nn.Linear(visual_dim + 129, 128), nn.ReLU(inplace=True),
                                             nn.Dropout(.25), nn.Linear(128, 1))

        def forward(self, image, log_mel, audio_present):
            import torch
            visual_features = self.visual(image)
            audio_features = self.audio(log_mel) * audio_present.reshape(-1, 1)
            fused = torch.cat([visual_features, audio_features, audio_present.reshape(-1, 1)], dim=1)
            return {
                "visual_logits": self.visual_head(visual_features).squeeze(1),
                "audio_logits": self.audio_head(audio_features).squeeze(1),
                "fused_logits": self.fusion_head(fused).squeeze(1),
            }

    return AudioVisualDetector()
