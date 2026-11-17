"""A small convolutional policy: one camera frame and the car's state in, a plan out."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

SPEED_SCALE = 3.0           # m/s that maps to 1.0 at the network's input


class ChunkNet(nn.Module):
    """About 0.3 M parameters, in the shape of NVIDIA's PilotNet.

    Five strided convolutions turn the image into a feature vector; the state joins it;
    an MLP emits `steps` x `[steer, throttle]`, squashed to the action range by tanh.
    """

    def __init__(self, steps: int, image_hw=(96, 128)):
        super().__init__()
        self.steps, self.image_hw = steps, tuple(image_hw)
        widths = [3, 24, 36, 48, 64, 64]
        self.vision = nn.Sequential(*[layer for a, b in zip(widths, widths[1:])
                                      for layer in (nn.Conv2d(a, b, 3, stride=2, padding=1), nn.ReLU())],
                                    nn.Flatten())
        with torch.no_grad():
            features = self.vision(torch.zeros(1, 3, *self.image_hw)).shape[1]
        self.head = nn.Sequential(nn.Linear(features + 2, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                  nn.Linear(128, steps * 2), nn.Tanh())

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """images: uint8 or float (B, H, W, 3) in 0..255; states: (B, 2) raw. -> (B, steps, 2)"""
        # float before permute: on MPS (torch 2.5) casting a permuted uint8 view that starts
        # inside its storage, such as batch[512:1024], ignores the offset and reads frame 0 on.
        x = images.float().permute(0, 3, 1, 2) / 255.0 - 0.5
        states = states.float() * states.new_tensor([1.0 / SPEED_SCALE, 1.0])
        return self.head(torch.cat([self.vision(x), states], dim=1)).view(-1, self.steps, 2)


def save(path, model: ChunkNet, plan_dt: float, report: dict) -> None:
    torch.save({"weights": model.state_dict(), "steps": model.steps, "image_hw": list(model.image_hw),
                "plan_dt": plan_dt, "report": report}, path)


def load(path, device: str = "cpu") -> tuple[ChunkNet, dict]:
    """(model in eval mode, the checkpoint's metadata)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ChunkNet(checkpoint["steps"], checkpoint["image_hw"]).to(device)
    model.load_state_dict(checkpoint.pop("weights"))
    return model.eval(), checkpoint


def predict(model: ChunkNet, image: np.ndarray, speed: float, steer: float) -> np.ndarray:
    """One already-shrunk frame -> (steps, 2) actions."""
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.from_numpy(image).to(device)[None], torch.tensor([[speed, steer]], device=device))
    return out[0].cpu().numpy().astype(np.float64)
