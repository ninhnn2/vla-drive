"""Where a project keeps its files. A project is one directory (default runs/drive):

    <project>/episodes/<name>/   episode.npz, meta.json, labels.npz     collect, ROS 2 record, label
              models/<version>.pt and .json                             train
              reports/<name>.json                                       evaluate
              lerobot/                    a LeRobotDataset              export
              vla/<version>/              a lerobot-train output dir    (fine-tuning a VLA, outside drive.sh)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT = Path("runs/drive")


@dataclass(frozen=True)
class Project:
    home: Path = DEFAULT

    def __post_init__(self):
        object.__setattr__(self, "home", Path(self.home))      # a string (a ROS parameter) is fine too

    @property
    def episodes(self) -> Path:
        return self.home / "episodes"

    def model(self, version: str) -> Path:
        return self.home / "models" / f"{version}.pt"

    def report(self, name: str) -> Path:
        return self.home / "reports" / f"{name}.json"

    def versions(self) -> list[str]:
        """Trained models, oldest first."""
        models = sorted((self.home / "models").glob("*.pt"), key=lambda path: path.stat().st_mtime)
        return [path.stem for path in models]

    def vla(self, version: str) -> Path:
        """The last checkpoint of a VLA fine-tuned with lerobot-train into vla/<version>."""
        return self.home / "vla" / version / "checkpoints" / "last" / "pretrained_model"

    def checkpoint(self, ref: str) -> Path:
        """A trained model by version (`v1`, or a VLA's `smolvla-v1`), by file, or a LeRobot checkpoint directory."""
        for path in (self.model(str(ref)), Path(ref)):
            if path.is_file():
                return path
        for path in (self.vla(str(ref)), Path(ref)):
            if (path / "config.json").is_file():
                return path
        raise ValueError(f"no model {ref!r}: not a version in {self.home / 'models'} or {self.home / 'vla'}, "
                         "not a file, and not a LeRobot checkpoint directory")

    @staticmethod
    def policy_for(checkpoint: Path) -> str:
        """The policy that runs a checkpoint: a directory is a LeRobot (VLA) checkpoint, a file a CNN."""
        return "vla" if Path(checkpoint).is_dir() else "cnn"


def add_project_argument(parser) -> None:
    parser.add_argument("--project", type=Path, default=DEFAULT, metavar="DIR",
                        help=f"the project directory (default {DEFAULT})")
