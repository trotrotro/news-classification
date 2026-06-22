from pathlib import Path

import gdown
from omegaconf import DictConfig


def download_model(cfg: DictConfig) -> None:
    checkpoint_path = Path(cfg.checkpoint.checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading model checkpoint...")

    gdown.download(
        url=cfg.checkpoint.url,
        output=str(checkpoint_path),
        quiet=False,
    )

    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")
