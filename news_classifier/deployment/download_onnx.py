from pathlib import Path

import gdown
from omegaconf import DictConfig


def download_onnx(cfg: DictConfig) -> None:
    onnx_path = Path(cfg.onnx.onnx_path)

    onnx_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading model onnx...")

    gdown.download(
        url=cfg.onnx.url,
        output=str(onnx_path),
        quiet=False,
    )

    print(f"Checkpoint saved to: {onnx_path.resolve()}")
