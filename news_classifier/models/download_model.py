import shutil
import zipfile
from pathlib import Path

import gdown
from omegaconf import DictConfig


def download_model(cfg: DictConfig) -> None:
    checkpoint_path = Path(cfg.model.checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = Path("model.zip")

    gdown.download(
        cfg.model.url,
        str(zip_path),
        quiet=False,
    )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(checkpoint_path.parent)

    macosx_path = checkpoint_path.parent / "__MACOSX"

    if macosx_path.exists():
        shutil.rmtree(macosx_path)

    zip_path.unlink()
