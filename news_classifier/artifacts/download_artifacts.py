import shutil
import zipfile
from pathlib import Path

import gdown
from omegaconf import DictConfig


def download_artifacts(cfg: DictConfig) -> None:
    output_dir = Path(cfg.artifacts.artifacts_path)
    zip_path = Path("artifacts.zip")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdown.download(
        cfg.artifacts.url,
        str(zip_path),
        quiet=False,
    )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    macosx_path = output_dir / "__MACOSX"

    if macosx_path.exists():
        shutil.rmtree(macosx_path)

    zip_path.unlink()
