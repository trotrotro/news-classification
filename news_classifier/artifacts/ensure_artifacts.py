import subprocess
from pathlib import Path

from news_classifier.artifacts.download_artifacts import (
    download_artifacts,
)


def artifacts_exist(cfg) -> bool:
    artifacts_dir = Path(cfg.artifacts.artifacts_path)

    files = [
        artifacts_dir / "word2idx.json",
        artifacts_dir / "idx2label.json",
        artifacts_dir / "label2idx.json",
    ]

    return all(file.exists() for file in files)


def ensure_artifacts(cfg) -> None:
    if artifacts_exist(cfg):
        return

    try:
        subprocess.run(
            ["dvc", "pull"],
            check=True,
        )

        if artifacts_exist(cfg):
            return

    except subprocess.CalledProcessError:
        pass

    download_artifacts(cfg)
