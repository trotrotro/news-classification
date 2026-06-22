from pathlib import Path
import subprocess

from news_classifier.data.download_data import (
    download_data,
)


def data_exists(cfg) -> bool:
    paths = [
        Path(cfg.data.train_path),
        Path(cfg.data.val_path),
        Path(cfg.data.test_path),
    ]

    return all(
        path.exists()
        for path in paths
    )


def ensure_data(cfg) -> None:
    if data_exists(cfg):
        return

    try:
        subprocess.run(
            ["dvc", "pull"],
            check=True,
        )

        if data_exists(cfg):
            return

    except subprocess.CalledProcessError:
        pass

    download_data(cfg)