import subprocess
from pathlib import Path

from news_classifier.models.download_model import (
    download_model,
)


def ensure_model(cfg) -> None:
    checkpoint = Path(cfg.checkpoint.checkpoint_path)

    if checkpoint.exists():
        return

    try:
        subprocess.run(
            ["dvc", "pull"],
            check=True,
        )

        if checkpoint.exists():
            return

    except subprocess.CalledProcessError:
        pass

    download_model(cfg)
