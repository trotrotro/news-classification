from pathlib import Path
import shutil
import zipfile

import gdown
import hydra
from omegaconf import DictConfig


def download_data(cfg: DictConfig) -> None:
    gdrive_url = cfg.data_load.url
    output_dir = Path(cfg.data_load.data_path)
    zip_path = Path("rus_news_data.zip")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading zip file...")
    gdown.download(gdrive_url, str(zip_path), quiet=False)

    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    macosx_path = output_dir / "__MACOSX"
    if macosx_path.exists():
        print("Removing __MACOSX folder...")
        shutil.rmtree(macosx_path)

    print("Deleting zip file...")
    zip_path.unlink()

    print(f"Extracted contents saved to: {output_dir}")


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    download_data(cfg)


if __name__ == "__main__":
    main()