import gzip
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    url = config["embeddings"]["url"]
    output_dir = Path(config["embeddings"]["dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    gz_path = output_dir / "cc.ru.300.vec.gz"
    vec_path = output_dir / "cc.ru.300.vec"

    if vec_path.exists():
        print("Embeddings already downloaded.")
        return

    print("Downloading embeddings...")
    urlretrieve(url, gz_path)

    print("Extracting embeddings...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(vec_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)

    print(f"Embeddings saved to: {vec_path}")


if __name__ == "__main__":
    main()
