"""Download CSV dataset from Google Drive.

Usage:
    uv run python scripts/download.py download --file_id 1oZzxQbRRvkp_UQZIoypKDamWaHBljbEw --destination_path ./data
    uv run python scripts/download.py download --file_id 1oZzxQbRRvkp_UQZIoypKDamWaHBljbEw --destination_path ./data --filename news_ru.csv
"""

from __future__ import annotations

import os

import fire
import pandas as pd
import requests
from googledrivedownloader import download_file_from_google_drive as gdd


class NewsDownloader:
    """CLI for downloading CSV dataset from Google Drive."""

    def download(
        self,
        file_id: str = "1oZzxQbRRvkp_UQZIoypKDamWaHBljbEw",
        destination_path: str = "./data",
        filename: str = "news_ru.csv",
        extract: bool = True,
    ):
        """Download CSV dataset from Google Drive.

        Args:
            file_id: Google Drive file ID from the URL.
            destination_path: Directory to save the downloaded file.
            filename: Name to save the file as.
            extract: If True, reads the CSV and shows basic info.
        """
        os.makedirs(destination_path, exist_ok=True)

        file_path = os.path.join(destination_path, filename)

        # Method 1: Using google-drive-downloader (recommended)
        try:
            print(f"Downloading from Google Drive file ID: {file_id}")
            gdd(file_id=file_id, dest_path=file_path, overwrite=True)
            print(f"Download complete. File saved to '{file_path}'.")
        except ImportError:
            print("google-drive-downloader not installed. Using alternative method...")
            # Method 2: Using direct download URL
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            session = requests.Session()
            response = session.get(url, stream=True)

            # Get the confirmation token if needed
            if "confirm" in response.cookies:
                params = {"confirm": response.cookies["confirm"]}
                response = session.get(url, params=params, stream=True)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Download complete. File saved to '{file_path}'.")

        # Load and display CSV information
        if extract and filename.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                print("\nDataset Info:")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("\nFirst 5 rows:")
                print(df.head())
                print("\nBasic statistics:")
                print(df.describe())
                return df
            except Exception as e:
                print(f"Could not read CSV: {e}")
                return None

    def load_csv(self, file_path: str, show_info: bool = True, sample_rows: int = 5):
        """Load and inspect an existing CSV file.

        Args:
            file_path: Path to the CSV file.
            show_info: Whether to display dataset information.
            sample_rows: Number of rows to display in sample.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        if show_info:
            print(f"Dataset loaded from: {file_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst {sample_rows} rows:")
            print(df.head(sample_rows))
            print("\nData types:")
            print(df.dtypes)
            print("\nMissing values:")
            print(df.isnull().sum())

        return df


if __name__ == "__main__":
    fire.Fire(NewsDownloader)
