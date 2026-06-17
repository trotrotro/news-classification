"""Download CSV dataset from Google Drive and split into train/test/val.

Usage:
    uv run python scripts/download.py download --file_id 1oZzxQbRRvkp_UQZIoypKDamWaHBljbEw --destination_path ./data
    uv run python scripts/download.py download --file_id 1oZzxQbRRvkp_UQZIoypKDamWaHBljbEw --destination_path ./data --filename news_ru.csv
    uv run python scripts/download.py split --file_path ./data/news_ru.csv --output_dir ./data/news_splitted
"""

from __future__ import annotations

import os

import fire
import pandas as pd
import requests
from googledrivedownloader import download_file_from_google_drive as gdd
from sklearn.model_selection import train_test_split


class NewsDownloader:
    """CLI for downloading CSV dataset from Google Drive and splitting it."""

    def download(
        self,
        file_id: str = "1oZzxQbRRvkp_UQZIoypKDamWaHBljbEw",
        destination_path: str = "./data",
        filename: str = "news_ru.csv",
        extract: bool = True,
        split: bool = True,
        split_ratio: tuple = (0.7, 0.15, 0.15),
        random_state: int = 42,
    ):
        """Download CSV dataset from Google Drive and optionally split it.

        Args:
            file_id: Google Drive file ID from the URL.
            destination_path: Directory to save the downloaded file.
            filename: Name to save the file as.
            extract: If True, reads the CSV and shows basic info.
            split: If True, splits the dataset into train/test/val.
            split_ratio: Tuple of (train_ratio, test_ratio, val_ratio).
            random_state: Random seed for reproducibility.
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

                # Split the dataset if requested
                if split:
                    self.split_data(
                        file_path=file_path,
                        output_dir=os.path.join(destination_path, "news_splitted"),
                        split_ratio=split_ratio,
                        random_state=random_state,
                    )

                return df
            except Exception as e:
                print(f"Could not read CSV: {e}")
                return None

    def split_data(
        self,
        file_path: str,
        output_dir: str = "./data/news_splitted",
        split_ratio: tuple = (0.7, 0.15, 0.15),
        random_state: int = 42,
    ):
        """Split CSV dataset into train/test/val with stratification by rubric.

        Args:
            file_path: Path to the CSV file.
            output_dir: Directory to save the split files.
            split_ratio: Tuple of (train_ratio, test_ratio, val_ratio).
            random_state: Random seed for reproducibility.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if "rubric" not in df.columns:
            raise ValueError("Dataset must contain 'rubric' column for stratification")

        if "text" not in df.columns:
            raise ValueError("Dataset must contain 'text' column")

        print(f"\nSplitting dataset with {len(df)} rows...")
        print(
            f"Split ratio: train={split_ratio[0]}, test={split_ratio[1]}, val={split_ratio[2]}"
        )

        # First split: separate train and temp (test + val)
        train_ratio, test_ratio, val_ratio = split_ratio
        temp_ratio = test_ratio + val_ratio

        # Adjust ratios for the second split
        test_val_ratio = test_ratio / temp_ratio  # proportion of test in temp

        # First split: train vs temp (test+val)
        train_df, temp_df = train_test_split(
            df, test_size=temp_ratio, stratify=df["rubric"], random_state=random_state
        )

        # Second split: test vs val from temp
        test_df, val_df = train_test_split(
            temp_df,
            test_size=1 - test_val_ratio,  # val ratio in temp
            stratify=temp_df["rubric"],
            random_state=random_state,
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save the split datasets
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        val_path = os.path.join(output_dir, "val.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"\nSplit complete! Files saved to '{output_dir}':")
        print(
            f"  - train.csv: {len(train_df)} rows ({len(train_df) / len(df) * 100:.1f}%)"
        )
        print(
            f"  - test.csv: {len(test_df)} rows ({len(test_df) / len(df) * 100:.1f}%)"
        )
        print(f"  - val.csv: {len(val_df)} rows ({len(val_df) / len(df) * 100:.1f}%)")

        # Display rubric distribution for each split
        print("\nRubric distribution:")
        print("\nTrain set:")
        print(train_df["rubric"].value_counts())
        print("\nTest set:")
        print(test_df["rubric"].value_counts())
        print("\nValidation set:")
        print(val_df["rubric"].value_counts())

        return train_df, test_df, val_df

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
