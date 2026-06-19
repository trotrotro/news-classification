import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scripts.modules.dataset import NewsDataset
from scripts.modules.preprocess import preprocessing
from scripts.modules.vocabulary import build_vocab


class NewsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_path = cfg.data.train_path
        self.val_path = cfg.data.val_path
        self.test_path = cfg.data.test_path

        self.batch_size = cfg.train.batch_size
        self.max_length = cfg.train.max_length
        self.min_freq = cfg.preprocess.min_freq

        self.artifacts_dir = getattr(cfg.paths, "artifacts_dir", "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def setup(self, stage=None):

        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)

        for df in [self.train_df, self.val_df, self.test_df]:
            df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
            df.rename(columns={"rubric": "label"}, inplace=True)
            df["text"] = df["text"].apply(preprocessing)

        labels = sorted(self.train_df["label"].unique())

        self.label2idx = {label: i for i, label in enumerate(labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}

        # save mapping (CRITICAL)
        with open(os.path.join(self.artifacts_dir, "label2idx.json"), "w") as f:
            json.dump(self.label2idx, f, ensure_ascii=False)

        with open(os.path.join(self.artifacts_dir, "idx2label.json"), "w") as f:
            json.dump(self.idx2label, f, ensure_ascii=False)

        # apply mapping
        for df in [self.train_df, self.val_df, self.test_df]:
            df["label"] = df["label"].map(self.label2idx)

        assert not self.train_df["label"].isna().any()

        self.word2idx, self.idx2word = build_vocab(
            self.train_df["text"], min_freq=self.min_freq
        )

        # save vocab (CRITICAL)
        with open(os.path.join(self.artifacts_dir, "word2idx.json"), "w") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

        self.train_dataset = NewsDataset(self.train_df, self.word2idx, self.max_length)
        self.val_dataset = NewsDataset(self.val_df, self.word2idx, self.max_length)
        self.test_dataset = NewsDataset(self.test_df, self.word2idx, self.max_length)

        self.vocab_size = len(self.word2idx)
        self.num_classes = len(self.label2idx)

        class_counts = self.train_df["label"].value_counts().sort_index()
        weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
        weights = weights / weights.sum() * len(class_counts)

        self.class_weights = weights

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
