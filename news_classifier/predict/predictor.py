import json
from pathlib import Path

import pandas as pd
import torch

from news_classifier.data.data_module import NewsDataModule
from news_classifier.models.lstm import LSTMClassifier
from news_classifier.preprocessing.preprocess import preprocessing


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        artifacts_dir = Path(cfg.artifacts.artifacts_path)

        with open(
            artifacts_dir / "word2idx.json",
            encoding="utf-8",
        ) as file:
            self.word2idx = json.load(file)

        with open(
            artifacts_dir / "idx2label.json",
            encoding="utf-8",
        ) as file:
            self.idx2label = json.load(file)

        checkpoint_path = cfg.checkpoint.checkpoint_path

        dm = NewsDataModule(cfg)
        dm.setup()

        self.model = LSTMClassifier.load_from_checkpoint(
            checkpoint_path,
            class_weights=dm.class_weights,
        )

        self.model.to(self.device)
        self.model.eval()

        self.max_length = cfg.data.max_length

    def text_to_tensor(
        self,
        text: str,
    ) -> torch.Tensor:
        text = preprocessing(text)

        token_ids = []

        for token in text.split():
            token_ids.append(
                self.word2idx.get(
                    token,
                    self.word2idx["<UNK>"],
                )
            )

        token_ids = token_ids[: self.max_length]

        padding_length = self.max_length - len(token_ids)

        if padding_length > 0:
            token_ids.extend([self.word2idx["<PAD>"]] * padding_length)

        tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_text(
        self,
        text: str,
    ) -> str:
        inputs = self.text_to_tensor(text)

        logits = self.model(inputs)

        prediction_idx = logits.argmax(dim=1).cpu().item()

        return self.idx2label[str(prediction_idx)]

    def predict_file(
        self,
        csv_path: str,
    ) -> pd.DataFrame:
        dataframe = pd.read_csv(csv_path)

        predictions = []

        for text in dataframe["text"]:
            prediction = self.predict_text(text)
            predictions.append(prediction)

        dataframe["prediction"] = predictions

        return dataframe
