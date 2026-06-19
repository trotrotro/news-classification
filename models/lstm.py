import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class LSTMClassifier(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, embed_dim=128, hidden_dim=256, lr=1e-3):
        super().__init__()

        self.lr = lr
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.acc = MulticlassAccuracy(num_classes=num_classes)

        self.save_hyperparameters(ignore=["loss_fn"])

        # Убираем списки для накопления метрик - они не нужны

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Логируем loss с правильными параметрами
        self.log(
            "train_loss",
            loss,
            on_step=False,  # Не логируем на каждом шаге
            on_epoch=True,  # Логируем в конце эпохи
            prog_bar=True,  # Показываем в прогресс-баре
            logger=True,  # Отправляем в логгер (MLflow)
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Вычисляем метрики
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)

        # Логируем все метрики
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        # Опционально: логируем дополнительные метрики в конце эпохи
        # Например, можно посчитать средний loss за эпоху
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        return optimizer
