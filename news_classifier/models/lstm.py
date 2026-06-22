import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        num_classes,
        class_weights=None,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        dropout=0.3,
        lr=1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["class_weights"])
        self.lr = lr

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )
        self.embedding_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2

        self.attention_gate = nn.Linear(
            lstm_out_dim,
            1,
        )

        self.fc = nn.Linear(
            lstm_out_dim,
            num_classes,
        )

        self.dropout = nn.Dropout(0.4)

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)

        x = self.embedding(x)
        x = self.embedding_dropout(x)

        outputs, _ = self.lstm(x)

        weights = torch.sigmoid(self.attention_gate(outputs))
        weights = weights * mask

        context = (outputs * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)

        context = self.dropout(context)

        logits = self.fc(context)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)

        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
            },
        }
