import os

import lightning as pl
import matplotlib.pyplot as plt


class PlotCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        os.makedirs("plots", exist_ok=True)

        plt.figure()
        plt.plot(trainer.callback_metrics["train_loss"])
        plt.plot(trainer.callback_metrics["val_loss"])
        plt.title("Loss")
        plt.savefig("plots/loss.png")
