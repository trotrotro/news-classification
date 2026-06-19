import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class PlotCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):

        import os

        import matplotlib.pyplot as plt

        os.makedirs("plots", exist_ok=True)

        plt.figure()
        plt.plot([x.item() for x in pl_module.train_losses], label="train_loss")
        plt.plot([x.item() for x in pl_module.val_losses], label="val_loss")
        plt.legend()
        plt.title("Loss")
        plt.savefig("plots/loss.png")
