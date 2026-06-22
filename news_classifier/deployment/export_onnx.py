from pathlib import Path

import hydra
import torch

from news_classifier.data.data_module import NewsDataModule
from news_classifier.models.lstm import LSTMClassifier


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="config",
)
def main(cfg):
    dm = NewsDataModule(cfg)
    dm.setup()

    model = LSTMClassifier.load_from_checkpoint(
        cfg.checkpoint.checkpoint_path,
        class_weights=dm.class_weights,
    )

    model.to("cpu")
    model.eval()

    dummy_input = torch.randint(
        low=0,
        high=dm.vocab_size,
        size=(1, cfg.data.max_length),
        dtype=torch.long,
        device="cpu",
    )

    output_path = Path("models/model.onnx")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
            },
            "logits": {
                0: "batch_size",
            },
        },
    )

    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    main()
