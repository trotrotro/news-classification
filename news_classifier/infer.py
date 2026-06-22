from pathlib import Path

import hydra
from omegaconf import DictConfig

from news_classifier.artifacts.ensure_artifacts import (
    ensure_artifacts,
)
from news_classifier.data.ensure_data import (
    ensure_data,
)
from news_classifier.models.ensure_model import (
    ensure_model,
)
from news_classifier.predict.predictor import Predictor


def predict_single_text(
    predictor: Predictor,
    text: str,
) -> None:
    prediction = predictor.predict_text(text)

    print()
    print(f"Prediction: {prediction}")


def predict_csv(
    predictor: Predictor,
    input_path: str,
    output_path: str,
) -> None:
    predictions = predictor.predict_file(input_path)
    print(predictions["prediction"].value_counts())

    preview = predictions.copy()

    preview["text"] = preview["text"].str.slice(0, 80).str.replace("\n", " ")

    print("\nFirst predictions:")
    print(preview[["text", "prediction"]].head(5))

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print(f"Predictions saved to {output_path.resolve()}")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    ensure_data(cfg)
    ensure_artifacts(cfg)
    ensure_model(cfg)

    predictor = Predictor(cfg)

    if cfg.infer.interactive:
        text = input("Введите текст новости:\n")
        predict_single_text(predictor, text)
        return

    if cfg.infer.text is not None:
        predict_single_text(predictor, cfg.infer.text)
        return

    predict_csv(
        predictor,
        cfg.infer.data_path,
        cfg.infer.predictions_path,
    )


if __name__ == "__main__":
    main()
