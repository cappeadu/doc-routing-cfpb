import json
import time
from typing import Annotated

import torch
import tqdm
import typer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import logger
from src_embeddings.data import clean_text_for_embeddings

app = typer.Typer()


class BERTPredictor:
    """Predictor class to load BERT model."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model.eval()
        self.label2id = self.model.config.label2id

    def predict_batch(self, texts, batch_size=16):
        results = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Predicting batches...",
            colour="green",
        ):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_index = torch.argmax(probabilities, dim=-1)

            for idx, pred in enumerate(predicted_index.tolist()):
                results.append(
                    {
                        "label": self.model.config.id2label[pred],
                        "score": probabilities[idx, pred].item(),
                    }
                )
        return results


# predict with proba
def predict_with_proba_bert(
    text: str, predictor: BERTPredictor, threshold: float = 0.7
) -> list[dict]:
    """Perform predictions and show probabilities associated with predictions.

    Args:
        text (str): Text to be predicted.
        predictor (BERTPredictor): Predictor class to perform predictions.
        threshold (float): Prediction propabilities above this threshold are automatically routed.

    Returns:
        list[dict]: list of predictions.

    """
    start_time = time.time()
    text = clean_text_for_embeddings(text)
    inputs = predictor.tokenizer(
        text, padding=True, max_length=128, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(predictor.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = predictor.model(**inputs)
    probabilities = torch.softmax(output.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    end_time = time.time()

    id2label = predictor.model.config.id2label
    label2id = list(predictor.model.config.label2id)
    probabilities_list = probabilities[0].tolist()
    all_probs = {
        label: round(prob, 4) for label, prob in zip(label2id, probabilities_list)
    }
    sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    prob = probabilities_list[prediction.item()]
    decision = "auto_route" if prob >= threshold else "human_review"
    return {
        "prediction": id2label[prediction.item()],
        "probabilities": sorted_probs,
        "latency": f"{(end_time - start_time) * 1000:.2f} ms",
        "decision": decision,
    }


@app.command()
def predict_text_cli(
    text: Annotated[str, typer.Option(help="complaint text")],
) -> list[dict]:
    """Predict complaint text into appropriate department/unit.

    Args:
        text (str): complaint text to predict.

    Returns:
        list[dict]: Prediction result.
    """

    CHECKPOINT_BERT = "appcle/distilbert-base-uncased-cfpd"
    predictor = BERTPredictor(CHECKPOINT_BERT)
    results = predict_with_proba_bert(text, predictor)
    print(results)
    logger.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    app()
