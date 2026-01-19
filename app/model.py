from functools import lru_cache

from transformers import pipeline

MODEL_ID = "s-nlp/russian_toxicity_classifier"


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        truncation=True,
    )


def predict_single(text: str) -> dict:
    classifier = get_classifier()
    item = classifier(text)[0]
    return {
        "label": item["label"],
        "score": float(item["score"]),
        "raw": item,
    }


def predict_batch(texts: list[str]) -> list[dict]:
    classifier = get_classifier()
    outputs = classifier(texts)

    return [
        {
            "text": text,
            "prediction": {
                "label": item["label"],
                "score": float(item["score"]),
                "raw": item,
            },
        }
        for text, item in zip(texts, outputs)
    ]


def get_model_info() -> dict:
    return {
        "model_id": MODEL_ID,
        "framework": "transformers",
    }
