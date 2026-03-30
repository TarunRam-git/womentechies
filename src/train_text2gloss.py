import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import PathConfig


def train_text_to_gloss(paths: PathConfig):
    dict_path = paths.manifests_dir / "gloss_dict.json"
    if not dict_path.exists():
        raise FileNotFoundError(f"Run dataset_builder first. Missing {dict_path}")

    with dict_path.open("r", encoding="utf-8") as f:
        gloss_dict = json.load(f)

    X = []
    y = []
    for phrase, gloss in gloss_dict.items():
        X.append(phrase)
        y.append(gloss)

    if len(set(y)) < 2:
        raise ValueError("Need at least two gloss classes for training")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    model.fit(X, y)

    out_dir = paths.processed_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "text_to_gloss_model.pkl"

    import joblib

    joblib.dump(model, out_path)
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    train_text_to_gloss(PathConfig())
