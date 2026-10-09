"""Seleção do modelo do Router usando somente os dados de treinamento."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline

from common.data_loader import load_eval_dataset, load_router_training_data


def word_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )


def char_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )


def make_pipeline(features) -> Pipeline:
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    texts, labels = load_router_training_data()
    candidates = {
        "word": make_pipeline(word_features()),
        "char": make_pipeline(char_features()),
        "word+char": make_pipeline(
            FeatureUnion([("word", word_features()), ("char", char_features())])
        ),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    for name, pipeline in candidates.items():
        scores = cross_validate(pipeline, texts, labels, cv=cv, scoring=scoring)
        values = " ".join(
            f"{metric}={scores[f'test_{metric}'].mean():.3f}±{scores[f'test_{metric}'].std():.3f}"
            for metric in scoring
        )
        print(f"{name:10s} {values}")

    # O eval fornecido só é medido depois da seleção feita no conjunto de treino.
    selected = candidates["word+char"].fit(texts, labels)
    eval_items = load_eval_dataset()
    truth = [item["expected_route"] for item in eval_items]
    predictions = selected.predict([item["query"] for item in eval_items])
    print(
        "eval",
        {
            "accuracy": accuracy_score(truth, predictions),
            "balanced_accuracy": balanced_accuracy_score(truth, predictions),
            "f1_macro": f1_score(truth, predictions, average="macro"),
        },
    )
    for item, prediction in zip(eval_items, predictions):
        if item["expected_route"] != prediction:
            print("miss", item["expected_route"], prediction, item["query"])


if __name__ == "__main__":
    main()
