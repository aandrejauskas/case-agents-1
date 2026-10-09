"""Pilar 1: Router local entre FAST_PATH e AGENT."""
import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self) -> None:
        self._fitted = False
        # Palavras capturam expressões de intenção; caracteres dão robustez a
        # acentos, flexões, erros e linguagem informal.
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self.model = Pipeline(
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

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Treina o router com os exemplos de `data/router_training_data.json`."""
        if not texts:
            raise ValueError("Informe pelo menos um texto para treinar o Router.")
        if len(texts) != len(labels):
            raise ValueError("texts e labels devem ter a mesma quantidade de itens.")
        if set(labels) != {"FAST_PATH", "AGENT"}:
            raise ValueError("Os dados devem conter as classes FAST_PATH e AGENT.")

        self.model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, query: str) -> RouteResult:
        """Classifica `query` e retorna a rota escolhida + latência medida (em ms)."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()
        probabilities = self.model.predict_proba([query])[0]
        classifier = self.model.named_steps["classifier"]
        predicted_index = int(np.argmax(probabilities))
        route = str(classifier.classes_[predicted_index])
        confidence = float(probabilities[predicted_index])
        latency_ms = (time.perf_counter() - start) * 1_000

        return RouteResult(
            route=route,
            latency_ms=latency_ms,
            confidence=confidence,
        )
