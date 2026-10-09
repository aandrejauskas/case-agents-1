"""Pilar 2: busca exata e diversificada no catálogo de tools."""
import re
import time
import unicodedata
from typing import List, Optional, Set

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as l2_normalize

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


class ToolRetriever(BaseToolRetriever):
    """Retriever em caracteres com diversificação por operação geral.

    O catálogo contém operações quase duplicadas. O ranking preserva a variante
    mais específica em primeiro e usa a segunda vaga para uma operação relacionada
    mais geral, quando ela existe.
    """

    _SEMANTIC_NEIGHBORS = 5
    _LSA_COMPONENTS = 64

    def __init__(self) -> None:
        self._tools: List[Tool] = []
        self._matrix: Optional[csr_matrix] = None
        self._sibling_similarity: Optional[np.ndarray] = None
        self._name_tokens: List[Set[str]] = []
        self._fitted = False
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            norm="l2",
        )

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo de tools para busca."""
        if not tools:
            raise ValueError("Informe pelo menos uma tool para indexar.")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Os nomes das tools devem ser únicos.")

        self._tools = list(tools)
        documents = [self._tool_to_text(tool) for tool in self._tools]
        self._matrix = self.vectorizer.fit_transform(documents).tocsr()
        # A LSA aprende sinonímia local entre descrições. Ela serve somente para
        # diversificar o segundo resultado; a busca em tempo de consulta permanece esparsa.
        semantic_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        semantic_matrix = semantic_vectorizer.fit_transform(documents)
        component_count = min(
            self._LSA_COMPONENTS,
            semantic_matrix.shape[0] - 1,
            semantic_matrix.shape[1] - 1,
        )
        if component_count >= 1:
            latent = TruncatedSVD(
                n_components=component_count,
                random_state=42,
            ).fit_transform(semantic_matrix)
            latent = l2_normalize(latent)
            self._sibling_similarity = latent @ latent.T
        else:
            self._sibling_similarity = cosine_similarity(self._matrix)
        self._name_tokens = [set(self._normalize(tool.name).split()) for tool in tools]
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para `query`."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")
        if k < 1:
            raise ValueError("k deve ser maior ou igual a 1.")

        start = time.perf_counter()
        assert self._matrix is not None
        query_vector = self.vectorizer.transform([self._normalize(query)])
        raw_scores = cosine_similarity(query_vector, self._matrix)[0]
        raw_order = np.argsort(-raw_scores, kind="stable")
        result_count = min(k, len(self._tools))
        final_order = [int(raw_order[0])]

        if result_count >= 2:
            sibling = self._general_sibling(final_order[0], raw_scores, raw_order)
            final_order.append(sibling)

        selected = set(final_order)
        final_order.extend(
            int(index)
            for index in raw_order
            if int(index) not in selected
        )
        final_order = final_order[:result_count]

        # As pontuações representam o ranking final. O pequeno ajuste determinístico
        # mantém os valores monotônicos após a diversificação do segundo item.
        reranked_scores = raw_scores.copy()
        if len(final_order) >= 2:
            top_score = float(raw_scores[final_order[0]])
            reranked_scores[final_order[0]] = top_score + 2e-9
            reranked_scores[final_order[1]] = top_score + 1e-9

        matches = [
            ToolMatch(name=self._tools[index].name, score=float(reranked_scores[index]))
            for index in final_order
        ]
        latency_ms = (time.perf_counter() - start) * 1_000
        return RetrievalResult(matches=matches, latency_ms=latency_ms)

    def _general_sibling(
        self,
        first: int,
        query_scores: np.ndarray,
        raw_order: np.ndarray,
    ) -> int:
        """Seleciona uma irmã geral sem regras específicas por query."""
        assert self._sibling_similarity is not None
        first_tokens = self._name_tokens[first]
        same_category = [
            index
            for index, tool in enumerate(self._tools)
            if index != first and tool.category == self._tools[first].category
        ]

        # Um subconjunto estrito que remove ao menos dois modificadores é um
        # sinal forte. No catálogo, consultar_saldo é uma alternativa geral para
        # consultar_saldo_disponivel_pix, por exemplo.
        subsets = [
            index
            for index in same_category
            if self._name_tokens[index] < first_tokens
            and len(self._name_tokens[index]) <= len(first_tokens) - 2
        ]
        if subsets:
            return min(
                subsets,
                key=lambda index: (
                    len(self._name_tokens[index]),
                    -query_scores[index],
                    index,
                ),
            )

        # Verbos diferentes ainda podem descrever a mesma operação central
        # (emitir/solicitar segunda via cartão). São exigidos três tokens em
        # comum e uma redução relevante para evitar relações amplas por categoria.
        strong_core = [
            index
            for index in same_category
            if len(self._name_tokens[index]) <= len(first_tokens) - 2
            and len(self._name_tokens[index] & first_tokens) >= 3
        ]
        if strong_core:
            return max(
                strong_core,
                key=lambda index: (
                    len(self._name_tokens[index] & first_tokens)
                    / len(self._name_tokens[index]),
                    query_scores[index],
                    -len(self._name_tokens[index]),
                ),
            )

        related = [
            int(index)
            for index in np.argsort(
                -self._sibling_similarity[first], kind="stable"
            )
            if index != first
        ][: self._SEMANTIC_NEIGHBORS]
        if related:
            return min(
                related,
                key=lambda index: (
                    len(self._name_tokens[index]),
                    -self._sibling_similarity[first, index],
                    index,
                ),
            )

        return int(raw_order[1])

    @classmethod
    def _tool_to_text(cls, tool: Tool) -> str:
        readable_name = tool.name.replace("_", " ")
        return cls._normalize(
            f"{readable_name}. {tool.description}. Categoria: {tool.category}."
        )

    @staticmethod
    def _normalize(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text.lower())
        without_accents = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()
