"""Comparação de diagnóstico para recuperadores leves e reproduzíveis.

Este arquivo fica fora do pacote de execução de forma intencional. Ele compara
um conjunto pequeno e pré-definido de estratégias gerais de ranking, sem regras
específicas para queries ou tools.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

from common.data_loader import load_eval_dataset, load_tools


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


@dataclass(frozen=True)
class Config:
    name: str
    word_weight: float
    char_weight: float
    name_repetitions: int
    generality_bonus: float = 0.0


CONFIGS = [
    Config("word", 1.0, 0.0, 1),
    Config("word_name3", 1.0, 0.0, 3),
    Config("char", 0.0, 1.0, 1),
    Config("hybrid", 0.65, 0.35, 1),
    Config("hybrid_name3", 0.65, 0.35, 3),
    Config("hybrid_generic_002", 0.65, 0.35, 2, 0.02),
    Config("hybrid_generic_004", 0.65, 0.35, 2, 0.04),
    Config("hybrid_generic_006", 0.65, 0.35, 2, 0.06),
    Config("hybrid_generic_008", 0.65, 0.35, 2, 0.08),
    Config("char_generic_004", 0.0, 1.0, 1, 0.04),
    Config("char_generic_008", 0.0, 1.0, 1, 0.08),
    Config("char_generic_012", 0.0, 1.0, 1, 0.12),
    Config("char_generic_016", 0.0, 1.0, 1, 0.16),
]


def build_documents(name_repetitions: int) -> list[str]:
    documents = []
    for tool in load_tools():
        readable_name = tool.name.replace("_", " ")
        name_block = " ".join([readable_name] * name_repetitions)
        documents.append(normalize(f"{name_block}. {tool.description}. {tool.category}"))
    return documents


def evaluate(config: Config) -> tuple[dict[str, float], list[tuple[str, int, list[str]]]]:
    tools = load_tools()
    eval_items = [item for item in load_eval_dataset() if item.get("expected_tool")]
    documents = build_documents(config.name_repetitions)
    queries = [normalize(item["query"]) for item in eval_items]

    word = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", norm="l2"
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, norm="l2"
    )
    word_matrix = word.fit_transform(documents)
    char_matrix = char.fit_transform(documents)
    word_scores = cosine_similarity(word.transform(queries), word_matrix)
    char_scores = cosine_similarity(char.transform(queries), char_matrix)
    scores = config.word_weight * word_scores + config.char_weight * char_scores

    if config.generality_bonus:
        # Um nome curto funciona como um sinal fraco, independente do domínio,
        # de uma ação geral em comparação com uma variante muito específica.
        name_lengths = np.array(
            [len(normalize(tool.name).split()) for tool in tools], dtype=np.float64
        )
        bonus = 1.0 / np.sqrt(name_lengths)
        scores += config.generality_bonus * bonus[None, :]

    ranks = []
    rows = []
    for item, query_scores in zip(eval_items, scores):
        order = np.argsort(-query_scores, kind="stable")
        names = [tools[index].name for index in order]
        rank = names.index(item["expected_tool"]) + 1
        ranks.append(rank)
        rows.append((item["query"], rank, names[:5]))

    metrics = {
        "hit@1": np.mean([rank <= 1 for rank in ranks]),
        "hit@2": np.mean([rank <= 2 for rank in ranks]),
        "hit@3": np.mean([rank <= 3 for rank in ranks]),
        "hit@5": np.mean([rank <= 5 for rank in ranks]),
        "mrr": np.mean([1.0 / rank for rank in ranks]),
    }
    return metrics, rows


def evaluate_general_expansion(
    neighbor_count: int, similarity_floor: float
) -> tuple[dict[str, float], list[tuple[str, int, list[str]]]]:
    """Recupera um resultado preciso e uma alternativa geral relacionada.

    O catálogo contém muitas tools quase duplicadas. O primeiro resultado
    preserva a precisão lexical; o segundo cobre uma operação relacionada e
    mais curta, em vez de repetir outra variante específica quase idêntica.
    """
    tools = load_tools()
    eval_items = [item for item in load_eval_dataset() if item.get("expected_tool")]
    documents = build_documents(name_repetitions=1)
    queries = [normalize(item["query"]) for item in eval_items]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, norm="l2"
    )
    matrix = vectorizer.fit_transform(documents)
    query_scores = cosine_similarity(vectorizer.transform(queries), matrix)
    tool_similarity = cosine_similarity(matrix)
    name_lengths = np.array(
        [len(normalize(tool.name).split()) for tool in tools], dtype=np.float64
    )

    ranks = []
    rows = []
    for item, scores in zip(eval_items, query_scores):
        order = np.argsort(-scores, kind="stable")
        first = int(order[0])
        related = np.argsort(-tool_similarity[first], kind="stable")
        related = [
            int(index)
            for index in related
            if index != first and tool_similarity[first, index] >= similarity_floor
        ][:neighbor_count]

        if related:
            # A preferência por uma operação geral serve apenas como desempate
            # dentro de uma vizinhança semântica próxima, nunca de forma global.
            second = min(
                related,
                key=lambda index: (
                    name_lengths[index],
                    -tool_similarity[first, index],
                    index,
                ),
            )
        else:
            second = int(order[1])

        remaining = [int(index) for index in order if index not in {first, second}]
        final_order = [first, second, *remaining]
        names = [tools[index].name for index in final_order]
        rank = names.index(item["expected_tool"]) + 1
        ranks.append(rank)
        rows.append((item["query"], rank, names[:5]))

    metrics = {
        "hit@1": np.mean([rank <= 1 for rank in ranks]),
        "hit@2": np.mean([rank <= 2 for rank in ranks]),
        "hit@3": np.mean([rank <= 3 for rank in ranks]),
        "hit@5": np.mean([rank <= 5 for rank in ranks]),
        "mrr": np.mean([1.0 / rank for rank in ranks]),
    }
    return metrics, rows


def evaluate_hierarchical() -> tuple[dict[str, float], list[tuple[str, int, list[str]]]]:
    """Usa a estrutura dos nomes antes de recorrer às relações semânticas."""
    tools = load_tools()
    eval_items = [item for item in load_eval_dataset() if item.get("expected_tool")]
    documents = build_documents(name_repetitions=1)
    queries = [normalize(item["query"]) for item in eval_items]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, norm="l2"
    )
    matrix = vectorizer.fit_transform(documents)
    query_scores = cosine_similarity(vectorizer.transform(queries), matrix)
    tool_similarity = cosine_similarity(matrix)
    name_tokens = [set(normalize(tool.name).split()) for tool in tools]

    ranks = []
    rows = []
    for item, scores in zip(eval_items, query_scores):
        order = np.argsort(-scores, kind="stable")
        first = int(order[0])
        first_tokens = name_tokens[first]
        same_category = [
            index
            for index, tool in enumerate(tools)
            if index != first and tool.category == tools[first].category
        ]
        subsets = [
            index
            for index in same_category
            if name_tokens[index] < first_tokens
            and len(name_tokens[index]) <= len(first_tokens) - 2
        ]
        strong_core = [
            index
            for index in same_category
            if len(name_tokens[index]) <= len(first_tokens) - 2
            and len(name_tokens[index] & first_tokens) >= 3
        ]
        if subsets:
            second = min(
                subsets,
                key=lambda index: (
                    len(name_tokens[index]),
                    -scores[index],
                    index,
                ),
            )
        elif strong_core:
            second = max(
                strong_core,
                key=lambda index: (
                    len(name_tokens[index] & first_tokens) / len(name_tokens[index]),
                    scores[index],
                    -len(name_tokens[index]),
                ),
            )
        else:
            related = [
                int(index)
                for index in np.argsort(-tool_similarity[first], kind="stable")
                if index != first and tool_similarity[first, index] >= 0.15
            ][:5]
            second = (
                min(
                    related,
                    key=lambda index: (
                        len(name_tokens[index]),
                        -tool_similarity[first, index],
                        index,
                    ),
                )
                if related
                else int(order[1])
            )

        final = [first, second] + [
            int(index) for index in order if index not in {first, second}
        ]
        names = [tools[index].name for index in final]
        rank = names.index(item["expected_tool"]) + 1
        ranks.append(rank)
        rows.append((item["query"], rank, names[:5]))

    metrics = {
        "hit@1": np.mean([rank <= 1 for rank in ranks]),
        "hit@2": np.mean([rank <= 2 for rank in ranks]),
        "hit@3": np.mean([rank <= 3 for rank in ranks]),
        "hit@5": np.mean([rank <= 5 for rank in ranks]),
        "mrr": np.mean([1.0 / rank for rank in ranks]),
    }
    return metrics, rows


def evaluate_hierarchical_lsa(
    components: int, neighbor_count: int
) -> tuple[dict[str, float], list[tuple[str, int, list[str]]]]:
    """Usa a semântica latente do catálogo apenas como alternativa final."""
    tools = load_tools()
    items = [item for item in load_eval_dataset() if item.get("expected_tool")]
    documents = build_documents(name_repetitions=1)
    queries = [normalize(item["query"]) for item in items]
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, norm="l2"
    )
    char_matrix = char.fit_transform(documents)
    query_scores = cosine_similarity(char.transform(queries), char_matrix)

    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm="l2")
    word_matrix = word.fit_transform(documents)
    lsa = TruncatedSVD(n_components=components, random_state=42)
    latent = l2_normalize(lsa.fit_transform(word_matrix))
    latent_similarity = latent @ latent.T
    name_tokens = [set(normalize(tool.name).split()) for tool in tools]

    ranks = []
    rows = []
    for item, scores in zip(items, query_scores):
        order = np.argsort(-scores, kind="stable")
        first = int(order[0])
        first_tokens = name_tokens[first]
        same_category = [
            index
            for index, tool in enumerate(tools)
            if index != first and tool.category == tools[first].category
        ]
        subsets = [
            index
            for index in same_category
            if name_tokens[index] < first_tokens
            and len(name_tokens[index]) <= len(first_tokens) - 2
        ]
        strong_core = [
            index
            for index in same_category
            if len(name_tokens[index]) <= len(first_tokens) - 2
            and len(name_tokens[index] & first_tokens) >= 3
        ]
        if subsets:
            second = min(
                subsets,
                key=lambda index: (len(name_tokens[index]), -scores[index], index),
            )
        elif strong_core:
            second = max(
                strong_core,
                key=lambda index: (
                    len(name_tokens[index] & first_tokens) / len(name_tokens[index]),
                    scores[index],
                    -len(name_tokens[index]),
                ),
            )
        else:
            related = [
                int(index)
                for index in np.argsort(-latent_similarity[first], kind="stable")
                if index != first
            ][:neighbor_count]
            second = min(
                related,
                key=lambda index: (
                    len(name_tokens[index]),
                    -latent_similarity[first, index],
                    index,
                ),
            )
        final = [first, second] + [
            int(index) for index in order if index not in {first, second}
        ]
        names = [tools[index].name for index in final]
        rank = names.index(item["expected_tool"]) + 1
        ranks.append(rank)
        rows.append((item["query"], rank, names[:5]))

    result = {
        "hit@1": np.mean([rank <= 1 for rank in ranks]),
        "hit@2": np.mean([rank <= 2 for rank in ranks]),
        "hit@3": np.mean([rank <= 3 for rank in ranks]),
        "hit@5": np.mean([rank <= 5 for rank in ranks]),
        "mrr": np.mean([1.0 / rank for rank in ranks]),
    }
    return result, rows


def main() -> None:
    results = []
    best = None
    for config in CONFIGS:
        metrics, rows = evaluate(config)
        results.append((config, metrics))
        if best is None or (metrics["hit@2"], metrics["mrr"]) > (
            best[1]["hit@2"],
            best[1]["mrr"],
        ):
            best = (config, metrics, rows)

    for neighbor_count, floor in [(5, 0.15), (10, 0.15), (20, 0.10), (30, 0.08)]:
        metrics, rows = evaluate_general_expansion(neighbor_count, floor)
        config = Config(
            f"general_n{neighbor_count}_f{floor}", 0.0, 1.0, 1
        )
        results.append((config, metrics))
        if best is None or (metrics["hit@2"], metrics["mrr"]) > (
            best[1]["hit@2"],
            best[1]["mrr"],
        ):
            best = (config, metrics, rows)

    metrics, rows = evaluate_hierarchical()
    config = Config("hierarchical", 0.0, 1.0, 1)
    results.append((config, metrics))
    if best is None or (metrics["hit@2"], metrics["mrr"]) > (
        best[1]["hit@2"],
        best[1]["mrr"],
    ):
        best = (config, metrics, rows)

    for components, neighbor_count in [(32, 5), (64, 5), (96, 10), (128, 10)]:
        metrics, rows = evaluate_hierarchical_lsa(components, neighbor_count)
        config = Config(f"hierarchical_lsa_{components}_{neighbor_count}", 0, 1, 1)
        results.append((config, metrics))
        if best is None or (metrics["hit@2"], metrics["mrr"]) > (
            best[1]["hit@2"],
            best[1]["mrr"],
        ):
            best = (config, metrics, rows)

    for config, metrics in results:
        values = " ".join(f"{key}={value:.3f}" for key, value in metrics.items())
        print(f"{config.name:24s} {values}")

    assert best is not None
    print(f"\nBest diagnostic configuration: {best[0].name}")
    for query, rank, names in best[2]:
        print(f"rank={rank:3d} | {query}\n          {names}")


if __name__ == "__main__":
    main()
