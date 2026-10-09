"""Testes de ranking e casos de borda do recuperador de tools."""
import pytest

from candidate_starter.retrieval import ToolRetriever
from common.data_loader import load_tools
from common.schemas import Tool


@pytest.fixture(scope="module")
def retriever() -> ToolRetriever:
    return ToolRetriever().fit(load_tools())


def test_search_is_ranked_and_capped(retriever: ToolRetriever) -> None:
    result = retriever.search("Quero saber meu saldo", k=2)

    assert len(result.matches) == 2
    assert [match.score for match in result.matches] == sorted(
        (match.score for match in result.matches), reverse=True
    )
    assert result.latency_ms >= 0.0


def test_second_slot_covers_general_operation(retriever: ToolRetriever) -> None:
    result = retriever.search("Preciso saber o saldo disponível pra pix", k=2)

    assert result.matches[0].name == "consultar_saldo_disponivel_pix"
    assert result.matches[1].name == "consultar_saldo"


def test_search_caps_k_at_catalog_size(retriever: ToolRetriever) -> None:
    tool_count = len(load_tools())
    assert len(retriever.search("saldo", k=tool_count + 10).matches) == tool_count


def test_search_validates_state_and_k() -> None:
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        ToolRetriever().search("saldo")

    fitted = ToolRetriever().fit(load_tools())
    with pytest.raises(ValueError, match="maior ou igual a 1"):
        fitted.search("saldo", k=0)


def test_fit_rejects_empty_or_duplicate_catalog() -> None:
    with pytest.raises(ValueError, match="pelo menos uma tool"):
        ToolRetriever().fit([])

    duplicate = Tool("duplicada", "Descrição", "categoria")
    with pytest.raises(ValueError, match="únicos"):
        ToolRetriever().fit([duplicate, duplicate])
