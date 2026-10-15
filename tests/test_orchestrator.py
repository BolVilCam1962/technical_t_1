"""Tests del post-procesado del orquestador: validación de citas anti-alucinación
y extracción de JSON (ARQUITECTURA.md §7)."""

from app.agent.orchestrator import build_citations, extract_json_object
from app.services.retrieval import RetrievedRule


def _rule(number: str, distance: float) -> RetrievedRule:
    return RetrievedRule(
        rule_number=number, title=f"Rule {number}", text=f"Text of {number}.", distance=distance
    )


RETRIEVED = [_rule("702.7", 0.10), _rule("510.1", 0.20), _rule("505.1", 0.30)]


def test_citations_matching_retrieved_rules_are_kept():
    citations = build_citations([{"rule_number": "702.7"}], RETRIEVED)
    assert [c.rule_number for c in citations] == ["702.7"]
    assert citations[0].score > 0


def test_hallucinated_citations_are_dropped():
    citations = build_citations([{"rule_number": "702.7"}, {"rule_number": "999.9"}], RETRIEVED)
    assert [c.rule_number for c in citations] == ["702.7"]


def test_citation_matching_by_prefix_and_letter_suffix():
    # "505" casa con el chunk "505.1"; "702.7a" casa con el chunk "702.7".
    citations = build_citations([{"rule_number": "505"}, {"rule_number": "702.7a"}], RETRIEVED)
    assert {c.rule_number for c in citations} == {"505.1", "702.7"}


def test_fallback_top_rules_when_llm_cites_nothing():
    citations = build_citations([], RETRIEVED)
    assert [c.rule_number for c in citations] == ["702.7", "510.1", "505.1"]


def test_no_citations_when_nothing_retrieved():
    assert build_citations([{"rule_number": "702.7"}], []) == []


def test_citations_accept_plain_strings():
    citations = build_citations(["510.1"], RETRIEVED)
    assert [c.rule_number for c in citations] == ["510.1"]


def test_extract_json_object_plain():
    assert extract_json_object('{"answer": "hola"}') == {"answer": "hola"}


def test_extract_json_object_fenced():
    fenced = "```json\n{\"answer\": \"hola\", \"citations\": []}\n```"
    assert extract_json_object(fenced)["answer"] == "hola"


def test_extract_json_object_with_surrounding_text():
    text = 'Here you go:\n{"answer": "hola"}\nHope it helps!'
    assert extract_json_object(text) == {"answer": "hola"}


def test_extract_json_object_returns_none_on_plain_text():
    assert extract_json_object("no hay json aqui") is None
