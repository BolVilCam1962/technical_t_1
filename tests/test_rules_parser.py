"""Tests del parser estructural (integridad de la ingesta, ARQUITECTURA.md §7.1)."""

from app.ingest.rules_parser import check_integrity, parse_comp_rules

from tests.conftest import FIXTURE_RULES


def test_chapters_parsed_and_contents_ignored():
    result = parse_comp_rules(FIXTURE_RULES)
    chapters = dict(result.chapters)
    # Las líneas del índice (con puntos de guía) no deben contar como capítulos.
    assert chapters == {
        1: "Game Concepts",
        5: "Turn Structure",
        7: "Additional Rules",
    }


def test_sub_rules_and_letter_suffixes_merge():
    result = parse_comp_rules(FIXTURE_RULES)
    numbers = {chunk.rule_number for chunk in result.chunks}
    # 702.7a + 702.7b se fusionan en el chunk 702.7; 100.1 + 100.1a en 100.1.
    assert "702.7" in numbers
    assert "702.10" in numbers
    assert "100.1" in numbers
    assert not any(n.startswith("702.7a") for n in numbers)


def test_keyword_rules_keep_their_title():
    result = parse_comp_rules(FIXTURE_RULES)
    by_number = {c.rule_number: c for c in result.chunks}
    assert by_number["702.7"].title == "First Strike"
    assert by_number["702.10"].title == "Haste"
    # Las hijas de una regla raíz heredan el contexto del capítulo.
    assert by_number["510.1"].chapter_title == "Turn Structure"
    assert by_number["510.1"].chapter == "5"


def test_pure_headings_do_not_become_chunks():
    result = parse_comp_rules(FIXTURE_RULES)
    numbers = {chunk.rule_number for chunk in result.chunks}
    # "505. Main Phase" es cabecera: aporta título/estructura, no chunk.
    assert "505" not in numbers
    assert "505.1" in numbers


def test_examples_stay_attached_to_their_rule():
    result = parse_comp_rules(FIXTURE_RULES)
    by_number = {c.rule_number: c for c in result.chunks}
    assert "Example:" in by_number["510.1"].text


def test_glossary_entries_are_chunks():
    result = parse_comp_rules(FIXTURE_RULES)
    glossary = {c.rule_number: c for c in result.chunks if c.chunk_type == "glossary"}
    assert {"First Strike", "Haste", "Mana"} <= set(glossary)
    assert glossary["First Strike"].chapter_title == "Glossary"
    # Término + definición multilínea aplanados en un solo texto.
    assert glossary["First Strike"].text.startswith("First Strike. A keyword ability")
    assert "tap abilities" in glossary["Haste"].text


def test_no_duplicate_ids():
    result = parse_comp_rules(FIXTURE_RULES)
    ids = [chunk.id for chunk in result.chunks]
    assert len(ids) == len(set(ids))
    assert result.duplicate_ids == []


def test_embed_text_includes_chapter_context():
    result = parse_comp_rules(FIXTURE_RULES)
    by_number = {c.rule_number: c for c in result.chunks}
    assert by_number["510.1"].embed_text.startswith("Turn Structure —")


def test_integrity_check_on_fixture_reports_only_expected_gaps():
    result = parse_comp_rules(FIXTURE_RULES)
    warnings = check_integrity(result)
    # La fixture es minúscula: deben aparecer los avisos de tamaño, pero no
    # avisos estructurales graves (duplicados o chunks vacíos).
    assert any("sub-reglas" in w for w in warnings)
    assert not any("duplicados" in w for w in warnings)
