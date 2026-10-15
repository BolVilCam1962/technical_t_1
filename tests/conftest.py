"""Entorno de tests sin red ni modelos pesados.

Se fuerza: embedder ``fake`` (determinista), LLM ``mock`` y Redis en proceso
(``memory://``). El índice de Chroma se construye una vez por sesión con una
muestra del reglamento (mismo formato que el PDF real).
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="mtg-tests-")
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["REDIS_URL"] = "memory://"
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")
# El embedder fake produce distancias mayores que un modelo real: margen amplio
# para que los tests de API no dependan del umbral de calidad.
os.environ["RETRIEVAL_MAX_DISTANCE"] = "1.5"

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.ingest.rules_parser import parse_comp_rules  # noqa: E402
from app.services.embeddings import FakeEmbedder  # noqa: E402
from app.services.retrieval import open_collection  # noqa: E402

FIXTURE_RULES = """Magic: The Gathering Comprehensive Rules (fixture de tests)

Contents
1. Game Concepts . . . . . . . . . . 3
5. Turn Structure . . . . . . . . . 20
7. Additional Rules . . . . . . . . 40

1. Game Concepts

100. General

100.1. These rules apply to any Magic game with two or more players, including
two-player games and multiplayer games.

100.1a Two-player games are the default.

106. Mana

106.1. Mana is the primary resource in the game. Players spend mana to pay
costs, usually when casting spells and activating abilities.

106.1a The five colors of mana are white, blue, black, red, and green.

5. Turn Structure

501. Beginning Phase

501.1. The beginning phase is the first phase of the turn. It has three steps:
untap, upkeep, and draw.

505. Main Phase

505.1. There are two main phases in a turn. In a two-player game, the precombat
main phase is the second phase and the postcombat main phase is the fourth
phase of a turn.

510. Combat Damage Step

510.1. First, the active player announces how each attacking creature assigns
its combat damage.

Example: A creature with first strike assigns its combat damage in the first
combat damage step.

7. Additional Rules

702. Keyword Abilities

702.7. First Strike

702.7a First strike is a static ability that creates an additional combat
damage step.

702.7b Assigning first strike combat damage happens in the first combat damage
step.

702.10. Haste

702.10a Haste is a static ability that allows a creature to attack or use
activated abilities with the tap symbol in the cost on the turn it enters
the battlefield.

Glossary

First Strike

A keyword ability that lets a creature deal combat damage before creatures
without first strike. See rule 702.7.

Haste

A keyword ability that lets a creature attack or use tap abilities the turn it
enters the battlefield. See rule 702.10.

Mana

The primary resource in the game, spent to pay costs.
"""


@pytest.fixture(scope="session", autouse=True)
def chroma_index():
    """Construye el índice de Chroma con la fixture (una vez por sesión)."""
    settings = get_settings()
    result = parse_comp_rules(FIXTURE_RULES)
    assert result.chunks, "la fixture del reglamento no se ha parseado"

    embedder = FakeEmbedder()
    embeddings = asyncio.run(
        embedder.embed_documents([chunk.embed_text for chunk in result.chunks])
    )
    _, collection = open_collection(settings.chroma_dir, settings.chroma_collection)
    collection.add(
        ids=[chunk.id for chunk in result.chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in result.chunks],
        metadatas=[chunk.metadata("test") for chunk in result.chunks],
    )
    yield
