"""Evaluación offline del retrieval semántico (ARQUITECTURA.md §7.2).

Golden set: pares (pregunta en español → regla esperada) contra el índice real
(reglas en inglés) — el test cruzado de idioma que valida la elección del
embedder multilingüe.

Métricas:
- Recall@k: la regla esperada aparece en el top-k (k=5 y k=10).
- MRR@10: inverso del rango de la primera aparición.

Gate de calidad: ``--min-recall`` (default 0.8 sobre Recall@5). Exit code 1 si
no se cumple — pensado para CI (anti-regresión al cambiar embedder/chunking).

    docker compose --profile tools run --rm eval
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.services.embeddings import get_embedder  # noqa: E402
from app.services.retrieval import RulesRetriever  # noqa: E402


def rule_matches(candidate: str, expected: str) -> bool:
    """"505" casa con "505.1"; "702.7" casa con "702.7a"; exacto también."""
    candidate, expected = candidate.strip(), expected.strip()
    return (
        candidate == expected
        or candidate.startswith(expected + ".")
        or expected.startswith(candidate + ".")
    )


def chunk_matches(chunk, expected: str) -> str | None:
    """Cómo matchea el chunk con la regla esperada:
    'directo' (número de regla) | 'glosario' (la entrada del glosario
    referencia la regla esperada en su texto, p. ej. 'See rule 508') | None.

    El caso 'glosario' es un hit legítimo por diseño (§7.3): la entrada del
    glosario ES el puente ES→EN y apunta a la regla normativa.
    """
    if rule_matches(chunk.rule_number, expected):
        return "directo"
    if chunk.chapter_title == "Glossary":
        if re.search(rf"rules?\s+{re.escape(expected)}", chunk.text):
            return "glosario"
    return None


async def run(golden_path: str, k: int, min_recall: float) -> int:
    settings = get_settings()
    setup_logging("WARNING")

    items = []
    with open(golden_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Golden set: {len(items)} preguntas ({golden_path})\n")

    embedder = get_embedder(settings)
    await embedder.load()
    retriever = RulesRetriever(embedder, settings)
    total = await retriever.count()
    if total == 0:
        print("ERROR: índice ChromaDB vacío. Ejecuta antes la ingesta.")
        return 2
    print(f"Índice: {total} chunks | embedder: {embedder.name}\n")

    hits5 = hits10 = 0
    mrr_sum = 0.0
    print(f"{'#':>3} {'Recall@5':>8} {'Recall@10':>9} {'MRR@10':>7}  Pregunta")
    print("-" * 78)
    for index, item in enumerate(items, 1):
        rules = await retriever.search(item["question"], k=k)
        matches = [chunk_matches(rule, item["expected_rule"]) for rule in rules]
        ranks = [rank for rank, how in enumerate(matches, 1) if how]
        hit5 = bool(ranks and ranks[0] <= 5)
        hit10 = bool(ranks and ranks[0] <= 10)
        rr = 1.0 / ranks[0] if ranks else 0.0
        hits5 += hit5
        hits10 += hit10
        mrr_sum += rr
        flag5 = "  ✓" if hit5 else "  ✗"
        top3 = ",".join(r.rule_number for r in rules[:3]) or "-"
        print(
            f"{index:>3} {str(hit5):>8} {str(hit10):>9} {rr:>7.3f}{flag5}  "
            f"{item['question'][:48]:<48} esperado={item['expected_rule']:<7} top3={top3}"
        )

    n = len(items)
    recall5, recall10, mrr = hits5 / n, hits10 / n, mrr_sum / n
    print("-" * 78)
    print(f"Recall@5:  {recall5:.2f} ({hits5}/{n})")
    print(f"Recall@10: {recall10:.2f} ({hits10}/{n})")
    print(f"MRR@10:    {mrr:.3f}")
    print(f"Umbral:    Recall@5 >= {min_recall}")

    if recall5 < min_recall:
        print(f"\n❌ FAIL: Recall@5 {recall5:.2f} < {min_recall}.")
        print("   Revisa: modelo de embeddings (¿es multilingüe?), chunking (scripts/ingest_rules.py --stats-only),")
        print("   o RETRIEVAL_MAX_DISTANCE (demasiado restrictivo descarta resultados).")
        return 1
    print("\n✅ PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del retrieval semántico")
    parser.add_argument("--golden", default="eval/golden_set.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()

    if not Path(args.golden).exists():
        print(f"ERROR: no existe el golden set: {args.golden}", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(run(args.golden, args.k, args.min_recall)))


if __name__ == "__main__":
    main()
