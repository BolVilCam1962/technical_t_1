"""Ingesta del reglamento: PDF → chunks estructurales → embeddings → ChromaDB.

ARQUITECTURA.md §4. Se ejecuta como servicio one-shot de Docker:

    docker compose --profile tools run --rm ingest

o en local:

    python scripts/ingest_rules.py --pdf "MagicCompRules 20260417.pdf"

Es **idempotente**: borra y recrea la colección en cada ejecución, con
`rules_version` en los metadatos de cada chunk (citas trazables a la edición).
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Permite `python scripts/ingest_rules.py` desde cualquier cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.ingest.rules_parser import check_integrity, parse_comp_rules  # noqa: E402
from app.services.embeddings import get_embedder  # noqa: E402
from app.services.retrieval import open_collection  # noqa: E402

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    logger.info("PDF extraído", extra={"pages": len(pages), "path": pdf_path})
    return "\n".join(pages)


async def run(pdf_path: str, stats_only: bool, batch_size: int) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    text = extract_pdf_text(pdf_path)
    result = parse_comp_rules(text)

    print("\n=== Estadísticas del parseo ===")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")
    print(f"  capítulos: {[f'{n}. {t}' for n, t in result.chapters]}")
    for warning in check_integrity(result):
        print(f"  [WARN] {warning}")

    if stats_only:
        print("\n--stats-only: no se escribe en ChromaDB.")
        return

    print("\n=== Generando embeddings e índice ChromaDB ===")
    embedder = get_embedder(settings)
    await embedder.load()

    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_dir)
    try:
        client.delete_collection(settings.chroma_collection)
        logger.info("Colección previa eliminada", extra={"collection": settings.chroma_collection})
    except Exception:
        pass  # primera ejecución: no existía

    _, collection = open_collection(settings.chroma_dir, settings.chroma_collection)

    chunks = result.chunks
    started = time.perf_counter()
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        embeddings = await embedder.embed_documents([chunk.embed_text for chunk in batch])
        collection.add(
            ids=[f"{settings.rules_version}:{chunk.id}" for chunk in batch],
            embeddings=embeddings,
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata(settings.rules_version) for chunk in batch],
        )
        done = min(offset + batch_size, len(chunks))
        logger.info("Progreso de ingesta", extra={"done": done, "total": len(chunks)})

    stored = collection.count()
    elapsed = time.perf_counter() - started
    print(
        f"\n✅ Ingesta completada: {stored}/{len(chunks)} chunks en "
        f"'{settings.chroma_collection}' ({settings.chroma_dir}) en {elapsed:.1f}s"
    )
    print(f"   embedder: {embedder.name} (dim={embedder.dim})")
    print(f"   rules_version: {settings.rules_version}")
    print("\nSiguiente paso: levanta la API (docker compose up -d) y prueba en /docs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta del reglamento MTG en ChromaDB")
    parser.add_argument(
        "--pdf",
        default="MagicCompRules 20260417.pdf",
        help="Ruta al PDF del Comprehensive Rules",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Solo parsea y muestra estadísticas (no escribe en ChromaDB)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"ERROR: no existe el PDF: {args.pdf}", file=sys.stderr)
        sys.exit(2)

    asyncio.run(run(args.pdf, args.stats_only, args.batch_size))


if __name__ == "__main__":
    main()
