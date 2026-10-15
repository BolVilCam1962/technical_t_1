"""Parser estructural del Comprehensive Rules (ARQUITECTURA.md §4).

El reglamento está numerado (``100``–``9xx`` con sub-reglas ``NNN.x`` y
sufijos por letra ``NNN.xa``) y termina con un glosario. El chunking es
**estructural**, no por tamaño fijo:

- Un chunk por sub-regla ``NNN.x`` (los sufijos por letra se fusionan en su
  sub-regla: ``702.7a``/``702.7b`` -> chunk ``702.7``).
- Las reglas raíz ``NNN.`` son cabeceras (p. ej. ``505. Main Phase``): aportan
  el *título* a sus hijas y solo generan chunk propio si tienen texto.
- Los párrafos ``Example:`` se quedan pegados a su regla.
- Cada término del glosario es un chunk independiente (puente ES→EN, §7.3).
- Los ejemplos referenciados mid-linea ("(see rule 702.7)") pueden partirse en
  línea nueva y colarse como chunk espurio: la integridad reporta duplicados y
  el dedupe se queda con la primera aparición (la definición real).

Todo el parser es puro (str -> dataclasses) y está cubierto por tests.
"""

import re
from dataclasses import dataclass, field

# "1. Game Concepts", "5. Turn Structure"...
CHAPTER_RE = re.compile(r"^([1-9])\.\s+([A-Z][\w'’\-/,& ]{2,60})$")
# Líneas del índice ("1. Game Concepts . . . . 4") o con nº de página al final.
DOT_LEADER_RE = re.compile(r"\.{3,}|\s\.\s\.\s\.")
TRAILING_PAGE_RE = re.compile(r"\s\d{1,3}$")
# "100. General", "505.1. A turn...", "702.7a First strike is..."
RULE_RE = re.compile(r"^(\d{3})(?:\.(\d{1,3}))?([a-z])?\s*\.?\s+")
# El glosario real (verificado contra el PDF con pypdf) pone cada TÉRMINO solo
# en su línea, sin puntuación final, y la definición en las líneas siguientes:
#   'Abandon'
#   'To turn a face-up ongoing scheme card face down... See rule 701.33, "Abandon."'
GLOSSARY_TERM_CHARS = re.compile(r"^[A-Za-z'’\-/, &]+$")

# Comienzos de frase que NO son términos del glosario (líneas de definición
# envueltas que empiezan en mayúscula). Ojo: hay términos reales que empiezan
# por "Copy", "One-Shot...", etc. — no añadir esos.
_SENTENCE_STARTERS = {
    "The", "If", "A", "An", "When", "This", "That", "See", "Note", "For", "It",
    "In", "On", "At", "As", "To", "Any", "Each", "Only", "Also", "Some", "Such",
    "These", "Those", "All", "Players", "Example",
}

GLOSSARY_CHAPTER_TITLE = "Glossary"


@dataclass
class ParsedChunk:
    rule_number: str  # "505.1" | "100" | término del glosario ("First Strike")
    title: str
    text: str
    chapter: str | None  # "5" para reglas, None para glosario
    chapter_title: str | None
    chunk_type: str  # "rule" | "glossary"

    @property
    def id(self) -> str:
        return f"{self.chunk_type}:{self.rule_number}"

    @property
    def embed_text(self) -> str:
        """Texto que se embeddea: incluye el capítulo como contexto (§4.2)."""
        prefix = self.chapter_title or GLOSSARY_CHAPTER_TITLE
        return f"{prefix} — {self.text}"

    def metadata(self, rules_version: str) -> dict:
        # Chroma (binding Rust) no acepta None en los metadatos -> "".
        return {
            "rule_number": self.rule_number,
            "title": self.title or "",
            "chapter": self.chapter or "",
            "chapter_title": self.chapter_title or "",
            "type": self.chunk_type,
            "rules_version": rules_version,
        }


@dataclass
class ParseResult:
    chunks: list[ParsedChunk] = field(default_factory=list)
    chapters: list[tuple[int, str]] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    top_level_rule_chunks: int = 0
    sub_rule_chunks: int = 0
    glossary_chunks: int = 0

    @property
    def stats(self) -> dict:
        return {
            "chapters": len(self.chapters),
            "top_level_rule_chunks": self.top_level_rule_chunks,
            "sub_rule_chunks": self.sub_rule_chunks,
            "glossary_chunks": self.glossary_chunks,
            "total_chunks": len(self.chunks),
            "duplicate_ids": len(self.duplicate_ids),
        }


def parse_comp_rules(text: str) -> ParseResult:
    lines = text.splitlines()

    glossary_start = _find_glossary_start(lines)
    rules_lines = lines[:glossary_start] if glossary_start is not None else lines
    glossary_lines = lines[glossary_start + 1 :] if glossary_start is not None else []

    chapters = _parse_chapters(rules_lines)
    chapter_titles = dict(chapters)
    rule_chunks = _parse_rules(rules_lines, chapter_titles)
    glossary_chunks = _parse_glossary(glossary_lines)

    chunks = rule_chunks + glossary_chunks
    seen: set[str] = set()
    duplicates: list[str] = []
    unique: list[ParsedChunk] = []
    for chunk in chunks:
        if chunk.id in seen:
            duplicates.append(chunk.id)
            continue
        seen.add(chunk.id)
        unique.append(chunk)

    result = ParseResult(chunks=unique, chapters=chapters, duplicate_ids=duplicates)
    result.top_level_rule_chunks = sum(1 for c in unique if c.chunk_type == "rule" and "." not in c.rule_number)
    result.sub_rule_chunks = sum(1 for c in unique if c.chunk_type == "rule" and "." in c.rule_number)
    result.glossary_chunks = sum(1 for c in unique if c.chunk_type == "glossary")
    return result


def check_integrity(result: ParseResult) -> list[str]:
    """Advertencias de integridad (§7.1). Se muestran tras cada ingesta."""
    warnings: list[str] = []
    chapter_numbers = {number for number, _ in result.chapters}
    missing = [str(n) for n in range(1, 10) if n not in chapter_numbers]
    if missing:
        warnings.append(f"Capítulos no detectados: {', '.join(missing)} (esperados 1-9)")
    if result.sub_rule_chunks < 1000:
        warnings.append(f"Pocas sub-reglas parseadas: {result.sub_rule_chunks} (esperadas >1000)")
    if result.glossary_chunks < 100:
        warnings.append(f"Pocos términos de glosario: {result.glossary_chunks} (esperados >100)")
    if result.duplicate_ids:
        warnings.append(f"{len(result.duplicate_ids)} chunks duplicados (posible split de línea en referencias)")
    oversized = [c.id for c in result.chunks if len(c.text) > 6000]
    if oversized:
        warnings.append(f"{len(oversized)} chunks de más de 6000 caracteres: {oversized[:5]}")
    if not result.chunks:
        warnings.append("No se ha parseado ningún chunk: revisa la extracción del PDF")
    return warnings


# ---------------------------------------------------------------------------


def _find_glossary_start(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip().lower() == "glossary":
            return index
    return None


def _parse_chapters(lines: list[str]) -> list[tuple[int, str]]:
    chapters: list[tuple[int, str]] = []
    seen: set[int] = set()
    for line in lines:
        match = CHAPTER_RE.match(line)
        if not match:
            continue
        # Ignora el índice (líneas con puntos de guía o nº de página).
        if DOT_LEADER_RE.search(line) or TRAILING_PAGE_RE.search(line):
            continue
        number = int(match.group(1))
        title = match.group(2).strip()
        if number in seen or not title:
            continue
        seen.add(number)
        chapters.append((number, title))
    return chapters


def _parse_rules(lines: list[str], chapter_titles: dict[int, str]) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    last_top_title: dict[str, str] = {}
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        text = "\n".join(current["lines"]).strip()
        chunk_number, title, top_level = current["rule_number"], current["title"], current["top_level"]
        current = None
        # Cabecera sin cuerpo ("505. Main Phase") -> estructura, no chunk.
        if top_level and len(text) <= len(f"{chunk_number}. {title}") + 10 and title:
            return
        if len(text) < 5:
            return
        chapter = chunk_number[0]
        chunks.append(
            ParsedChunk(
                rule_number=chunk_number,
                title=title,
                text=text,
                chapter=chapter,
                chapter_title=chapter_titles.get(int(chapter), ""),
                chunk_type="rule",
            )
        )

    for line in lines:
        match = RULE_RE.match(line)
        if match:
            hundreds, sub = match.group(1), match.group(2)
            number = f"{hundreds}.{sub}" if sub else hundreds
            if current is not None and current["rule_number"] == number:
                # Sub-sub-regla por letra (702.7b tras 702.7): mismo chunk.
                current["lines"].append(line)
                continue
            flush()
            if sub:
                number = f"{hundreds}.{sub}"
                # Sub-reglas de keywords con título ("702.7. First Strike").
                rest = line[match.end() :].strip()
                title = last_top_title.get(hundreds, "")
                if _looks_like_rule_title(rest):
                    title = rest
                current = {
                    "rule_number": number,
                    "title": title,
                    "lines": [line],
                    "top_level": False,
                }
            else:
                # "505. Main Phase": cabecera raíz; el texto restante de la línea es el título.
                title = line[match.end() :].strip()
                last_top_title[hundreds] = title
                current = {
                    "rule_number": hundreds,
                    "title": title,
                    "lines": [line],
                    "top_level": True,
                }
        elif current is not None:
            current["lines"].append(line)
        # else: portada, créditos, índice... -> se ignora
    flush()
    return chunks


def _parse_glossary(lines: list[str]) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    term: str | None = None
    definition: list[str] = []

    def flush() -> None:
        nonlocal term, definition
        if term is None:
            return
        text = " ".join([f"{term}."] + definition)
        if len(text) > 10:
            chunks.append(
                ParsedChunk(
                    rule_number=term,
                    title=term,
                    text=text,
                    chapter=None,
                    chapter_title=GLOSSARY_CHAPTER_TITLE,
                    chunk_type="glossary",
                )
            )
        term, definition = None, []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_term_line(stripped):
            flush()
            term = stripped
            continue
        if term is not None:
            definition.append(stripped)
    flush()
    return chunks


def _looks_like_term_line(stripped: str) -> bool:
    """Línea de término del glosario: corta, sin puntuación final, tipo título.

    'Abandon' o 'Active Player, Nonactive Player Order' sí; 'To turn a face-up
    ongoing scheme... See' o '1. Text on an object...' no.
    """
    if not 2 <= len(stripped) <= 60:
        return False
    # Las definiciones (y sus wraps) acaban en puntuación o empiezan mal.
    if stripped[-1] in '.,;:!?“”"’)]' or not stripped[0].isupper():
        return False
    words = stripped.split()
    if not 1 <= len(words) <= 7:
        return False
    if words[0] in _SENTENCE_STARTERS:
        return False
    return bool(GLOSSARY_TERM_CHARS.fullmatch(stripped))


def _looks_like_rule_title(text: str) -> bool:
    """'First Strike' sí; 'There are two main phases in a turn.' no."""
    if not text or len(text) > 60 or text.endswith("."):
        return False
    return len(text.split()) <= 6 and text[0].isupper()
