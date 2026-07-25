"""Split documents into overlapping passage-level chunks.

RAG retrieval works best on chunks of ~1-3 paragraphs: big enough to
carry meaning, small enough that the retrieved context stays focused.
Strategy: split on blank lines (paragraphs), then greedily pack
paragraphs into chunks up to CHUNK_SIZE characters, carrying a small
overlap so information on chunk boundaries is not lost.
"""

from __future__ import annotations

import re
from typing import List

from . import config


def split_paragraphs(text: str) -> List[str]:
    """Split raw text into non-empty paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Pack paragraphs into chunks of at most `chunk_size` characters.

    A paragraph longer than `chunk_size` is split at word boundaries.
    Consecutive chunks share roughly `overlap` characters of context.
    """
    chunks: List[str] = []
    current = ""

    for paragraph in split_paragraphs(text):
        # Hard-split any oversized paragraph at word boundaries.
        pieces = _split_long_paragraph(paragraph, chunk_size)
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + len(piece) + 2 <= chunk_size:
                current += "\n\n" + piece
            else:
                chunks.append(current)
                # Start next chunk with a tail of the previous one (overlap).
                tail = current[-overlap:] if overlap > 0 else ""
                current = (tail + "\n\n" + piece).strip() if tail else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """Split a single paragraph longer than chunk_size at word boundaries."""
    if len(paragraph) <= chunk_size:
        return [paragraph]
    words = paragraph.split()
    pieces, current = [], ""
    for word in words:
        if not current:
            current = word
        elif len(current) + len(word) + 1 <= chunk_size:
            current += " " + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces
