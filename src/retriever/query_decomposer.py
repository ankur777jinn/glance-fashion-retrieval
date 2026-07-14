# src/retriever/query_decomposer.py
"""Query decomposition for compositional fashion search.

Breaks a free-text user query into structured components (clothing items,
colours, environment, style) using gazetteer matching, then generates
independent sub-phrases for per-attribute CLIP encoding.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config import (
    CLOTHING_CATEGORIES,
    COLOR_CATEGORIES,
    ENVIRONMENT_CATEGORIES,
    STYLE_CATEGORIES,
)

logger = logging.getLogger(__name__)

# ── Precompiled patterns ─────────────────────────────────────────────────────
# Conjunctions and prepositions used to split queries into sub-phrases.
_SPLIT_PATTERN = re.compile(
    r"\s+(?:and|,\s*and|,|with|in|at|on|for|during)\s+",
    re.IGNORECASE,
)

# Colour modifiers that combine with a base colour: "bright yellow".
_COLOR_MODIFIERS: set[str] = {
    "bright", "dark", "light", "pale", "deep", "neon",
    "pastel", "muted", "vivid",
}


def _build_sorted_terms(terms: list[str]) -> list[str]:
    """Return gazetteer terms sorted longest-first for greedy matching."""
    return sorted(terms, key=len, reverse=True)


class QueryDecomposer:
    """Extract structured fashion attributes from a natural-language query.

    Uses predefined gazetteers (from ``src.config``) and lightweight
    regex heuristics - no external NLP libraries required.

    Example
    -------
    >>> qd = QueryDecomposer()
    >>> result = qd.decompose("A bright yellow raincoat for a rainy day")
    >>> result['clothing_terms']
    ['raincoat']
    >>> result['color_terms']
    ['bright yellow']
    """

    def __init__(self) -> None:
        # Sort longest-first so "evening gown" is matched before "gown".
        self._clothing: list[str] = _build_sorted_terms(CLOTHING_CATEGORIES)
        self._colors: list[str] = _build_sorted_terms(COLOR_CATEGORIES)
        self._environments: list[str] = _build_sorted_terms(ENVIRONMENT_CATEGORIES)
        self._styles: list[str] = _build_sorted_terms(STYLE_CATEGORIES)

        # Pre-compile per-term word-boundary patterns for fast matching.
        self._clothing_pats: list[tuple[str, re.Pattern[str]]] = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self._clothing
        ]
        self._color_pats: list[tuple[str, re.Pattern[str]]] = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self._colors
            if t not in _COLOR_MODIFIERS  # modifiers are handled separately
        ]
        self._env_pats: list[tuple[str, re.Pattern[str]]] = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self._environments
        ]
        self._style_pats: list[tuple[str, re.Pattern[str]]] = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self._styles
        ]

    # ── Public API ───────────────────────────────────────────────────────────

    def decompose(self, query: str) -> dict[str, Any]:
        """Decompose *query* into structured fashion components.

        Parameters
        ----------
        query:
            Free-text user query, e.g.
            ``"A red tie and a white shirt in a formal setting"``.

        Returns
        -------
        dict with keys:
            ``original_query``, ``clothing_terms``, ``color_terms``,
            ``environment_terms``, ``style_terms``, ``sub_phrases``.
        """
        query_lower = query.lower().strip()
        logger.debug("Decomposing query: %r", query)

        clothing_terms = self._extract_terms(query_lower, self._clothing_pats)
        color_terms = self._extract_colors(query_lower)
        environment_terms = self._extract_terms(query_lower, self._env_pats)
        style_terms = self._extract_terms(query_lower, self._style_pats)
        sub_phrases = self.generate_sub_phrases(query)

        result: dict[str, Any] = {
            "original_query": query,
            "clothing_terms": clothing_terms,
            "color_terms": color_terms,
            "environment_terms": environment_terms,
            "style_terms": style_terms,
            "sub_phrases": sub_phrases,
        }
        logger.info(
            "Query decomposed - clothing=%s, colors=%s, env=%s, style=%s, "
            "sub_phrases=%d",
            clothing_terms, color_terms, environment_terms, style_terms,
            len(sub_phrases),
        )
        return result

    def generate_sub_phrases(self, query: str) -> list[str]:
        """Split *query* at conjunctions/prepositions into sub-phrases.

        Each sub-phrase is suitable for independent CLIP text encoding so
        the reranker can verify that *every* attribute is present in a
        candidate image (bottleneck scoring).

        Parameters
        ----------
        query:
            The original natural-language query.

        Returns
        -------
        list[str]
            De-duplicated, non-empty sub-phrases.  The full query is
            always included as the first element.

        Example
        -------
        >>> qd = QueryDecomposer()
        >>> qd.generate_sub_phrases(
        ...     "A red tie and a white shirt in a formal setting")
        ['A red tie and a white shirt in a formal setting',
         'A red tie', 'a white shirt', 'a formal setting']
        """
        # Always keep the full query as the first sub-phrase.
        phrases: list[str] = [query.strip()]

        # Split on conjunctions / prepositions.
        parts = _SPLIT_PATTERN.split(query)
        for part in parts:
            cleaned = part.strip().strip(".,;:!?")
            if cleaned and cleaned != query.strip():
                phrases.append(cleaned)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for p in phrases:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_terms(
        text: str,
        patterns: list[tuple[str, re.Pattern[str]]],
    ) -> list[str]:
        """Return all gazetteer terms whose pattern matches *text*.

        Because patterns are ordered longest-first, a match for
        ``"evening gown"`` prevents a second match for ``"gown"`` when
        the matched span is consumed.
        """
        found: list[str] = []
        remaining = text
        for term, pat in patterns:
            if pat.search(remaining):
                found.append(term)
                # Remove matched span so shorter overlapping terms won't
                # double-match (e.g. "tank top" should not also match "top").
                remaining = pat.sub("", remaining, count=1)
        return found

    def _extract_colors(self, text: str) -> list[str]:
        """Extract colour terms, handling modifier+base compounds.

        Matches compound colours like ``"bright yellow"`` as a single term
        while still catching standalone colours like ``"red"``.
        """
        found: list[str] = []
        remaining = text

        # 1) Try modifier + base colour compounds first.
        for modifier in _COLOR_MODIFIERS:
            for base_term, base_pat in self._color_pats:
                compound = f"{modifier} {base_term}"
                compound_pat = re.compile(
                    rf"\b{re.escape(compound)}\b", re.IGNORECASE,
                )
                if compound_pat.search(remaining):
                    found.append(compound)
                    remaining = compound_pat.sub("", remaining, count=1)

        # 2) Then match standalone base colours in the remaining text.
        for term, pat in self._color_pats:
            if pat.search(remaining):
                found.append(term)
                remaining = pat.sub("", remaining, count=1)

        return found


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    qd = QueryDecomposer()
    for test_query in [
        "A bright yellow raincoat for a rainy day",
        "A red tie and a white shirt in a formal setting",
        "Casual streetwear outfit with sneakers and a hoodie",
    ]:
        result = qd.decompose(test_query)
        print(f"\nQuery: {test_query}")
        for key, val in result.items():
            print(f"  {key}: {val}")
