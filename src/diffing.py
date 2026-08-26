#!/usr/bin/env python3
"""Model-agnostic, rule-based diff tagging.

Complements each model's self-reported rationale (see simplify.SimplificationResult).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from evaluate import split_sentences, tokenize_words

# Non-exhausive list of common Swiss administrative/legal jargon; extend as the corpus grows
JARGON_TERMS = [
	'einsprache',
	'verfügung',
	'rekurs',
	'beschwerdeführer',
	'gesuch',
	'vollzug',
	'vollzugsverordnung',
	'massgebend',
	'zuständig',
	'gemäss',
	'widerspruch',
	'fristgerecht',
	'verordnung',
	'erlass',
	'kognition',
	'auflagefrist'
]

_PASSIVE_RE = re.compile(r'\b(wird|werden|wurde|wurden|worden)\b[^.!?]{0,40}?\b\w+(t|en)\b', re.IGNORECASE)


def _contains_term(text: str, term: str) -> bool:
	"""Left-boundary containment check for a jargon term: `term` must start at a word
	boundary, but nothing is required after it, so German noun inflection (zuständig ->
	zuständige/zuständigen) still matches.

	Returns
		True if `term` occurs in `text` starting at a word boundary

	Note
		Still false-positives on same-root, different-meaning words that happen to also
		start at a word boundary (e.g. 'gesucht', past participle of 'suchen', vs. the
		noun 'Gesuch'). A robuster solution requires a lemmatizer, which is out of scope for a
		keyword-based heuristic.
	"""
	return re.search(rf'\b{re.escape(term)}', text) is not None


def _count_passive_constructions(text: str) -> int:
	"""Counts regex matches for the passive-voice heuristic (see _PASSIVE_RE).

	Returns
		The number of passive-construction matches found
	"""
	return len(_PASSIVE_RE.findall(text))


@dataclass
class DiffTags:
	"""Model-agnostic diff between an original and simplified text: sentence and
	passive-voice counts, jargon removed vs. remaining, and lexical substitutions."""

	sentences_before: int
	sentences_after: int
	avg_sentence_len_before: float
	avg_sentence_len_after: float
	passive_constructions_before: int
	passive_constructions_after: int
	jargon_terms_removed: list[str] = field(default_factory=list)
	jargon_terms_remaining: list[str] = field(default_factory=list)
	lexical_substitutions: list[tuple[str, str]] = field(default_factory=list)


def compute_diff_tags(original: str, simplified: str) -> DiffTags:
	"""Computes DiffTags for one (original, simplified) pair.

	Parameters
		original: The source Verwaltungstext, before simplification
		simplified: The same text after a model has simplified it

	Returns
		The computed DiffTags for this pair
	"""
	orig_sentences = split_sentences(original)
	simp_sentences = split_sentences(simplified)
	orig_words = tokenize_words(original.lower())
	simp_words = tokenize_words(simplified.lower())

	avg_before = len(orig_words) / max(1, len(orig_sentences))
	avg_after = len(simp_words) / max(1, len(simp_sentences))

	orig_lower = original.lower()
	simp_lower = simplified.lower()
	removed = [t for t in JARGON_TERMS if _contains_term(orig_lower, t) and not _contains_term(simp_lower, t)]
	remaining = [t for t in JARGON_TERMS if _contains_term(orig_lower, t) and _contains_term(simp_lower, t)]

	substitutions: list[tuple[str, str]] = []
	matcher = SequenceMatcher(a=orig_words, b=simp_words, autojunk=False)
	for tag, i1, i2, j1, j2 in matcher.get_opcodes():
		if tag == 'replace':
			substitutions.append((' '.join(orig_words[i1:i2]), ' '.join(simp_words[j1:j2])))

	return DiffTags(
		sentences_before=len(orig_sentences),
		sentences_after=len(simp_sentences),
		avg_sentence_len_before=round(avg_before, 2),
		avg_sentence_len_after=round(avg_after, 2),
		passive_constructions_before=_count_passive_constructions(original),
		passive_constructions_after=_count_passive_constructions(simplified),
		jargon_terms_removed=removed,
		jargon_terms_remaining=remaining,
		lexical_substitutions=substitutions
	)