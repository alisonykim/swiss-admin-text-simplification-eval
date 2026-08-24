#!/usr/bin/env python3
"""Model-agnostic, rule-based diff tagging.

Complements each model's self-reported rationale (see simplify.SimplificationResult) with an
independent, reproducible signal that doesn't depend on a model accurately describing its own
edits - the two are meant to be read side by side, not merged into one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .evaluate import split_sentences, tokenize_words

# Non-exhausive seed list of common Swiss administrative/legal jargon; extend as the corpus grows
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


def _count_passive_constructions(text: str) -> int:
	return len(_PASSIVE_RE.findall(text))


@dataclass
class DiffTags:
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
	orig_sentences = split_sentences(original)
	simp_sentences = split_sentences(simplified)
	orig_words = tokenize_words(original.lower())
	simp_words = tokenize_words(simplified.lower())

	avg_before = len(orig_words) / max(1, len(orig_sentences))
	avg_after = len(simp_words) / max(1, len(simp_sentences))

	orig_lower = original.lower()
	simp_lower = simplified.lower()
	removed = [t for t in JARGON_TERMS if t in orig_lower and t not in simp_lower]
	remaining = [t for t in JARGON_TERMS if t in orig_lower and t in simp_lower]

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