#!/usr/bin/env python3
"""Readability metrics and LLM-as-judge scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from wtpsplit import SaT
from zix.understandability import get_cefr, get_zix

import config
from llm_clients import call_model, extract_json
from prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

_VOWELS = 'aeiouyäöü'
_WORD_RE = re.compile(r'[A-Za-zÄÖÜäöüß]+')
_SAT = SaT('sat-3l-sm')


def _count_syllables(word: str) -> int:
	"""Counts vowel clusters in a word as a syllable-count proxy.

	Returns
		The syllable count (minimum 1)
	"""
	return max(1, len(re.findall(rf'[{_VOWELS}]+', word.lower())))


def tokenize_words(text: str) -> list[str]:
	"""Splits text into German-alphabet word tokens.

	Returns
		The list of word tokens, in order of appearance
	"""
	return _WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
	"""Splits text into sentences with wtpsplit's SaT model. Unlike a naive '.', '!', '?'
	regex, this correctly keeps legal-citation abbreviations (Art., Abs., lit.) and
	ordinals (e.g. '10. Schuljahr') from triggering a false sentence boundary.

	Returns
		The list of non-empty sentences, in order of appearance
	"""
	return [s.strip() for s in _SAT.split(text.strip()) if s.strip()]


@dataclass
class ReadabilityScores:
	"""Word/sentence counts and three German readability metrics (WSTF, LIX, ZIX) for one text."""

	n_words: int
	n_sentences: int
	avg_sentence_length: float
	wstf: float # Wiener Sachtextformel: higher = harder to read
	lix: float # Läsbarhetsindex: higher = harder to read
	zix: float | None # Zürcher Verständlichkeitsindex: higher = easier to read, -10 to 10
	cefr: str | None # CEFR level implied by zix (A1 easiest to C2 hardest), or None if zix is None


def compute_readability(text: str) -> ReadabilityScores:
	"""Computes ReadabilityScores for a text.

	Returns
		The computed ReadabilityScores
	"""
	words = tokenize_words(text)
	sentences = split_sentences(text)
	n_words = len(words)

	if n_words == 0:
		return ReadabilityScores(0, len(sentences), 0.0, 0.0, 0.0, None, None)

	n_sentences = max(1, len(sentences))
	avg_sentence_length = n_words / n_sentences
	pct_long = 100 * sum(1 for w in words if len(w) > 6) / n_words
	pct_poly = 100 * sum(1 for w in words if _count_syllables(w) >= 3) / n_words
	pct_mono = 100 * sum(1 for w in words if _count_syllables(w) == 1) / n_words

	wstf = 0.1935 * pct_poly + 0.1672 * avg_sentence_length + 0.1297 * pct_long - 0.0327 * pct_mono - 0.875
	lix = avg_sentence_length + pct_long
	zix = get_zix(text)
	cefr = get_cefr(zix) if zix is not None else None

	return ReadabilityScores(
		n_words=n_words, n_sentences=n_sentences,
		avg_sentence_length=round(avg_sentence_length, 2),
		wstf=round(wstf, 2), lix=round(lix, 2),
		zix=round(zix, 2) if zix is not None else None, cefr=cefr
	)


def judge(original: str, simplified: str, judge_model_id: str | None = None) -> dict:
	"""Scores a simplification's faithfulness, simplicity, and fluency via an LLM-as-judge call.

	Parameters
		original: The source Verwaltungstext, before simplification
		simplified: The same text after a model has simplified it
		judge_model_id: Which model judges, as a key into config.MODELS (defaults
			to config.JUDGE_MODEL_ID when not given)

	Returns
		A dict with 'faithfulness', 'simplicity', 'fluency' (each 1-5, or None if
		the judge's reply didn't parse as JSON) and 'comment'

	Raises
		Whatever call_model raises for a non-transient error or exhausted retries
		(see llm_clients.call_model)
	"""
	model_id = judge_model_id or config.JUDGE_MODEL_ID
	raw = call_model(model_id, JUDGE_SYSTEM_PROMPT, build_judge_user_prompt(original, simplified))
	try:
		return extract_json(raw)
	except json.JSONDecodeError as e:
		print(f'  WARNING: judge ({model_id}) did not return valid JSON ({e}); scores set to None')
		return {'faithfulness': None, 'simplicity': None, 'fluency': None, 'comment': raw}