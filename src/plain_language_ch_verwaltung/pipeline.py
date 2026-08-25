#!/usr/bin/env python3
"""End-to-end pipeline: load texts, simplify with each model, tag diffs, score readability
and faithfulness, and write results to data/results/.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import config
from .diffing import compute_diff_tags
from .evaluate import compute_readability, judge
from .simplify import simplify

TEXTS_DIR = config.DATA_DIR / 'texts'
TEXTS_MANIFEST = TEXTS_DIR / 'manifest.json'
RESULTS_DIR = config.DATA_DIR / 'results'


def load_texts() -> list[dict]:
	"""Loads every source text listed in data/texts/manifest.json."""
	with open(TEXTS_MANIFEST, encoding='utf-8') as f:
		manifest = json.load(f)

	texts = []
	for entry in manifest:
		with open(TEXTS_DIR / entry['file'], encoding='utf-8') as f:
			texts.append({**entry, 'text': f.read().strip()})
	return texts


def run(model_ids: list[str] | None = None) -> list[dict]:
	"""Runs the full pipeline (simplify, score readability, diff-tag, judge) for every
	(text x model) pair.

	Parameters
		model_ids: Which models to run, as keys into config.MODELS (defaults to
			every registered model in config.MODELS when not given)
	"""
	model_ids = model_ids or list(config.MODELS)
	texts = load_texts()
	rows = []

	for text_entry in texts:
		original = text_entry['text']
		readability_before = compute_readability(original)

		for model_id in model_ids:
			print(f"[{text_entry['id']}] simplifying with {model_id}...")
			t0 = time.time()
			result = simplify(model_id, original)
			latency_s = round(time.time() - t0, 2)

			readability_after = compute_readability(result.simplified_text)
			diff_tags = compute_diff_tags(original, result.simplified_text)
			scores = judge(original, result.simplified_text)

			rows.append({
				'text_id': text_entry['id'],
				'title': text_entry.get('title'),
				'level': text_entry['level'],
				'source_url': text_entry.get('source_url'),
				'model_id': model_id,
				'is_self_judged': model_id == config.JUDGE_MODEL_ID,
				'latency_s': latency_s,
				'wstf_before': readability_before.wstf,
				'wstf_after': readability_after.wstf,
				'lix_before': readability_before.lix,
				'lix_after': readability_after.lix,
				'avg_sentence_len_before': readability_before.avg_sentence_length,
				'avg_sentence_len_after': readability_after.avg_sentence_length,
				'sentences_before': diff_tags.sentences_before,
				'sentences_after': diff_tags.sentences_after,
				'passive_before': diff_tags.passive_constructions_before,
				'passive_after': diff_tags.passive_constructions_after,
				'jargon_removed': diff_tags.jargon_terms_removed,
				'jargon_remaining': diff_tags.jargon_terms_remaining,
				'lexical_substitutions': diff_tags.lexical_substitutions,
				'n_lexical_substitutions': len(diff_tags.lexical_substitutions),
				'judge_faithfulness': scores.get('faithfulness'),
				'judge_simplicity': scores.get('simplicity'),
				'judge_fluency': scores.get('fluency'),
				'judge_comment': scores.get('comment'),
				'rationale': result.rationale,
				'original_text': original,
				'simplified_text': result.simplified_text
			})

	return rows


def save_results(rows: list[dict]) -> None:
	"""Writes results to results.json (full) and results.csv (flattened, for quick
	comparison across models)."""
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	with open(RESULTS_DIR / 'results.json', 'w', encoding='utf-8') as f:
		json.dump(rows, f, ensure_ascii=False, indent=2)

	df = pd.DataFrame(rows)
	flat_cols = ['rationale', 'simplified_text', 'original_text', 'lexical_substitutions']
	df.drop(columns=flat_cols).to_csv(RESULTS_DIR / 'results.csv', index=False)


def build_arg_parser() -> argparse.ArgumentParser:
	"""Builds the CLI argument parser for plz-run."""
	parser = argparse.ArgumentParser(description='Compare LLMs on Swiss admin-text simplification.')
	parser.add_argument(
		'--models', nargs='*', default=None,
		help='Subset of model ids to run, e.g. --models claude gpt'
	)
	return parser


def main() -> None:
	"""CLI entry point: runs the pipeline and saves results."""
	args = build_arg_parser().parse_args()

	rows = run(args.models)
	save_results(rows)
	print(f'Saved {len(rows)} results to {RESULTS_DIR}')


if __name__ == '__main__':
	main()