#!/usr/bin/env python3
"""Explainability analyses:
1. Sentence-ablation attribution: black-box perturbation (the same principle LIME/SHAP use
	for models with no internals access), removing one sentence from the original at a time,
	re-simplifying, and measuring how much the output changes. Works identically across models.
2. TF-IDF faithfulness cross-check: an independent, deterministic (non-LLM) faithfulness
	signal to compare against the LLM-judge's faithfulness score.
3. DeepSeek per-token logprobs: model-internal confidence for DeepSeek.
4. Self-consistency: run each model multiple times on the same text and measure how much the
	output varies.
"""

from __future__ import annotations

import json
import time
import numpy as np
from difflib import SequenceMatcher

import config
from evaluate import compute_readability, split_sentences
from llm_clients import call_huggingface_with_logprobs, extract_json
from pipeline import load_texts
from prompts import SIMPLIFY_SYSTEM_PROMPT, build_simplify_user_prompt
from simplify import simplify

RESULTS_DIR = config.DATA_DIR / 'results'
MODEL_IDS = ['claude', 'deepseek', 'mistral', 'qwen']
N_CONSISTENCY_SAMPLES = 3

GERMAN_STOPWORDS = [
	'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
	'und', 'oder', 'ist', 'sind', 'war', 'waren', 'wird', 'werden', 'wurde', 'wurden',
	'für', 'von', 'mit', 'bei', 'im', 'in', 'auf', 'zu', 'zur', 'zum', 'sich', 'sie', 'er', 'es',
	'ich', 'wir', 'ihr', 'als', 'auch', 'nicht', 'kann', 'können', 'muss', 'müssen', 'so', 'wie',
	'an', 'am', 'um', 'nach', 'durch', 'über', 'unter', 'aus', 'diese', 'dieser', 'dieses'
]


def _text_similarity(a: str, b: str) -> float:
	"""Word-level similarity ratio between two texts (SequenceMatcher, 0-1).

	Parameters
		a: First text to compare (order doesn't matter, this is symmetric)
		b: Second text to compare

	Returns
		The similarity ratio, from 0 (no overlap) to 1 (identical)
	"""
	return SequenceMatcher(a=a.split(), b=b.split(), autojunk=False).ratio()


def _pick_subset_texts(n_per_level: int = 3) -> list[dict]:
	"""Texts with the most sentences per level.

	Parameters
		n_per_level: How many texts to keep per level (cantonal/federal/municipal)

	Returns
		The selected manifest entries, in level-then-sentence-count order

	Raises
		FileNotFoundError: propagated from load_texts() if the manifest or a text
			file it references is missing
	"""
	texts = load_texts()
	by_level: dict[str, list[dict]] = {'cantonal': [], 'federal': [], 'municipal': []}
	for t in texts:
		by_level[t['level']].append(t)
	subset = []
	for items in by_level.values():
		items.sort(key=lambda t: len(split_sentences(t['text'])), reverse=True)
		subset.extend(items[:n_per_level])
	return subset


def _load_main_results() -> list[dict]:
	"""Loads the core pipeline's results.json.

	Returns
		The results.json rows

	Raises
		FileNotFoundError: if data/results/results.json doesn't exist yet (run the
			core pipeline first)
	"""
	with open(RESULTS_DIR / 'results.json', encoding='utf-8') as f:
		return json.load(f)


# ---------------------------------------------------------------------------
# 1. Sentence-ablation attribution (all 4 models, model-agnostic perturbation)
# ---------------------------------------------------------------------------

def run_ablation_attribution(subset_texts: list[dict], main_results: list[dict]) -> list[dict]:
	"""For each (text, model) pair, removes one sentence at a time, re-simplifies, and
	scores how much the output changed relative to the full-text baseline. Saves progress
	to xai_ablation.json after every text (across all four models), not just at the end,
	since this is by far the most expensive of the four analyses (one simplify() call per
	sentence per model) and the one most likely to hit a transient API failure partway
	through, this way a crash only loses the current text's calls, not the whole subset's.

	Parameters
		subset_texts: The manifest entries to run ablation on (from load_texts(),
			usually the output of _pick_subset_texts())
		main_results: The core pipeline's results.json rows, used to look up each
			(text, model) pair's full, unablated simplified_text as the baseline

	Returns
		One row per (text, model) pair, each with the full sentence list and, per
		sentence, its similarity/attribution score relative to the full-text baseline;
		also incrementally checkpointed to xai_ablation.json

	Raises
		Whatever simplify() raises for a non-transient API error or exhausted retries
	"""
	baseline_by_key = {(r['text_id'], r['model_id']): r['simplified_text'] for r in main_results}
	rows = []
	for text_entry in subset_texts:
		sentences = split_sentences(text_entry['text'])
		if len(sentences) < 2:
			continue
		for model_id in MODEL_IDS:
			full_simplified = baseline_by_key.get((text_entry['id'], model_id))
			if not full_simplified:
				continue
			print(f"[ablation] {text_entry['id']} / {model_id}...")
			ablations = []
			for i in range(len(sentences)):
				if i > 0:
					time.sleep(1.5) # Small pause between calls
				ablated_text = ' '.join(s for j, s in enumerate(sentences) if j != i)
				ablated_result = simplify(model_id, ablated_text)
				similarity = _text_similarity(full_simplified, ablated_result.simplified_text)
				ablations.append({
					'removed_sentence_index': i,
					'removed_sentence': sentences[i],
					'similarity_to_full': round(similarity, 4),
					'attribution': round(1 - similarity, 4)
				})
			rows.append({
				'text_id': text_entry['id'],
				'model_id': model_id,
				'sentences': sentences,
				'full_simplified_text': full_simplified,
				'ablations': ablations
			})
		with open(RESULTS_DIR / 'xai_ablation.json', 'w', encoding='utf-8') as f:
			json.dump(rows, f, ensure_ascii=False, indent=2)
		print(f'  ...checkpointed {len(rows)} rows to xai_ablation.json')
	return rows


# ---------------------------------------------------------------------------
# 2. TF-IDF faithfulness cross-check
# ---------------------------------------------------------------------------

def run_tfidf_faithfulness_check(main_results: list[dict]) -> dict:
	"""Computes TF-IDF cosine similarity between original and simplified text for every
	row, and its correlation with the judge's faithfulness score.

	Parameters
		main_results: The core pipeline's results.json rows

	Returns
		A dict with 'pearson_r' (correlation between tfidf_similarity and
		judge_faithfulness), 'n' (row count), and 'per_row' (per-row detail)
	"""
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity

	vectorizer = TfidfVectorizer(lowercase=True, stop_words=GERMAN_STOPWORDS)
	corpus = []
	for r in main_results:
		corpus.append(r['original_text'])
		corpus.append(r['simplified_text'])
	vectorizer.fit(corpus)

	per_row = []
	for r in main_results:
		vecs = vectorizer.transform([r['original_text'], r['simplified_text']])
		sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
		per_row.append({
			'text_id': r['text_id'], 'model_id': r['model_id'],
			'tfidf_similarity': round(sim, 4),
			'judge_faithfulness': r['judge_faithfulness'],
			'is_self_judged': r['is_self_judged']
		})

	sims = np.array([p['tfidf_similarity'] for p in per_row])
	faiths = np.array([p['judge_faithfulness'] for p in per_row])
	correlation = float(np.corrcoef(sims, faiths)[0, 1])

	return {'pearson_r': round(correlation, 3), 'n': len(per_row), 'per_row': per_row}


# ---------------------------------------------------------------------------
# 3. DeepSeek per-token logprobs
# ---------------------------------------------------------------------------

def _group_tokens_into_words(tokens: list[dict]) -> list[dict]:
	"""Groups DeepSeek's subword logprob tokens into whole words with a mean logprob each.

	Parameters
		tokens: A list of {'token': str, 'logprob': float} dicts, one per subword
			token, as returned by call_huggingface_with_logprobs

	Returns
		A list of {'word': str, 'mean_logprob': float} dicts, in generation order
	"""
	words = []
	current = None
	for t in tokens:
		tok = t['token']
		if tok.startswith(' ') or current is None:
			if current:
				words.append(current)
			current = {'word': tok.lstrip(), 'logprobs': [t['logprob']]}
		else:
			current['word'] += tok
			current['logprobs'].append(t['logprob'])
	if current:
		words.append(current)
	for w in words:
		w['mean_logprob'] = round(sum(w['logprobs']) / len(w['logprobs']), 4)
		del w['logprobs']
	return words


def _slice_tokens_for_substring(tokens: list[dict], substring: str) -> list[dict]:
	"""Slices the token stream down to just the tokens covering `substring` (e.g. the
	simplified_text value, not the surrounding JSON scaffolding).

	Parameters
		tokens: A list of {'token': str, 'logprob': float} dicts, in generation order
		substring: The exact text to locate within the tokens once reconstructed

	Returns
		The subset of `tokens` covering `substring`, in original order, falling back
		to the full `tokens` list if `substring` isn't found in the reconstructed text
	"""
	reconstructed = ''.join(t['token'] for t in tokens)
	needle = substring[:60] if len(substring) > 60 else substring
	start = reconstructed.find(needle)
	if start == -1:
		return tokens # Fall back to full stream rather than dropping the row
	end = start + len(substring)
	cursor = 0
	relevant = []
	for t in tokens:
		tok_len = len(t['token'])
		if cursor + tok_len > start and cursor < end:
			relevant.append(t)
		cursor += tok_len
		if cursor >= end:
			break
	return relevant or tokens


def run_deepseek_logprobs(texts: list[dict]) -> list[dict]:
	"""Captures per-token logprobs from DeepSeek for every text and groups them into words.

	Parameters
		texts: The manifest entries to run (from load_texts())

	Returns
		One row per text that returned logprobs (texts where the backend gave none
		are skipped): 'text_id', 'simplified_text', and 'words' (see
		_group_tokens_into_words)

	Raises
		Whatever call_huggingface_with_logprobs raises for a non-transient API error
		or exhausted retries
	"""
	model_name = config.MODELS['deepseek'].model_name
	rows = []
	for text_entry in texts:
		print(f"[logprobs] {text_entry['id']}...")
		raw, tokens = call_huggingface_with_logprobs(
			model_name, SIMPLIFY_SYSTEM_PROMPT, build_simplify_user_prompt(text_entry['text'])
		)
		if not tokens:
			continue
		try:
			simplified_text = extract_json(raw)['simplified_text']
		except Exception:
			simplified_text = raw
		relevant_tokens = _slice_tokens_for_substring(tokens, simplified_text)
		rows.append({
			'text_id': text_entry['id'],
			'simplified_text': simplified_text,
			'words': _group_tokens_into_words(relevant_tokens)
		})
	return rows


# ---------------------------------------------------------------------------
# 4. Self-consistency (all 4 models, repeated sampling)
# ---------------------------------------------------------------------------

def run_self_consistency(subset_texts: list[dict]) -> list[dict]:
	"""Runs each model N_CONSISTENCY_SAMPLES times per text and measures output
	similarity and WSTF variance across the samples. Saves progress to
	xai_self_consistency.json after every text, same reasoning as
	run_ablation_attribution: several API calls per text, checkpoint so a crash
	partway through doesn't lose everything already paid for.

	Parameters
		subset_texts: The manifest entries to run (from load_texts(), usually the
			output of _pick_subset_texts())

	Returns
		One row per (text, model) pair: 'n_samples', 'mean_pairwise_similarity'
		across the samples, 'wstf_std', and the raw 'wstf_values', also
		incrementally checkpointed to xai_self_consistency.json

	Raises
		Whatever simplify() raises for a non-transient API error or exhausted retries
	"""
	rows = []
	for text_entry in subset_texts:
		for model_id in MODEL_IDS:
			print(f"[consistency] {text_entry['id']} / {model_id}...")
			samples = []
			for sample_i in range(N_CONSISTENCY_SAMPLES):
				if sample_i > 0:
					time.sleep(1.5) # Small pause between calls
				result = simplify(model_id, text_entry['text'])
				samples.append({
					'simplified_text': result.simplified_text,
					'wstf_after': compute_readability(result.simplified_text).wstf
				})
			sims = [
				_text_similarity(samples[i]['simplified_text'], samples[j]['simplified_text'])
				for i in range(len(samples)) for j in range(i + 1, len(samples))
			]
			wstf_values = [s['wstf_after'] for s in samples]
			rows.append({
				'text_id': text_entry['id'],
				'model_id': model_id,
				'n_samples': N_CONSISTENCY_SAMPLES,
				'mean_pairwise_similarity': round(sum(sims) / len(sims), 4),
				'wstf_std': round(float(np.std(wstf_values)), 4),
				'wstf_values': wstf_values
			})
		with open(RESULTS_DIR / 'xai_self_consistency.json', 'w', encoding='utf-8') as f:
			json.dump(rows, f, ensure_ascii=False, indent=2)
		print(f'  ...checkpointed {len(rows)} rows to xai_self_consistency.json')
	return rows


def main() -> None:
	"""CLI entry point: runs all four XAI analyses and saves each to its own JSON file."""
	main_results = _load_main_results()
	subset = _pick_subset_texts(n_per_level=3)
	print(f'Subset for ablation/consistency: {[t["id"] for t in subset]}\n')

	print('=== 1. Sentence-ablation attribution ===')
	ablation_rows = run_ablation_attribution(subset, main_results)
	with open(RESULTS_DIR / 'xai_ablation.json', 'w', encoding='utf-8') as f:
		json.dump(ablation_rows, f, ensure_ascii=False, indent=2)
	print(f'Saved {len(ablation_rows)} rows to xai_ablation.json\n')

	print('=== 2. TF-IDF faithfulness cross-check ===')
	tfidf_result = run_tfidf_faithfulness_check(main_results)
	with open(RESULTS_DIR / 'xai_tfidf_faithfulness.json', 'w', encoding='utf-8') as f:
		json.dump(tfidf_result, f, ensure_ascii=False, indent=2)
	print(f"Pearson r (tfidf_similarity vs. judge_faithfulness), n={tfidf_result['n']}: {tfidf_result['pearson_r']}\n")

	print('=== 3. DeepSeek per-token logprobs ===')
	all_texts = load_texts()
	logprob_rows = run_deepseek_logprobs(all_texts)
	with open(RESULTS_DIR / 'xai_deepseek_logprobs.json', 'w', encoding='utf-8') as f:
		json.dump(logprob_rows, f, ensure_ascii=False, indent=2)
	print(f'Saved {len(logprob_rows)} rows to xai_deepseek_logprobs.json\n')

	print('=== 4. Self-consistency ===')
	consistency_rows = run_self_consistency(subset)
	with open(RESULTS_DIR / 'xai_self_consistency.json', 'w', encoding='utf-8') as f:
		json.dump(consistency_rows, f, ensure_ascii=False, indent=2)
	print(f'Saved {len(consistency_rows)} rows to xai_self_consistency.json')


if __name__ == '__main__':
	main()