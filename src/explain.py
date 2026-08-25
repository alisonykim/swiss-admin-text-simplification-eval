#!/usr/bin/env python3
"""SHAP-based feature attribution over the collected results.

Note: This module does not cover explainability of the four LLMs themselves
(none of them expose model internals). Instead, it explains a small proxy model
trained on the model-agnostic diff-tag features already computed by diffing.py,
to answer: of the structural changes a simplification makes (jargon removed,
sentences split, passive voice reduced, words substituted), which ones actually predict
a bigger readability improvement (WSTF drop)? Both the features and the target are
deterministic/formula-based, ensuring that the explainability layer stays independent
of the evaluated models.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

import config

RESULTS_DIR = config.DATA_DIR / 'results'
MODEL_IDS = ['claude', 'deepseek', 'mistral', 'qwen']

FEATURE_LABELS = {
	'n_jargon_removed': 'Fachbegriffe entfernt',
	'n_jargon_remaining': 'Fachbegriffe geblieben',
	'n_lexical_substitutions': 'Wortersetzungen',
	'passive_delta': 'Passiv-Reduktion',
	'sentence_delta': 'Satz-Aufteilung',
	'is_claude': 'Modell: Claude',
	'is_deepseek': 'Modell: DeepSeek',
	'is_mistral': 'Modell: Mistral',
	'is_qwen': 'Modell: Qwen'
}


def build_features(rows: list[dict]) -> pd.DataFrame:
	"""Builds the diff-tag feature matrix (jargon/substitution/passive/sentence deltas,
	one-hot model id) and the WSTF-improvement target.

	Parameters
		rows: The core pipeline's results.json rows
	"""
	records = []
	for r in rows:
		rec = {
			'n_jargon_removed': len(r['jargon_removed']),
			'n_jargon_remaining': len(r['jargon_remaining']),
			'n_lexical_substitutions': r['n_lexical_substitutions'],
			'passive_delta': r['passive_before'] - r['passive_after'],
			'sentence_delta': r['sentences_after'] - r['sentences_before']
		}
		for m in MODEL_IDS:
			rec[f'is_{m}'] = int(r['model_id'] == m)
		rec['wstf_improvement'] = r['wstf_before'] - r['wstf_after']
		rec['text_id'] = r['text_id']
		rec['model_id'] = r['model_id']
		records.append(rec)
	return pd.DataFrame.from_records(records)


def run() -> dict:
	"""Fits a small GradientBoostingRegressor on the diff-tag features, cross-validates
	it, and computes SHAP values. Does not generalize - see module docstring."""
	with open(RESULTS_DIR / 'results.json', encoding='utf-8') as f:
		rows = json.load(f)

	df = build_features(rows)
	feature_cols = list(FEATURE_LABELS.keys())
	X = df[feature_cols]
	y = df['wstf_improvement']

	model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=0)
	model.fit(X, y)

	cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')

	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(X)
	base_value = np.asarray(explainer.expected_value).reshape(-1)[0]

	mean_abs_shap = np.abs(shap_values).mean(axis=0)
	importance = sorted(zip(feature_cols, mean_abs_shap), key=lambda t: t[1], reverse=True)

	per_row = []
	for i, r in enumerate(rows):
		per_row.append({
			'text_id': r['text_id'],
			'model_id': r['model_id'],
			'wstf_improvement': float(y.iloc[i]),
			'shap': {col: float(shap_values[i][j]) for j, col in enumerate(feature_cols)},
			'base_value': float(base_value)
		})

	summary = {
		'n_rows': len(rows),
		'cv_r2_mean': float(cv_r2.mean()),
		'cv_r2_std': float(cv_r2.std()),
		'feature_labels': FEATURE_LABELS,
		'feature_importance': [{'feature': f, 'mean_abs_shap': float(v)} for f, v in importance],
		'per_row': per_row
	}

	with open(RESULTS_DIR / 'shap_analysis.json', 'w', encoding='utf-8') as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	return summary


def main() -> None:
	"""CLI entry point: runs the SHAP analysis and prints a feature-importance summary."""
	summary = run()
	print(f"Trained on {summary['n_rows']} rows. 5-fold CV R²: "
		f"{summary['cv_r2_mean']:.2f} ± {summary['cv_r2_std']:.2f} (negative - does not generalize, see module docstring)")
	print('\nFeature importance (mean |SHAP| on WSTF improvement):')
	for item in summary['feature_importance']:
		label = summary['feature_labels'][item['feature']]
		print(f"  {label:28s} {item['mean_abs_shap']:.3f}")
	print(f"\nSaved full per-row SHAP values to {RESULTS_DIR / 'shap_analysis.json'}")


if __name__ == '__main__':
	main()