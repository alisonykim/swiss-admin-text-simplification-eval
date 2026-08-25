#!/usr/bin/env python3
"""Model backends that simplify a text and return the model's own rationale for its edits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm_clients import call_model, extract_json
from prompts import SIMPLIFY_SYSTEM_PROMPT, build_simplify_user_prompt


@dataclass
class SimplificationResult:
	"""One model's simplified text, its self-reported rationale, and the raw API response."""

	model_id: str
	simplified_text: str
	rationale: list[dict] = field(default_factory=list)
	raw_response: str = ''


def simplify(model_id: str, text: str) -> SimplificationResult:
	"""Simplifies a text with the given model and parses its JSON response into a
	SimplificationResult.

	Parameters
		model_id: A key into config.MODELS, e.g. 'claude', 'qwen', 'mistral', 'deepseek'
		text: The source Verwaltungstext to simplify
	"""
	raw = call_model(model_id, SIMPLIFY_SYSTEM_PROMPT, build_simplify_user_prompt(text))

	try:
		parsed = extract_json(raw)
		simplified_text = parsed['simplified_text']
		rationale = parsed.get('rationale', [])
	except (json.JSONDecodeError, KeyError) as e:
		print(f'  WARNING: {model_id} did not return valid JSON ({e}); using raw reply as-is')
		simplified_text = raw
		rationale = []

	return SimplificationResult(model_id=model_id, simplified_text=simplified_text, rationale=rationale, raw_response=raw)