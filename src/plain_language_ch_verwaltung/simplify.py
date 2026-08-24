#!/usr/bin/env python3
"""Model backends that simplify a text and return the model's own rationale for its edits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm_clients import call_model, extract_json
from .prompts import SIMPLIFY_SYSTEM_PROMPT, build_simplify_user_prompt


@dataclass
class SimplificationResult:
	model_id: str
	simplified_text: str
	rationale: list[dict] = field(default_factory=list)
	raw_response: str = ''


def simplify(model_id: str, text: str) -> SimplificationResult:
	raw = call_model(model_id, SIMPLIFY_SYSTEM_PROMPT, build_simplify_user_prompt(text))

	try:
		parsed = extract_json(raw)
		simplified_text = parsed['simplified_text']
		rationale = parsed.get('rationale', [])
	except (json.JSONDecodeError, KeyError) as e:
		# If model didn't follow the JSON contract, fall back to using the raw reply as-is.
		# This is silent data corruption if unnoticed - readability/diff metrics would run
		# against a raw (possibly truncated or markdown-wrapped) blob instead of clean text.
		print(f'  WARNING: {model_id} did not return valid JSON ({e}); using raw reply as-is')
		simplified_text = raw
		rationale = []

	return SimplificationResult(model_id=model_id, simplified_text=simplified_text, rationale=rationale, raw_response=raw)
