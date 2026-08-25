#!/usr/bin/env python3
import pytest

from llm_clients import extract_json


def test_extract_json_handles_markdown_fence():
	raw = '```json\n{"simplified_text": "Hallo", "rationale": []}\n```'
	parsed = extract_json(raw)
	assert parsed['simplified_text'] == 'Hallo'


def test_extract_json_handles_plain_json():
	raw = '{"simplified_text": "Hallo", "rationale": []}'
	parsed = extract_json(raw)
	assert parsed['simplified_text'] == 'Hallo'


def test_extract_json_raises_on_garbage():
	with pytest.raises(Exception):
		extract_json('not json at all')
