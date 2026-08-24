#!/usr/bin/env python3
"""Thin, provider-agnostic wrappers around each backend's chat-completion API.

Hugging Face Inference Providers exposes an OpenAI-compatible endpoint that routes to a
third-party backend (Together, Novita, Cerebras, ...) under a single HF token. The backend
is pinned explicitly via HF_PROVIDER (see config.py) rather than left on "auto" selection.
Mistral is called directly against Mistral's own (also OpenAI-compatible) API instead, to
use existing Mistral credits rather than HF's.
"""

from __future__ import annotations

import functools
import json
import random
import re
import time

from . import config

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2


def _retry_on_transient_error(fn):
	"""Retries on rate-limit (429) and server errors (5xx) with exponential backoff plus
	jitter - hit in practice during xai.py's self-consistency phase, which fires several
	back-to-back calls at the same hosted backend and can trip its rate limiting. Anything
	else (bad request, auth, etc.) is a real problem and is re-raised immediately, since
	retrying it would just fail the same way again.
	"""
	@functools.wraps(fn)
	def wrapper(*args, **kwargs):
		for attempt in range(MAX_RETRIES + 1):
			try:
				return fn(*args, **kwargs)
			except Exception as e:
				status_code = getattr(e, 'status_code', None)
				is_transient = status_code == 429 or (status_code is not None and status_code >= 500)
				if not is_transient or attempt == MAX_RETRIES:
					raise
				wait = BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1)
				print(f'  transient error ({status_code}) from {fn.__name__}, retrying in {wait:.1f}s '
					f'(attempt {attempt + 1}/{MAX_RETRIES})...')
				time.sleep(wait)
	return wrapper


@_retry_on_transient_error
def call_anthropic(model_name: str, system_prompt: str, user_prompt: str) -> str:
	import anthropic

	client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
	response = client.messages.create(
		model=model_name,
		# Generous headroom: this model returns extended-thinking blocks ahead of the text
		# block, and both count against max_tokens - 2048 was truncating the JSON output.
		max_tokens=8192,
		system=system_prompt,
		messages=[{'role': 'user', 'content': user_prompt}]
	)
	return next(block.text for block in response.content if block.type == 'text')


@_retry_on_transient_error
def call_huggingface(model_name: str, system_prompt: str, user_prompt: str) -> str:
	from openai import OpenAI

	client = OpenAI(api_key=config.HF_TOKEN, base_url='https://router.huggingface.co/v1')
	response = client.chat.completions.create(
		model=f'{model_name}:{config.HF_PROVIDER}',
		messages=[
			{'role': 'system', 'content': system_prompt},
			{'role': 'user', 'content': user_prompt}
		]
	)
	return response.choices[0].message.content


@_retry_on_transient_error
def call_huggingface_with_logprobs(model_name: str, system_prompt: str, user_prompt: str) -> tuple[str, list[dict] | None]:
	"""Like call_huggingface, but also requests per-token logprobs. Confirmed empirically
	(2026-08-24) that only DeepSeek-V3 on the novita backend actually returns them - Qwen
	accepts the parameter but silently returns logprobs=None. Returns (text, token_list) where
	token_list is None if the backend didn't supply logprobs.
	"""
	from openai import OpenAI

	client = OpenAI(api_key=config.HF_TOKEN, base_url='https://router.huggingface.co/v1')
	response = client.chat.completions.create(
		model=f'{model_name}:{config.HF_PROVIDER}',
		messages=[
			{'role': 'system', 'content': system_prompt},
			{'role': 'user', 'content': user_prompt}
		],
		logprobs=True,
		top_logprobs=1
	)
	choice = response.choices[0]
	tokens = None
	if choice.logprobs and choice.logprobs.content:
		tokens = [{'token': t.token, 'logprob': t.logprob} for t in choice.logprobs.content]
	return choice.message.content, tokens


@_retry_on_transient_error
def call_mistral(model_name: str, system_prompt: str, user_prompt: str) -> str:
	from openai import OpenAI

	client = OpenAI(api_key=config.MISTRAL_API_KEY, base_url='https://api.mistral.ai/v1')
	response = client.chat.completions.create(
		model=model_name,
		messages=[
			{'role': 'system', 'content': system_prompt},
			{'role': 'user', 'content': user_prompt}
		]
	)
	return response.choices[0].message.content


def call_model(model_id: str, system_prompt: str, user_prompt: str) -> str:
	model = config.MODELS[model_id]

	if model.provider == 'anthropic':
		return call_anthropic(model.model_name, system_prompt, user_prompt)
	if model.provider == 'huggingface':
		return call_huggingface(model.model_name, system_prompt, user_prompt)
	if model.provider == 'mistral':
		return call_mistral(model.model_name, system_prompt, user_prompt)
	raise ValueError(f'Unknown provider: {model.provider}')


def extract_json(raw: str) -> dict:
	cleaned = raw.strip()
	if cleaned.startswith('```'):
		cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.MULTILINE).strip()
	# raw_decode (rather than loads) parses just the first complete JSON value and ignores
	# anything trailing - some models (observed with Qwen as judge) append a self-correction
	# or second JSON block after the first one. strict=False additionally tolerates literal
	# newlines inside string values, which some models (observed with Mistral) emit instead
	# of escaping as \n.
	return json.JSONDecoder(strict=False).raw_decode(cleaned)[0]
