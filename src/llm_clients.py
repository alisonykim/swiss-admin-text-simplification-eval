#!/usr/bin/env python3
"""Thin, provider-agnostic wrappers around each backend's chat-completion API."""

from __future__ import annotations

import json
import random
import re
import time

import anthropic
from openai import OpenAI

import config

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2


def call_anthropic(model_name: str, system_prompt: str, user_prompt: str) -> str:
	"""Calls Claude via the Anthropic Messages API. Retries on rate-limit (429) or
	server (5xx) errors, with a wait that doubles each attempt - hit in practice during
	xai.py's self-consistency phase, which fires several back-to-back calls at the same
	hosted backend and can trip its rate limiting. Any other error (bad request, auth,
	etc.) is a real problem and is re-raised immediately.

	Parameters
		model_name: The Claude model id, e.g. 'claude-sonnet-5' (see config.MODELS)
		system_prompt: The system-role instructions (see prompts.py)
		user_prompt: The user-role message content
	"""
	client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

	for attempt in range(MAX_RETRIES + 1):
		try:
			response = client.messages.create(
				model=model_name,
				max_tokens=8192,
				system=system_prompt,
				messages=[{'role': 'user', 'content': user_prompt}]
			)
			return next(block.text for block in response.content if block.type == 'text')
		except Exception as e:
			status_code = getattr(e, 'status_code', None)
			is_transient = status_code == 429 or (status_code is not None and status_code >= 500)
			if not is_transient or attempt == MAX_RETRIES:
				raise
			wait = BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1)
			print(f'  transient error ({status_code}), retrying in {wait:.1f}s '
				f'(attempt {attempt + 1}/{MAX_RETRIES})...')
			time.sleep(wait)


def call_huggingface(model_name: str, system_prompt: str, user_prompt: str) -> str:
	"""Calls a model via Hugging Face Inference Providers (OpenAI-compatible). Retries
	on rate-limit (429) or server (5xx) errors; see call_anthropic for why.

	Parameters
		model_name: The Hugging Face repo id, e.g. 'Qwen/Qwen2.5-72B-Instruct' (see
			config.MODELS); gets combined with config.HF_PROVIDER below to pin the
			routed backend
		system_prompt: The system-role instructions
		user_prompt: The user-role message content
	"""
	client = OpenAI(api_key=config.HF_TOKEN, base_url='https://router.huggingface.co/v1')

	for attempt in range(MAX_RETRIES + 1):
		try:
			response = client.chat.completions.create(
				model=f'{model_name}:{config.HF_PROVIDER}',
				messages=[
					{'role': 'system', 'content': system_prompt},
					{'role': 'user', 'content': user_prompt}
				]
			)
			return response.choices[0].message.content
		except Exception as e:
			status_code = getattr(e, 'status_code', None)
			is_transient = status_code == 429 or (status_code is not None and status_code >= 500)
			if not is_transient or attempt == MAX_RETRIES:
				raise
			wait = BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1)
			print(f'  transient error ({status_code}), retrying in {wait:.1f}s '
				f'(attempt {attempt + 1}/{MAX_RETRIES})...')
			time.sleep(wait)


def call_huggingface_with_logprobs(model_name: str, system_prompt: str, user_prompt: str) -> tuple[str, list[dict] | None]:
	"""Like call_huggingface, but also requests per-token logprobs. Confirmed empirically
	(2026-08-24) that only DeepSeek-V3 on the novita backend actually returns them - Qwen
	accepts the parameter but silently returns logprobs=None. Returns (text, token_list) where
	token_list is None if the backend didn't supply logprobs. Retries on rate-limit/server
	errors; see call_anthropic for why.

	Parameters
		model_name: The Hugging Face repo id (see config.MODELS); in practice only
			ever called with the DeepSeek entry
		system_prompt: The system-role instructions
		user_prompt: The user-role message content
	"""
	client = OpenAI(api_key=config.HF_TOKEN, base_url='https://router.huggingface.co/v1')

	for attempt in range(MAX_RETRIES + 1):
		try:
			response = client.chat.completions.create(
				model=f'{model_name}:{config.HF_PROVIDER}',
				messages=[
					{'role': 'system', 'content': system_prompt},
					{'role': 'user', 'content': user_prompt}
				],
				logprobs=True, top_logprobs=1
			)
			choice = response.choices[0]
			tokens = None
			if choice.logprobs and choice.logprobs.content:
				tokens = [{'token': t.token, 'logprob': t.logprob} for t in choice.logprobs.content]
			return choice.message.content, tokens
		except Exception as e:
			status_code = getattr(e, 'status_code', None)
			is_transient = status_code == 429 or (status_code is not None and status_code >= 500)
			if not is_transient or attempt == MAX_RETRIES:
				raise
			wait = BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1)
			print(f'  transient error ({status_code}), retrying in {wait:.1f}s '
				f'(attempt {attempt + 1}/{MAX_RETRIES})...')
			time.sleep(wait)


def call_mistral(model_name: str, system_prompt: str, user_prompt: str) -> str:
	"""Calls a model directly against Mistral's own API. Retries on rate-limit (429) or
	server (5xx) errors; see call_anthropic for why.

	Parameters
		model_name: Mistral's own model id, e.g. 'ministral-8b-2512' (see config.MODELS)
		system_prompt: The system-role instructions
		user_prompt: The user-role message content
	"""
	client = OpenAI(api_key=config.MISTRAL_API_KEY, base_url='https://api.mistral.ai/v1')

	for attempt in range(MAX_RETRIES + 1):
		try:
			response = client.chat.completions.create(
				model=model_name,
				messages=[
					{'role': 'system', 'content': system_prompt},
					{'role': 'user', 'content': user_prompt}
				]
			)
			return response.choices[0].message.content
		except Exception as e:
			status_code = getattr(e, 'status_code', None)
			is_transient = status_code == 429 or (status_code is not None and status_code >= 500)
			if not is_transient or attempt == MAX_RETRIES:
				raise
			wait = BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1)
			print(f'  transient error ({status_code}), retrying in {wait:.1f}s '
				f'(attempt {attempt + 1}/{MAX_RETRIES})...')
			time.sleep(wait)


def call_model(model_id: str, system_prompt: str, user_prompt: str) -> str:
	"""Dispatches to the right provider's call_* function for the given model id.

	Parameters
		model_id: A key into config.MODELS
		system_prompt: The system-role instructions
		user_prompt: The user-role message content
	"""
	model = config.MODELS[model_id]

	if model.provider == 'anthropic':
		return call_anthropic(model.model_name, system_prompt, user_prompt)
	if model.provider == 'huggingface':
		return call_huggingface(model.model_name, system_prompt, user_prompt)
	if model.provider == 'mistral':
		return call_mistral(model.model_name, system_prompt, user_prompt)
	raise ValueError(f'Unknown provider: {model.provider}')


def extract_json(raw: str) -> dict:
	"""Parses a model's raw reply into a dict, tolerating a markdown code fence, literal
	newlines in string values, and trailing content after the JSON object."""
	cleaned = raw.strip()
	if cleaned.startswith('```'):
		cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.MULTILINE).strip()
	return json.JSONDecoder(strict=False).raw_decode(cleaned)[0]