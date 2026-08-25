#!/usr/bin/env python3
"""Model registry and environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

HF_PROVIDER = os.getenv('HF_PROVIDER', 'novita')


@dataclass(frozen=True)
class ModelConfig:
	"""One entry in the model registry."""

	id: str
	provider: str 
	model_name: str


MODELS: dict[str, ModelConfig] = {
	'claude': ModelConfig(id='claude', provider='anthropic', model_name=os.getenv('CLAUDE_MODEL', 'claude-sonnet-5')),
	'qwen': ModelConfig(id='qwen', provider='huggingface', model_name=os.getenv('HF_MODEL_QWEN', 'Qwen/Qwen2.5-72B-Instruct')),
	'mistral': ModelConfig(id='mistral', provider='mistral', model_name=os.getenv('MISTRAL_MODEL', 'ministral-8b-2512')),
	'deepseek': ModelConfig(id='deepseek', provider='huggingface', model_name=os.getenv('HF_MODEL_DEEPSEEK', 'deepseek-ai/DeepSeek-V3'))
}

JUDGE_MODEL_ID = os.getenv('JUDGE_MODEL_ID', 'qwen')

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')