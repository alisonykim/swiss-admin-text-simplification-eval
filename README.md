# plain-language-ch-verwaltung

Comparing LLMs on **Sprachvereinfachung** (plain-language simplification) of Swiss
administrative texts — cantonal, federal, and municipal — with a focus on making the
*evaluation* as rigorous as the simplification itself.

## Motivation

Swiss Verwaltungstexte are often dense, legalistic, and hard to parse for people without
domain expertise or native German — immigrants, people with lower literacy, anyone in a
hurry. This project asks: how well do current LLMs do at rewriting these texts in plain
language, and how do different models compare, on *readability*, *faithfulness to the
original*, and the *kind* of changes they actually make?

## Related work

The Kanton Zürich data science team already runs
[`simply-simplify-language`](https://github.com/machinelearningZH/simply-simplify-language)
in production — a Streamlit app that sends a text to several LLMs via OpenRouter and lets
staff pick the best draft, with a custom readability index (ZIX) and sentence-by-sentence
coaching feedback. It's a mature assistive tool with over a year of real-world use.

This project asks a narrower, different question and isn't trying to duplicate that tool.
Where `simply-simplify-language` optimizes for *giving a human good drafts to choose from*,
this project optimizes for *how would you know, systematically, whether a given model or
prompt is actually good at this* — a benchmark and evaluation harness rather than an
end-user app:

- **Published, citable readability formulas** (Wiener Sachtextformel, LIX) instead of a
  custom index, so results are reproducible outside this one project.
- **A model-agnostic, rule-based diff tagger** that cross-checks what a model *claims* it
  changed (sentence splits, passive→active, jargon removed/kept) against an independent,
  non-LLM signal — a self-reported rationale isn't proof the edit happened.
- **A judge model that never evaluates its own output** — deliberately using a different
  model than any being scored, to avoid a model grading its own homework.

## What it does

For each source text, the pipeline:

1. **Simplifies** the text with four models: **Claude** (the only paid/commercial model)
   plus three open-weight models from three different labs — **Qwen2.5-72B-Instruct**
   (Alibaba) and **DeepSeek-V3** (DeepSeek), both via Hugging Face Inference Providers with
   a single `HF_TOKEN`, and **Ministral 3 8B** (Mistral AI, Apache 2.0), called directly
   against Mistral's own API to use existing Mistral credits instead. Each model also
   returns a short rationale for its key edits.
2. **Scores readability** before/after with two independent German metrics: the *Wiener
   Sachtextformel* and *LIX*.
3. **Tags what changed**, model-agnostically, via a rule-based diff: sentence splitting,
   passive→active shifts, jargon terms removed vs. still present, lexical substitutions.
   This is deliberately independent of each model's self-reported rationale — one LLM
   describing its own edits is not proof those edits happened; the rule-based diff is a
   reproducible cross-check.
4. **Judges faithfulness and simplicity** with a separate LLM-as-judge call (defaults to
   Qwen2.5-72B-Instruct — the strongest open model — judging all four, including Claude, so
   Claude isn't grading its own output, and judging stays free).

Results are written to `data/results/results.json` (full, including simplified texts and
rationale) and `data/results/results.csv` (flattened, for quick comparison across models).

### Why not "real" model-internals explainability?

Attention maps / SHAP / gradient attribution would need white-box access to the model
internals. Claude is a closed commercial API, so that kind of explainability isn't available
for it at all — and even for the open-weight models, they're called via a hosted inference
API rather than run locally, so there's no local access to internals either. The rationale +
rule-based diff approach above is the model-agnostic alternative: it doesn't explain *why* a
model produced a given token, but it does give a comparable, auditable account of *what
changed* across all four models on equal footing.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, HF_TOKEN, and MISTRAL_API_KEY
```

Qwen and DeepSeek are both ungated (no license click-through needed) and confirmed hosted on
the `novita` backend as of 2026-08-22 — see the methodology note below on why that's worth
re-checking rather than trusting blindly. Ministral 3 8B doesn't go through Hugging Face at
all — it's called directly against `https://api.mistral.ai/v1`.

The `openai` package is a dependency here purely as a generic OpenAI-compatible HTTP client,
reused for two different base URLs (`router.huggingface.co` and `api.mistral.ai`) — no
OpenAI account or API key involved anywhere in this project.

## Usage

```bash
# run all four models over every text in data/texts/manifest.json
plz-run

# or restrict to a subset of models
plz-run --models claude qwen
```

## Tests

```bash
pytest
```

Tests cover the readability metrics, the rule-based diff tagger, and JSON-parsing of model
output — all without hitting any API, so they run offline and free.

## Data

`data/texts/manifest.json` lists the source texts, each with a `level`
(`cantonal`/`federal`/`municipal`), a `source_url`, and a path to the raw `.txt` file.

**The three seed texts are placeholders I wrote to match the style of real Verwaltungsdeutsch
so the pipeline has something to run against out of the box — they are not scraped from live
government pages.** Swap them for real excerpts (with the source URL recorded) before treating
any results as representative; keep excerpts short and cite the source, same convention used
by plain-language research corpora.

## Methodology notes / limitations

- **The Hugging Face Inference Providers catalog rotates** — which backend hosts a given
  model, and which models are hosted at all, changes over time (this is why the original
  Llama pick stopped working). Each model ID + `HF_PROVIDER=novita` combination in
  `.env.example` was verified directly against the Hub API on 2026-08-22:
  `curl -s "https://huggingface.co/api/models/<org>/<model>?expand[]=inferenceProviderMapping"`.
  Re-run that check before trusting these defaults months later, rather than assuming a
  doc page's example snippet is current — that's what broke the first time. Mistral's own
  API model catalog rotates too — `ministral-3-8b-2512` was current as of 2026-08-22; check
  [docs.mistral.ai/getting-started/models/models_overview](https://docs.mistral.ai/getting-started/models/models_overview/)
  if it 404s later, and specifically re-check it's still Apache 2.0 / open-weight, not a
  model that's since moved to Mistral's proprietary tier.
- **WSTF and LIX** are formula-based proxies for reading difficulty, not comprehension
  measures — they don't know if a "simple" sentence is also *correct*. That's what the
  LLM-judge faithfulness score is for.
- **LLM-as-judge** has known biases (verbosity, style preferences, imperfect agreement with
  human raters). Treat judge scores as a signal to inspect, not ground truth — the raw
  outputs are all saved in `results.json` so every score is traceable back to the actual text.
- **The jargon wordlist** in `diffing.py` is a small seed list, not a comprehensive
  Verwaltungsdeutsch lexicon — extend it as the corpus grows.
- **Passive-voice detection** is a regex heuristic, not a parser — it will miss and
  false-positive on some constructions. Good enough for a rough before/after signal, not
  for a claim like "model X removed exactly N passive constructions."

## Suggested build order (2-3 afternoons)

1. Curate a real corpus (10-15 texts, mixed cantonal/federal/municipal) and wire up all
   four model backends end to end on 1-2 texts.
2. Run the full pipeline, sanity-check the readability/diff/judge outputs, fix prompt or
   parsing issues that come up on real data.
3. Polish: write up results (a short comparison table/summary), finish test coverage, clean
   up README/code for portfolio review.
