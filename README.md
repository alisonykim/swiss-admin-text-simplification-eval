# Interactive Language Simplification: Administrative German *(Verwaltungsdeutsch)*

Comparing LLMs on **Sprachvereinfachung** (plain-language simplification) of Swiss
administrative texts (cantonal, federal, and municipal), with a focus on *evaluation* and *explainability*.

## Motivation

Swiss Verwaltungstexte are often dense, legalistic, and hard to parse for people without domain expertise or native German (e.g., people with immigration backgrounds, lower literacy, cognitive impairments, etc.). This project asks: how well do current LLMs do at rewriting these texts in plain language, and how do different models compare, on *readability*, *faithfulness to the original*, and the *kind* of changes they actually make?

## Related work

The Kanton Zürich data science team already runs [`simply-simplify-language`](https://github.com/machinelearningZH/simply-simplify-language) in production. It is a Streamlit app that sends a text to several LLMs via OpenRouter and lets staff pick the best draft, with a custom readability index (ZIX) and sentence-by-sentence
coaching feedback.

This project asks a narrower, different question: *How would one know, systematically, whether a given model or prompt is actually good at this?* It is a benchmark and evaluation harness rather than an end-user app:

- **Published, citable readability formulas** (Wiener Sachtextformel, LIX)
- **A model-agnostic, rule-based diff tagger** that cross-checks what a model *claims* it changed (sentence splits, passive→active, jargon removed/kept) against an independent, non-LLM signal
- **A fixed, separate judge model** (Qwen) scoring all four models, including itself, since Qwen is also one of the four being compared. The interactive dashboard's *Vergleich* tab has a toggle to include/exclude those rows from Qwen's aggregate score, and the *Erklärbarkeit* tab's self-consistency analysis found Qwen is *also* the least reproducible of the four models under repeated sampling.

## What it does

For each source text, the pipeline:

1. **Simplifies** the text with four models: **Claude** (the only paid/commercial model) plus three open-weight models from three different labs, namely **Qwen2.5-72B-Instruct** (Alibaba) and **DeepSeek-V3** (DeepSeek), both via Hugging Face Inference Providers with a single `HF_TOKEN`; and **Ministral 3 8B** (Mistral AI, Apache 2.0), called directly against Mistral's own API. Each model also returns a short rationale for its key edits.
2. **Scores readability** before/after with two independent German metrics: the *Wiener Sachtextformel* and *LIX*.
3. **Tags what changed**, model-agnostically, via a rule-based diff: sentence splitting, passive-to-active shifts, jargon terms removed vs. still present, lexical substitutions. This is deliberately independent of each model's self-reported rationale. In other words one LLM describing its own edits is not proof those edits happened; the rule-based diff is a reproducible cross-check.
4. **Judges faithfulness, simplicity, and fluency** with a separate LLM-as-judge call (defaults to Qwen2.5-72B-Instruct, the strongest open model, judging all four; judging stays free this way). Qwen is also one of the four models being scored, so its own 30 rows are self-judged, flagged per-row (`is_self_judged`) rather than hidden; see "Related work" above.

Results are written to `data/results/results.json` (full, including simplified texts, rationale, and per-row `is_self_judged`) and `data/results/results.csv` (flattened, for quick comparison across models).

On top of the core 120-row comparison, `plz-xai` runs four further black-box
explainability analyses: sentence-level ablation attribution, a TF-IDF faithfulness
cross-check against the judge, DeepSeek per-token confidence, and self-consistency under repeated sampling. `plz-explain` fits a small SHAP-explained proxy model over the diff-tag features to test whether they predict readability improvement (they do not, robustly; see Methodology notes). All of this is explorable interactively in a published dashboard (*Vergleich* / *Text-Explorer* / *Erklärbarkeit* tabs) built from the same `data/results/*.json` files.

### Why not "real" model-internals explainability?

Attention maps / SHAP / gradient attribution would need white-box access to the model internals. As Claude is a closed commercial API, that kind of explainability is not available for it. For the open-weight models, they are called via a hosted inference API rather than run locally, so there is no local access to internals either. The rationale + rule-based diff approach above is the model-agnostic alternative: it does not explain *why* a model produced a given token, but it does give a comparable, auditable account of *what changed* across all four models.

`plz-xai` pushes further within that same constraint, using four different techniques:
- **Sentence-ablation attribution:** remove one sentence from the source at a time, re-simplify, measure how much the output changes. This is the actual mechanism LIME/SHAP use for black-box models (perturb input, observe output), so it works identically across all four models without needing internals.
- **TF-IDF faithfulness cross-check:** an independent, deterministic lexical-overlap signal compared against the LLM judge's faithfulness score (Pearson $r = 0.356$ across all 120 rows); the judge tracks *some* real signal, but far from perfectly.
- **DeepSeek per-token logprobs:** the one piece of genuine model-internal signal in this project. Confirmed empirically (not assumed from docs) that of all four providers, only DeepSeek's backend actually returns logprobs.
- **Self-consistency:** Each model run 3x on the same text; how much the output varies is a reliability signal independent of how good any single output is. Notable finding: Qwen, the fixed judge, is the least self-consistent of the four (mean pairwise similarity 0.46 vs. 0.65-0.72 for the others).

All four run on a representative 9-text subset (ablation, self-consistency) or the full corpus (TF-IDF: 120 rows; logprobs: all 30 texts). These data subsets are illustrative, not exhaustive. The full per-row output in `data/results/xai_*.json`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env # fill in ANTHROPIC_API_KEY, HF_TOKEN, and MISTRAL_API_KEY
```

Qwen and DeepSeek are both ungated (no license click-through needed) and confirmed hosted on the `novita` backend as of 2026-08-22 (see the methodology note below on why this is worth re-checking). Ministral 3 8B doesn't go through Hugging Face, but is rather called directly against `https://api.mistral.ai/v1`.

The `openai` package is a dependency here purely as a generic OpenAI-compatible HTTP client, reused for two different base URLs (`router.huggingface.co` and `api.mistral.ai`). No OpenAI account or API key involved anywhere in this project.

## Usage

```bash
# Run all four models over every text in data/texts/manifest.json...
plz-run

# ...or restrict to a subset of models
plz-run --models claude qwen

# Fit + SHAP-explain a proxy model over the diff-tag features (needs pip install -e ".[analysis]")
plz-explain

# Run the four black-box explainability analyses (ablation, TF-IDF, logprobs, self-consistency)
plz-xai
```

## Tests

```bash
pytest
```

Tests cover the readability metrics, the rule-based diff tagger, and JSON-parsing of model output.

## Data

`data/texts/manifest.json` lists the source texts, each with a `level` (`cantonal`/`federal`/`municipal`), a `source_url`, and a path to the raw `.txt` file.

**All 30 texts are real, verbatim excerpts** from official Swiss government pages (11 cantonal, 9 federal, 10 municipal), spanning 11 German-speaking cantons/cities. Lengths vary by source (15-66 words). Every excerpt is short and cites its source URL.

## Methodology Notes & Limitations

- **The Hugging Face Inference Providers catalog rotates** which backend hosts a given model, and which models are hosted at all, changes over time (this is why the original Llama pick stopped working). Each model ID + `HF_PROVIDER=novita` combination in `.env.example` was verified directly against the Hub API on August 22, 2026: `curl -s "https://huggingface.co/api/models/<org>/<model>?expand[]=inferenceProviderMapping"`. Re-run this check rather than assuming a doc page's example snippet is current.

Mistral's own API model catalog rotates, too; `ministral-3-8b-2512` was current as of August 22, 2026. Check [docs.mistral.ai/getting-started/models/models_overview](https://docs.mistral.ai/getting-started/models/models_overview/) if it throws an error later, and specifically re-check that it is still open-weight.

- **WSTF and LIX** are formula-based proxies for reading difficulty, not comprehension measures. These metrics do not judge whether a "simple" sentence is also *correct*, hence the inclusion of the LLM-judge faithfulness score.

- **LLM-as-judge** has known biases (verbosity, style preferences, imperfect agreement with human raters). Thus, treat judge scores as a signal to inspect, not ground truth. The raw outputs are saved in `results.json`, so every score is traceable back to the actual text.

- **The jargon wordlist** in `diffing.py` is a small seed list, not a comprehensive lexicon of *Verwaltungsdeutsch*. It should be extended as the corpus grows.

- **Passive-voice detection** is a regex heuristic, not a parser, so it will miss and false-positive on some constructions. It suffices for a rough before/after signal, but not for a claim like "this model removed exactly $N$ passive constructions."
- **The SHAP proxy-model analysis (`plz-explain`) is a documented negative result.** A gradient-boosted model over the diff-tag features (jargon removed, substitutions, passive delta, sentence-split delta) predicting WSTF improvement does not generalize: 5-fold CV $R^2$ is negative for every tree-ensemble configuration tried, even shrunk to 15 trees/depth 1 (best case, regularized linear regression: $R^2 \approx 0.08 $, so essentially noise). Likely cause: WSTF is sensitive to word-length/syllable nuances that coarse count-based features cannot capture at $N=120$.

## Status

First iteration, complete: 30-text real corpus, full 4-model pipeline (120 rows), rule-based diff tagging, LLM-judge scoring with the self-judging caveat surfaced rather than hidden, four black-box explainability analyses, and an interactive dashboard (*Vergleich* / *Text-Explorer* / *Erklärbarkeit* tabs) built directly from `data/results/*.json`.

## Next iteration

- **Multi-judge panel**: Have all four models judge all four models' outputs and measure inter-judge agreement.
- Expand the corpus past 30 texts, and past the current 11 cantons/cities.
- Extend the jargon wordlist beyond its current small seed list.