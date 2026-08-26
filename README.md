# Simplifying German-Swiss Administrative Text: An Interactive Tool for Evaluating LLM Output

**<a href="https://alisonykim.github.io/swiss-admin-text-simplification-eval/" target="_blank" rel="noopener">Open the live dashboard →</a>**

Comparing LLMs on **Sprachvereinfachung** (plain-language simplification) of Swiss administrative texts (currently focused on Kanton Zürich and Bund), with a focus on *evaluation* and *explainability*.

## Motivation

*Verwaltungstexte* are often dense, legalistic, and hard to parse without domain expertise or native German skills (e.g., those with immigration backgrounds, lower literacy, etc.). This project asks: *how well do current LLMs do at rewriting these texts in plain language, and how do different models compare, on **readability**, **faithfulness** to the original, and the **kind  of changes** they actually make*?

The current corpus was inspired by:
* **Methodological fit:** The selected *Weisungen* and *Kreisschreiben* are structured into numbered, citable subsections that map directly onto auditable excerpt boundaries (see [Data](#data) below). Additionally, they cite actual statute articles rather than being pre-simplified themselves, which grounds the faithfulness evaluation in real legal specifics.
* **Personal experience:** I chose documents that I have had to read through myself, such as residence/migration directives and foreigner-specific tax rules, both from Kanton Zürich and the federal tax authority.

## Disclaimer

The simplified texts in this project are LLM output, NOT verified official guidance. They are provided to be *evaluated*, not to be *relied on* for personal or professional use. Please do not treat a simplified text as accurate or complete. Always refer back to the original, cited source, which is provided for every entry in the corpus.

## Related work

The Kanton Zürich data science team already runs <a href="https://github.com/machinelearningZH/simply-simplify-language" target="_blank" rel="noopener"><code>simply-simplify-language</code></a> in production. It is a Streamlit app that sends a text to several LLMs via OpenRouter and lets staff pick the best draft, with a custom readability index (ZIX) and sentence-by-sentence coaching feedback.

Rather than developing another end-user app, this project is intended to provide a benchmark and evaluation harness:

- **Published, citable readability formulas** (Wiener Sachtextformel, LIX)
- **A model-agnostic, rule-based diff tagger** that cross-checks what a model *claims* it changed (sentence splits, passive→active, jargon removed/kept) against an independent, non-LLM signal
- **A fixed, separate judge model** (Qwen) scoring all four models, including itself, since Qwen is also one of the four being compared. The interactive dashboard's *Modell-Vergleich* tab has a toggle to include/exclude those rows from Qwen's aggregate score.

## Project Steps

### Step 1: Core pipeline (`klartext-simplify`)

For each source text, the pipeline:

1. **Simplifies** the text with four models:
    - **Claude** (Anthropic): closed/proprietary, the only one of the four with no open-weight release, called via paid API
    - **Qwen2.5-72B-Instruct** (Alibaba): open-weight, called via Hugging Face Inference Providers, free
    - **DeepSeek-V3** (DeepSeek): open-weight, called via Hugging Face Inference Providers, free
    - **Ministral 3 8B** (Mistral AI, Apache 2.0): open-weight, called via paid API

    Each model also returns a short rationale for its key edits.
2. **Scores readability** before/after with two independent German metrics: the *Wiener Sachtextformel* and *LIX*.
3. **Tags what changed**, model-agnostically, via a rule-based diff: sentence splitting, passive-to-active shifts, jargon terms removed vs. still present, lexical substitutions. This is independent of each model's self-reported rationale and serves as a reproducible cross-check.
4. **Judges faithfulness, simplicity, and fluency** with a separate LLM-as-judge call (defaults to Qwen2.5-72B-Instruct, the strongest open model, judging all four; judging stays free this way). Qwen is also one of the four models being scored, so its own rows (one per source text) are self-judged, flagged per-row (`is_self_judged`) rather than hidden; see [Related work](#related-work) above.

Output: `data/results/results.json` (full, including simplified texts, rationale, and per-row `is_self_judged`) and `data/results/results.csv` (flattened, for quick comparison across models).

### Step 2: Explainability analysis (`klartext-xray`)

On top of the core (text x model) comparison, `klartext-xray` runs four further black-box analyses: sentence-level ablation attribution, a TF-IDF faithfulness cross-check against the judge, self-consistency under repeated sampling, and DeepSeek per-token confidence.

Output: `data/results/xai_*.json`.

#### Why not "real" model-internals explainability?

Attention maps / SHAP / gradient attribution in their classic form need white-box access to the model internals, and none of the four models offer that (Claude is closed; the open-weight ones are called via a hosted inference API rather than run locally). But "no internals" doesn't mean "no explainability" full stop, it rules out *white-box* XAI specifically. There is a separate, equally established branch, **model-agnostic, perturbation-based XAI**, that needs nothing but query access: perturb an input, observe how the output changes. LIME and the general (Kernel) form of SHAP both work this way. That is exactly the access available to all four APIs here.

`klartext-xray` runs four techniques within that constraint, though they are not all the same *kind* of technique, and this project does not treat them as such:

- **Sentence-ablation attribution** *(black-box attribution)*: remove one sentence from the source at a time, re-simplify, then measure how much the output changes. This is the same underlying mechanism as LIME and SHAP (perturb input, observe output), specifically **leave-one-out / occlusion-based attribution**, simplified (😉) relative to LIME (no surrogate-model fit) and SHAP (no combinatorial coalition averaging).
- **TF-IDF faithfulness cross-check** and **Self-consistency** *(evaluation/reliability checks, not attribution)*: the former is an independent, deterministic lexical-overlap signal compared against the LLM judge's faithfulness score, run across the full corpus; the latter runs each model 3x on the same text to measure output stability under repeated sampling. Neither explains a specific model decision, but rather answers, "Is this score trustworthy?" and "Is this model's behaviour stable?"
- **DeepSeek per-token logprobs** *(uncertainty quantification, not attribution)*: the single genuine model-internal signal in this project, indicating how confident the model was per token, not why it made a choice.

All four analyses can be found in the *Erklärbarkeit* tab on the dashboard. Ablation and self-consistency run on a representative six-text subset, chosen as the three longest texts per source (Kanton Zürich, Bund) by sentence count. This way, each source has enough sentences to produce a meaningful attribution signal. Illustrative, not exhaustive.

---

Both steps are explorable interactively in a published dashboard (*Modell-Vergleich* / *Text-Explorer* / *Erklärbarkeit* tabs), built directly from the `data/results/*.json` files these steps write, and published via GitHub Pages from this repo's `docs/` folder.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env # fill in ANTHROPIC_API_KEY, HF_TOKEN, and MISTRAL_API_KEY
```

Qwen and DeepSeek are both ungated (no license click-through needed) and confirmed hosted on the `novita` backend as of 2026-08-22 (see [Methodology Notes & Limitations](#methodology-notes--limitations) below on why this is worth re-checking). Ministral 3 8B doesn't go through Hugging Face, but is rather called directly against `https://api.mistral.ai/v1`.

The `openai` package is a dependency here purely as a generic OpenAI-compatible HTTP client, reused for two different base URLs (`router.huggingface.co` and `api.mistral.ai`). No OpenAI account or API key involved anywhere in this project.

## Usage

```bash
# Run all four models over every text in data/texts/manifest.json...
klartext-simplify

# ...or restrict to a subset of models
klartext-simplify --models claude qwen

# Run the four black-box explainability analyses (ablation, TF-IDF, logprobs, self-consistency)
klartext-xray
```

## Tests

```bash
pytest
```

Tests cover the readability metrics, the rule-based diff tagger, and JSON-parsing of model output.

## Data

`data/texts/manifest.json` lists the source texts, each with a `level` (`cantonal`/`federal`), a `source_url`, and a path to the raw `.txt` file.

**All texts are real, verbatim excerpts**, each one a single, self-contained subsection from an official Swiss government directive or circular, verified by reading the actual rendered source page (never trusting an automated text-extraction summary) and transcribed exactly, citations and all. The current corpus draws from three source documents: two Weisungen (directives) from Sicherheitsdirektion Kanton Zürich, Migrationsamt (Freizügigkeitsabkommen EU/EFTA-Staaten; Aufenthalt mit Erwerbstätigkeit aus Drittstaaten), and Kreisschreiben Nr. 45 from the Eidgenössische Steuerverwaltung ESTV (Quellenbesteuerung des Erwerbseinkommens von Arbeitnehmern). Every entry uses the source document's own subsection boundary as the excerpt boundary, rather than a hand-picked sentence range, so the excerpt selection is auditable against the source's own table of contents.

## Methodology Notes & Limitations

- **The Hugging Face Inference Providers catalog rotates** which backend hosts a given model, and which models are hosted at all, changes over time (this is why the original Llama pick stopped working). Each model ID + `HF_PROVIDER=novita` combination in `.env.example` was verified directly against the Hub API on August 22, 2026: `curl -s "https://huggingface.co/api/models/<org>/<model>?expand[]=inferenceProviderMapping"`. Re-run this check rather than assuming a doc page's example snippet is current.

- **Mistral's own API model catalog rotates, too**; `ministral-3-8b-2512` was current as of August 22, 2026. Check <a href="https://docs.mistral.ai/getting-started/models/models_overview/" target="_blank" rel="noopener">docs.mistral.ai/getting-started/models/models_overview</a> if it throws an error later, and specifically re-check that it is still open-weight.

- **WSTF and LIX** are formula-based proxies for reading difficulty, not comprehension measures. These metrics do not judge whether a "simple" sentence is also *correct*, hence the inclusion of the LLM-judge faithfulness score.

- **LLM-as-judge** has known biases (verbosity, style preferences, imperfect agreement with human raters). Thus, treat judge scores as a signal to inspect, not ground truth. The raw outputs are saved in `results.json`, so every score is traceable back to the actual text.

- **The jargon wordlist** in `diffing.py` is a small seed list, not a comprehensive lexicon of *Verwaltungsdeutsch*. It should be extended as the corpus grows.

- **Jargon matching is keyword-based, not morphological.** A term must start at a word boundary, but nothing is required after it, so German noun inflection (*zuständig* → *zuständige*/*zuständigen*) still matches correctly. This can still false-positive on an unrelated word that shares a root and also starts at a boundary (e.g. *gesucht*, the past participle of *suchen*, vs. the noun *Gesuch*); fully resolving that needs a lemmatizer. An earlier version matched a term anywhere inside a word at all (e.g. *erlass* inside the unrelated compound *Hauptniederlassung*); fixed to left-boundary-only matching and re-verified against the full corpus.

- **Passive-voice detection** is a regex heuristic, not a parser, so it will miss and false-positive on some constructions. It suffices for a rough before/after signal, but not for a claim like "this model removed exactly $N$ passive constructions."

- **The corpus was rebuilt once already** after an audit against live source URLs found verbatim-quoting errors in most entries. The current corpus uses the subsection-boundary method described in [Data](#data) above specifically to make the excerpt boundary auditable.

- **The interactive dashboard was built with AI coding assistance** (Claude), which was also used to edit code docstrings and check my German prose throughout. The ideas, corrections, and decisions steering the prompting, the methodology choices, the corpus curation, and the interpretation of results, are my own.

## Status

Second iteration: corpus rebuilt (60 texts, all real verbatim subsections from three official Zürich/federal migration and tax directives, see [Data](#data)), pipeline and explainability analyses re-run against it. Full four-model pipeline, rule-based diff tagging, LLM-judge scoring, four black-box explainability analyses, and an interactive dashboard (*Text-Explorer* / *Modell-Vergleich* / *Erklärbarkeit* tabs) built directly from `data/results/*.json`, published via GitHub Pages.

## Next iteration

- **Multi-judge panel**: Have all four models judge all four models' outputs and measure inter-judge agreement.
- **Proxy-model feature importance** (`src/explain.py`, not wired to a CLI command): fits a small model over the diff-tag features to test whether they predict readability improvement. Hasn't found a generalizable signal at the current sample size; worth revisiting with more data or richer features rather than treating the current result as final.
- Extend to a second German-speaking canton with a comparably high foreign-resident share (Basel-Stadt is the leading candidate), and to the remaining Migrationsamt directives and Quellensteuer guidance not yet included from Zürich.
- Extend the jargon wordlist beyond its current small seed list.