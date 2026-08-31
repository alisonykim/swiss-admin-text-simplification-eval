# Simplifying German-Swiss Administrative Text: An Interactive Tool for Evaluating and Diagnosing LLM Output

**<a href="https://alisonykim.github.io/swiss-admin-text-simplification-eval/" target="_blank" rel="noopener">Open the live dashboard →</a>**

Comparing LLMs on **Sprachvereinfachung** (plain-language simplification) of Swiss administrative texts (currently focused on Kanton Zürich and Bund), with a focus on *evaluation* and *explainability*.

This project grew out of my own experience of navigating *Verwaltungsdeutsch* as a non-native German speaker. After consulting an administrative resource myself, I sometimes use chatbots to summarise longer, denser texts in order to check my understanding. While the chatbot checks have helped, I still want concrete metrics and an understanding of the chatbot's decision in order to judge the quality of a simplification myself. This project is my attempt at building such an evaluation and explainability tool.

I welcome feedback on any and all elements of this project.

## Motivation

*Verwaltungstexte* are often dense, legalistic, and hard to parse without domain expertise or native German skills (e.g., those with immigration backgrounds, lower literacy, etc.). This project asks: *how well do current LLMs do at rewriting these texts in plain language, and how do different models compare on **readability**, **faithfulness** to the original, and the **kind of changes** they actually make*?

The current corpus was inspired by:
* **Methodological fit:** The selected *Weisungen* and *Kreisschreiben* are structured into numbered, citable subsections that map directly onto auditable excerpt boundaries (see [Data](#data) below). Additionally, they cite actual statute articles rather than being pre-simplified themselves, which grounds the faithfulness evaluation in real legal specifics.
* **Personal experience:** I chose documents that I have had to read through myself, such as residence/migration directives and foreigner-specific tax rules, both from Kanton Zürich and the federal tax authority.

## Related Work

The Kanton Zürich data science team has already piloted <a href="https://github.com/machinelearningZH/simply-simplify-language" target="_blank" rel="noopener"><code>simply-simplify-language</code></a>, a Streamlit app that sends a text to several LLMs via OpenRouter and lets staff pick the best draft, with a custom understandability index (<a href="https://github.com/machinelearningZH/zix_understandability-index" target="_blank" rel="noopener">ZIX</a>) and sentence-by-sentence coaching feedback.

Rather than developing another end-user app, this project aims to help users assess a text simplification model's performance and understand its decisions:

- **Published readability formulas *(Lesbarkeit)*:** Wiener Sachtextformel, LIX
- **Comprehensibility score *(Verständlichkeit)*, specifically for *Swiss Standard German*:** ZIX
- **A model-agnostic, rule-based diff tagger** that cross-checks what a model *claims* it changed against an independent, non-LLM signal
- **A fixed, separate judge model** (Qwen) scoring all four models.
- **Four model diagnostic analyses** (`klartext-xray`): one attribution method and three reliability/confidence checks

## Project Steps

### Step 1: Core pipeline (`klartext-simplify`)

For each source text, the pipeline:

1. **Simplifies** the text with four models:
    - **Claude** (Anthropic): closed/proprietary, the only one of the four with no open-weight release, called via paid API
    - **Qwen2.5-72B-Instruct** (Alibaba): open-weight, called via Hugging Face Inference Providers, free
    - **DeepSeek-V3** (DeepSeek): open-weight, called via Hugging Face Inference Providers, free
    - **Ministral 3 8B** (Mistral AI, Apache 2.0): open-weight, called via paid API

    Each model also returns a short rationale for its key edits.
2. **Scores readability and comprehensibility** before/after with three independent German metrics: the *Wiener Sachtextformel*, *LIX*, and *ZIX* (see [Related Work](#related-work)).
3. **Tags what changed**, model-agnostically, via a rule-based diff. This is independent of each model's self-reported rationale and serves as a reproducible cross-check.
4. **Judges faithfulness, simplicity, and fluency** with a separate LLM-as-judge call (defaults to Qwen2.5-72B-Instruct, which is also one of the four models being scored).

Output: `data/results/results.json` (full, including simplified texts, rationale, and per-row `is_self_judged`) and `data/results/results.csv` (flattened, for quick comparison across models).

### Step 2: Model Diagnostics (`klartext-xray`)

On top of the core (text x model) comparison, `klartext-xray` runs four further analyses: sentence-level ablation attribution, a TF-IDF faithfulness cross-check against the judge, self-consistency under repeated sampling, and DeepSeek per-token confidence.

Output: `data/results/diagnostics_*.json`.

#### Why not white-box explainability methods?

Attention maps, SHAP, and gradient attribution need white-box access to model internals, which none of the four models here provide. This project nevertheless uses one **model-agnostic, perturbation-based XAI** method, sentence ablation, explained below.

#### `klartext-xray` runs four model diagnostic analyses:

- **Sentence-ablation attribution** *(black-box XAI)*: remove one sentence from the source at a time, re-simplify, then measure how much the output changes. This is the same underlying mechanism as LIME and SHAP (perturb input, observe output), specifically **leave-one-out / occlusion-based attribution**, simplified (get it? 😉) relative to LIME and SHAP.
- **TF-IDF faithfulness cross-check** and **Self-consistency** *(evaluation/reliability checks, not attribution)*: the former is an independent, deterministic lexical-overlap signal compared against the LLM judge's faithfulness score, run across the full corpus; the latter runs each model 3x on the same text to measure output stability under repeated sampling. Neither explains a specific model decision, but rather answers, "Is this score trustworthy?" and "Is this model's behaviour stable?"
- **DeepSeek per-token logprobs** *(uncertainty quantification, not attribution)*: the single genuine model-internal signal in this project, indicating how confident the model was per token.

All four analyses can be found in the *Modell-Diagnostik* tab on the dashboard. Ablation and self-consistency run on a representative six-text subset, chosen as the three longest texts per source (Kanton Zürich, Bund) by sentence count. This way, each source has enough sentences to produce a meaningful attribution signal.

---

Both steps are explorable interactively in a published dashboard (*Text-Explorer* / *Modell-Vergleich* / *Modell-Diagnostik* tabs), built directly from the `data/results/*.json` files these steps write, and published via GitHub Pages from this repo's `docs/` folder.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env # Fill in ANTHROPIC_API_KEY, HF_TOKEN, and MISTRAL_API_KEY
```

Qwen and DeepSeek are both ungated and confirmed hosted on the `novita` backend as of August 22, 2026 (see [Methodology Notes & Limitations](#methodology-notes--limitations)). Ministral 3 8B doesn't go through Hugging Face, but is rather called directly against `https://api.mistral.ai/v1`.

`pip install` also pulls in [ZIX](https://github.com/machinelearningZH/zix_understandability-index).

The `openai` package is a dependency here as a generic OpenAI-compatible HTTP client, reused for two different base URLs (`router.huggingface.co` and `api.mistral.ai`). No OpenAI account or API key involved anywhere in this project.

## Usage

```bash
# Run all four models over every text in data/texts/manifest.json...
klartext-simplify

# ...or restrict to a subset of models
klartext-simplify --models claude qwen

# Run the four black-box model diagnostics (ablation, TF-IDF, self-consistency, logprobs)
klartext-xray
```

## Tests

```bash
pytest
```

Tests cover the readability metrics, the rule-based diff tagger, and JSON-parsing of model output.

## Data

`data/texts/manifest.json` lists the source texts, each with a `level` (`cantonal`/`federal`), a `source_url`, and a path to the raw `.txt` file.

**All texts are real, verbatim excerpts** from official Swiss government sources. The current corpus draws from three source documents: two Weisungen (directives) from Sicherheitsdirektion Kanton Zürich, Migrationsamt (Freizügigkeitsabkommen EU/EFTA-Staaten; Aufenthalt mit Erwerbstätigkeit aus Drittstaaten), and Kreisschreiben Nr. 45 from the Eidgenössische Steuerverwaltung ESTV (Quellenbesteuerung des Erwerbseinkommens von Arbeitnehmern).

## Methodology Notes & Limitations

- **The Hugging Face Inference Providers catalogue rotates** which backend hosts a given model, and which models are hosted at all, changes over time (this is why the original Llama pick stopped working). Each model ID + `HF_PROVIDER=novita` combination in `.env.example` was verified directly against the Hub API on August 22, 2026: `curl -s "https://huggingface.co/api/models/<org>/<model>?expand[]=inferenceProviderMapping"`. Re-run this check before use.

- **Mistral's own API model catalogue rotates, too**; `ministral-8b-2512` was current as of August 22, 2026. Check <a href="https://docs.mistral.ai/getting-started/models/models_overview/" target="_blank" rel="noopener">docs.mistral.ai/getting-started/models/models_overview</a> if it throws an error later, and specifically re-check that it is still open-weight.

- **WSTF and LIX** are formula-based proxies for reading difficulty, not comprehension measures. These metrics do not judge whether a "simple" sentence is also *correct*, hence the inclusion of the LLM-judge faithfulness score.

- **<a href="https://github.com/machinelearningZH/zix_understandability-index/" target="_blank" rel="noopener">ZIX</a>** is built on a small model (CEFR-vocabulary coverage, word-frequency scores, and RIX sentence length). As a comprehensibility metric, ZIX aims to quantify how easily the content can be grasped, while readability quantifies how easily one can follow a written text.

- **LLM-as-judge** has known biases (verbosity, style preferences, imperfect agreement with human raters), so do not take the scores at face value. As the raw outputs are saved in `results.json`, every score is traceable back to the actual text.

- **The judge occasionally code-switches into Chinese characters** in its free-text comment (2.5% of rows, 6/240), despite the system prompt requiring German; the structured 1-5 scores themselves stay clean.

- **The jargon wordlist** in `diffing.py` is a small seed list, not a comprehensive lexicon of *Verwaltungsdeutsch*. It should be extended as the corpus grows.

- **Jargon matching is keyword-based, not morphological.** A term must start at a word boundary, but nothing is required after it, so German noun inflection (e.g., *zuständig* → *zuständige*/*zuständigen*) still matches correctly. This can still false-positive on an unrelated word that shares a root and also starts at a boundary (e.g. *gesucht*, the past participle of *suchen*, vs. the noun *Gesuch*). Resolving this robustly would require a lemmatiser.

- **Passive-voice detection** is a regex heuristic, not a parser, so it will miss and false-positive on some constructions. It suffices for a rough before/after signal but is not a comprehensive methodology.

- **The corpus was rebuilt once already** after a manual check against live source URLs revealed verbatim-quoting errors in most entries. The current corpus instead uses each source document's own subsection boundaries as excerpt boundaries.

- **TF-IDF similarity is a proxy for shared vocabulary, not shared meaning.** This makes it a weaker fit for a task where good simplification is expected to change the wording on purpose. It is nevertheless kept as a simple cross-check, not a faithfulness metric.

- **AI coding assistance** (Claude) was used to build the interactive dashboard, edit code docstrings and README text, and check my German throughout. The ideas, corrections, and decisions behind prompting, methodology, corpus curation, and interpretation of results are my own.

## Current Status

### Third Iteration
- **Issue:** Discovered that the naive regex split was frequently incorrectly detecting sentence boundaries. In particular it was fragmenting legal citations (e.g. "Art. 65 Abs. 5") into false sentences. This inflated source-text sentence counts far more than simplified-text sentence counts.
- **Solution:** Implemented a sentence-segmentation model (<a href="https://github.com/segment-any-text/wtpsplit" target="_blank" rel="noopener">wtpsplit</a>) and re-ran the pipeline and diagnostic analyses against the corrected splitter.
- **Results:** Increased measured readability improvement scores (WSTF +69%, LIX +135% versus the pre-fix numbers), more accurate sentence counts.
- **Note:** The segmentation model is not perfect: it can still misread a dense abbreviation chain as a boundary (e.g. splitting "– finanzielle Mittel i.S.v." from "Art. 24 Abs. 4 Anhang I FZA vorweisen kann;", which belongs in the same sentence). Nevertheless, measuring directly against the naive regex splitter revealed that degenerate ≤3-word sentence fragments dropped from 36.7% of all split output (247/673, affecting 47/60 texts) to 1.2% (4/321, affecting 3/60 texts). This is a large, if imperfect, improvement.

### Second Iteration
- **Issue:** Manual check against live source URLs revealed verbatim-quoting errors in most corpus entries.
- **Solution:** Corpus rebuilt (60 texts, all real verbatim subsections from three official Zürich/federal migration and tax directives, using each source document's own subsection boundaries as excerpt boundaries, see [Data](#data)); pipeline and diagnostic analyses re-run against it.
- **Results:** Full four-model simplification, evaluation, and diagnostic analyses, plus an interactive dashboard (*Text-Explorer* / *Modell-Vergleich* / *Modell-Diagnostik* tabs) built directly from `data/results/*.json` (published via GitHub Pages).

## Next Iteration

- **Multi-judge panel**: Have all four models judge all four models' outputs and measure inter-judge agreement.
- **Proxy-model feature importance** (`src/explain.py`, not wired to a CLI command): fits a small model over the diff-tag features to test whether they predict readability improvement.
- **Extend corpus** to a second German-speaking canton with a comparably high foreign-resident proportion (Basel-Stadt is the leading candidate), and to the remaining Migrationsamt directives and Quellensteuer guidance not yet included from Zürich.
- **Extend the *Verwaltungsdeutsch* jargon wordlist** beyond its current small seed list.
- **More robust sentence-segmentation preprocessing**: Detect and standardise paragraph formatting (e.g., bullet-point/list formatting: "–"-prefixed items that continue one grammatical sentence) before running the segmentation model, which currently has no notion of paragraph structure and can misjudge sentence boundaries inside one.
- **A semantic or NLI-based faithfulness cross-check with the LLM's judgment** (e.g. embedding similarity) to complement or replace TF-IDF similarity.

## Disclaimer

The simplified texts in this project are LLM output, not verified official guidance. Please do not rely on this project, including its tools and outputs, for personal or professional use. Always refer back to the original, cited source, which is provided for every entry in the corpus.