# Phishing Detection System

[![CI](https://github.com/Bharath-Naveen/phishing-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Bharath-Naveen/phishing-detection-system/actions/workflows/ci.yml)

An explainable phishing detector. Give it a URL and it returns **likely phishing**, **uncertain** or **likely legitimate**, with the reasons behind the verdict. A fast machine-learning model scores the URL, a headless browser loads the page, and a rule-based judge weighs all the evidence. When the evidence disagrees, it says "uncertain" instead of guessing.

**[Try the live demo](https://bharathnaveen.com/projects/phishing-detection-system.html#demo)**: paste any URL and the URL model scores it in your browser. Nothing is sent to a server.

Built solo as an MS capstone, then rebuilt so that every number below can be regenerated with one script.

## Results

| What was measured | Result |
|---|---|
| URL model (XGBoost) on 147,702 test URLs from websites it never saw in training | F1 0.801 (95% CI 0.799 to 0.803), ROC-AUC 0.880, precision 0.772, recall 0.832 |
| 136 real brand sites (PayPal, Microsoft, Chase, Amazon and others), URL model alone | 40.4% flagged as phishing |
| The same 136 sites, full system with page capture | 4 called phishing, 24 uncertain, 108 legitimate |
| 99 fresh phishing pages from the PhishStats feed, full system | 39 called phishing, 20 uncertain, 40 legitimate |
| Time per URL | 15.7 ms for the URL model, 6.8 s median with live page capture |
| URL model against a free local LLM (llama3.2:3b, zero-shot) on the same 2,000 test URLs and the same CPU | F1 0.802 against 0.635 (difference +0.167, 95% CI +0.134 to +0.202), at 11 ms against 742 ms per URL (65 times faster) |
| URL model as a first filter: only scores between 0.30 and 0.70 go on to a slow second step | 80% of slow calls skipped across all 147,702 test URLs |
| Automated tests | 385, all passing, run in GitHub Actions on every push along with a Docker build |

**What it is good at, and what it is not.** The layers do their job of not crying wolf on real sites: the URL model alone would flag 40.4% of real brand pages, and the full system flags 4 of 136. Catching brand-new phishing is the weak spot (39 of 99). The URL model scores many famous homepages as high as phishing pages, so no rule can separate them without adding false alarms. A better URL model is the next step.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/layers_on_brand_sites-dark.svg">
  <img alt="What each layer does to 136 real brand URLs: 55 flagged by the URL model alone, 4 called phishing by the full system" src="docs/figures/layers_on_brand_sites-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/verdicts_by_set-dark.svg">
  <img alt="Final verdicts per evaluation set, with and without page capture" src="docs/figures/verdicts_by_set-light.svg">
</picture>

## Compared with an existing tool

Two comparisons, both replayable on any PC with no API key, no account and no cost.

**Against a language model.** A free open-source LLM (llama3.2:3b, a 2.0 GB download run locally with [Ollama](https://ollama.com)) was asked once whether each of 2,000 randomly drawn held-out test URLs is phishing. Its raw answers are saved in the repo, and `phishguard baseline-llm` re-scores them offline.

| Same 2,000 URLs, same 2-core CPU | Precision | Recall | F1 | False alarms on legitimate URLs | Time per URL (median) | Size |
|---|---|---|---|---|---|---|
| URL model (Layer 1) | 0.752 | 0.860 | 0.802 | 24.6% | 11 ms | 14.6 MB of model files |
| llama3.2:3b, zero-shot | 0.868 | 0.501 | 0.635 | 6.6% | 742 ms | 2.0 GB |

The URL model finds far more of the phishing (86% against 50%) and is 65 times faster. The LLM is the more careful of the two: it raises about a quarter as many false alarms. This is one small open model with one fixed prompt, so it says nothing about large hosted models.

**As a first filter.** `phishguard cascade` asks: if the URL model answers first and passes on only the URLs it is unsure about, how much slow checking is saved? With an unsure band of 0.30 to 0.70, 80% of URLs never reach the slow step. Measured end to end with the recorded LLM as that slow step (same 2,000 URLs, 82% of them settled by the URL model): false alarms fall from 24.6% to 16.1%, recall falls from 0.860 to 0.770, F1 goes from 0.802 to 0.787, and the pair runs 5.4 times faster than the LLM on every URL. So the filter buys speed and fewer false alarms, not a higher F1. Every band width is in [metrics/VERIFIED_METRICS.md](metrics/VERIFIED_METRICS.md); none was picked as "the" setting, because picking one from test rows would be choosing on the test set.

To ask the LLM again yourself (optional, free, about 25 minutes on 2 CPU cores): install Ollama, run `ollama pull llama3.2:3b`, then `phishguard train --full` and `phishguard baseline-llm --record`. Everything runs on your own machine.

Every chart is drawn by `phishguard figures` from the same result files as the tables, in light and dark versions. All ten, including ROC and precision-recall curves, the confusion matrix, calibration and feature contributions, are in [docs/figures](docs/figures/README.md).

Full tables, with the command, commit, data hash and environment that produced them: [metrics/VERIFIED_METRICS.md](metrics/VERIFIED_METRICS.md). Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## How it works

A URL passes through five layers:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/architecture-dark.svg">
  <img alt="How a URL moves through the five layers to a verdict" src="docs/figures/architecture-light.svg">
</picture>

1. **URL model and model agreement.** XGBoost scores 54 features of the URL text (lengths, entropy, lure words, brand names in the wrong place, free hosting). Three more models (Logistic Regression, Random Forest, LightGBM) vote as supporting evidence. No network access is needed.
2. **Live capture.** Playwright loads the page and records the final URL, redirects, form targets and TLS state.
3. **HTML and behavior.** Login harvesters, wrapper pages, cross-domain forms and suspicious scripts are detected from the captured page.
4. **Brand and trust context.** Does the brand on the page match the domain? A small registry of official domains is used as a weak prior, never as a whitelist.
5. **Evidence Adjudication Layer.** A deterministic judge adds up phishing, legitimacy and ambiguity signals, applies hard blockers (for example a password form posting to another domain) and returns the verdict with its reasons.

No external AI service is called at runtime, so the same input always gives the same verdict.

More detail in plain language: [docs/rebuild/HOW_IT_WORKS.md](docs/rebuild/HOW_IT_WORKS.md).

## Why the numbers can be trusted

An audit of the first version found a leaky train/test split and other problems, so training and evaluation were rebuilt. The full record is in [docs/rebuild/REBUILD_LOG.md](docs/rebuild/REBUILD_LOG.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/rebuild_before_after-dark.svg">
  <img alt="Held-out scores of the four models before and after the evaluation was fixed" src="docs/figures/rebuild_before_after-light.svg">
</picture>

- **Split by website.** Train and test are split by registered domain, so no website is in both. A guard stops the run if any domain overlaps.
- **No dataset shortcut.** In this dataset `https` is far more common on phishing rows, so features are computed without the scheme.
- **Evaluation stays out of training.** Every curated evaluation URL is removed from the training data.
- **Model chosen on validation data.** The test set is never used to pick a model or tune a rule.
- **Rule tuning written down first.** The plan was fixed before any tuning ([docs/rebuild/TUNING_PLAN.md](docs/rebuild/TUNING_PLAN.md)), rules were tuned on a validation half of one saved snapshot of real pages, and the test half was scored once.
- **One command reproduces everything.** `bash metrics/reproduce.sh` retrains and re-evaluates from scratch (seed 42, about 25 minutes on 2 CPU cores).
- **The demo is the evaluated model.** The in-browser version is checked against the Python app: identical features and scores on 148,746 URLs and identical verdicts on 21,032 ([demo/](demo/)).

## Run it

With Docker (the image includes the browser and the verified model):

```bash
docker compose up --build
```

Then open http://localhost:8501.

Or locally with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
playwright install chromium
phishguard serve                 # dashboard on http://localhost:8501
```

The `phishguard` command:

| Command | What it does |
|---|---|
| `phishguard analyze --url "https://example.com"` | Score one URL (add `--no-reinforcement` to skip the page capture) |
| `phishguard serve` | Start the dashboard |
| `phishguard train` | Train on a 50,000-row sample (`--full` for the whole dataset; needs the Kaggle CSV, see [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md)) |
| `phishguard evaluate` | Regenerate the verified metrics and the model card |
| `phishguard baseline-llm` | Compare the URL model with a free local LLM by replaying saved answers (offline; `--record` asks a local Ollama model again) |
| `phishguard cascade` | Measure how much slow checking the URL model saves as a first filter (offline) |
| `phishguard evaluate-live` | Evaluate the full system with live page capture (run by GitHub Actions) |
| `pytest` | Run the tests, including golden-output tests that pin the dashboard's behavior |

Live capture fills a dummy login on pages with a password field, to see where credentials go. Set `PHISH_ENABLE_LOGIN_INTERACTION=false` to turn that off.

## Repo layout

```text
src/phishguard/     the package: urls, data, features, models, pipelines, evaluation, app (dashboard, capture, judge)
models/layer1/      the verified model the app ships, with hashes in MANIFEST.json
metrics/            verified metrics, raw results and the reproduce script
data/evaluation/    evaluation URL sets and the frozen snapshot of real pages
demo/               export, parity tests and page builder for the in-browser demo
tests/              unit, regression, golden-output and JS parity tests
docs/               how it works, rebuild log, tuning plan, model card
archive/legacy/     retired code (an earlier AI adjudication path, old scripts)
```

## Limitations

- Recall on fresh phishing is low, as shown above.
- A URL-only model cannot see page content, so it over-flags modern legitimate sites; the page layers correct most of that.
- Live capture can be blocked by bot checks or shown harmless content by phishing kits, and phishing pages are often taken down within hours.
- When a page cannot be loaded, a high URL score usually ends as "likely phishing", which claims more certainty than the evidence gives.
- The training data has no dates, so performance on future campaigns is measured only on the PhishStats samples.
