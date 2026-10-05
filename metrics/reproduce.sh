#!/usr/bin/env bash
# Regenerate metrics/VERIFIED_METRICS.md, metrics/results/evaluation.json, docs/MODEL_CARD.md,
# metrics/results/curves.json and the charts in docs/figures/.
#
# Needs: the Kaggle CSV in data/raw/kaggle/ (see docs/DATASET_SETUP.md), Python deps from
# requirements.txt, and `pip install -e .` for the phishguard command. About 25 minutes on 2 vCPU (measured: 1,316 s).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
pip install -q "pytest-cov>=5" >/dev/null 2>&1 || true
phishguard train --full --seed 42 --no-enrich-resume "$@"
phishguard evaluate --with-tests
phishguard curves
phishguard cascade --rescore   # held-out scores + first-filter sweep (offline)
phishguard baseline-llm        # replays the saved LLM answers (offline; --record needs Ollama)
phishguard cascade             # again, now with the LLM result for the time column
python -m phishguard.evaluation.report
phishguard figures
