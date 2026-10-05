# Verified metrics

Every number below was produced by running `phishguard evaluate` on the trained run. Nothing is copied from older reports. No synthetic data is used. Raw values: `metrics/results/evaluation.json`.

- Generated: 2026-09-30T23:51:06+00:00
- Git commit: `618516e321d8052fc78c1bf6490382ff5988c0b7` (with uncommitted changes)
- Code trees: src `468470e060`, tests `61670bc6de`, data/evaluation `a5fd2fd9ec`
- Seed: 42 (sampling, splits, models, bootstrap)
- Environment: Python 3.11.15, Linux 6.18.44-fc-v50, Intel(R) Xeon(R) Processor @ 2.80GHz (2 vCPU), none (CPU only)
- Packages: numpy 1.26.4, pandas 2.3.3, scikit-learn 1.5.1, xgboost 2.1.4, lightgbm 4.7.0, joblib 1.4.2, tldextract 5.3.2, pytest 8.4.2, playwright 1.49.0
- Reproduce: `phishguard train --full` then `phishguard evaluate --with-tests` (see `metrics/reproduce.sh`)

## Headline

| What | Value | Context |
|---|---|---|
| Primary model (XGBoost) held-out F1 | 0.801 | 95% CI 0.799 to 0.803; 147,702 test URLs from domains never seen in training; majority baseline F1 0.000 |
| ROC-AUC / PR-AUC | 0.880 / 0.849 | ROC-AUC 95% CI 0.878 to 0.881; baseline 0.500 / 0.464 |
| Precision / recall / false-positive rate | 0.772 / 0.832 / 21.3% | threshold 0.5 on raw model probability |
| Official brand URLs flagged by Layer 1 alone | 40.4% | 136 real official brand URLs (test half), app threshold |
| Official brand URLs the dashboard calls likely_phishing | 0.0% | ML-only dashboard path (no live capture); the rest are routed to uncertain |
| Real PhishStats phishing URLs flagged by Layer 1 | 81.0% | 284 URLs collected 2025-04 to 2026-04, hosts excluded from training |
| PhishStats URLs the dashboard calls likely_phishing | 24.3% | ML-only dashboard path |
| Tests | 372 / 372 passed | 53.5% line coverage of src/phishguard |

## Data

| What | Value |
|---|---|
| Source | Kaggle harisudhan411/phishing-and-legitimate-urls (`data/raw/kaggle/phishing_and_legitimate_urls.csv`, sha256 `330901bb4fbb63ff...`) |
| Raw rows x columns | 822,010 x 2 (427,028 legitimate, 394,982 phishing); no date column |
| After canonical dedupe | 794,563 |
| Run mode | FULL_DATASET; 823 curated legitimate homepages added |
| Removed so evaluation stays out-of-sample | 46,712 rows on legit-evaluation domains, 199 on phishing-evaluation hosts |
| Train / test rows | 600,743 / 147,702 (train 297,933 phishing, test 68,568 phishing) |
| Split | StratifiedGroupKFold by registered domain: 259,487 / 64,867 domains, overlap 0, rows without a parsable domain 820 |
| Fit / validation rows (inside train) | 511,749 / 88,994 (validation also grouped by domain) |
| Train/serve feature parity | 300 rows re-extracted the app's way, 0 mismatches |
| Model features | 54 |

## Held-out test (threshold 0.5)

| Model | Precision | Recall | F1 | F1 95% CI | ROC-AUC | PR-AUC | FPR | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | n/a | 0.500 | 0.464 | 0.0% | 79,134 / 0 / 68,568 / 0 |
| Logistic Regression | 0.679 | 0.800 | 0.735 | 0.732 to 0.737 | 0.773 | 0.696 | 32.8% | 53,165 / 25,969 / 13,683 / 54,885 |
| Random Forest | 0.784 | 0.802 | 0.793 | 0.791 to 0.795 | 0.884 | 0.866 | 19.2% | 63,966 / 15,168 / 13,585 / 54,983 |
| XGBoost (primary) | 0.772 | 0.832 | 0.801 | 0.799 to 0.803 | 0.880 | 0.849 | 21.3% | 62,253 / 16,881 / 11,499 / 57,069 |
| LightGBM | 0.764 | 0.847 | 0.803 | 0.801 to 0.806 | 0.877 | 0.833 | 22.7% | 61,203 / 17,931 / 10,502 / 58,066 |

Primary at the app's threshold (calibrated probability >= 0.5): precision 0.756, recall 0.859, F1 0.804, FPR 24.1%. Brier score raw 0.1408, calibrated 0.1455.

Selection: highest validation PR-AUC (domain-grouped validation split of the training rows); tie-break: fewer false alarms on the official-brand validation half. Validation PR-AUC per model: Logistic Regression 0.897, Random Forest 0.936, XGBoost 0.938, LightGBM 0.936.

## Cross-validation (StratifiedGroupKFold(5) by registered domain, 100,000 rows)

| Model | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|
| Majority-class baseline | 0.125 ± 0.280 | 0.500 ± 0.000 | 0.492 ± 0.025 | 20.0% ± 44.7 |
| Logistic Regression | 0.759 ± 0.018 | 0.832 ± 0.019 | 0.835 ± 0.027 | 28.4% ± 4.5 |
| Random Forest | 0.815 ± 0.013 | 0.902 ± 0.012 | 0.902 ± 0.014 | 16.6% ± 2.3 |
| XGBoost | 0.828 ± 0.015 | 0.908 ± 0.016 | 0.907 ± 0.019 | 18.3% ± 2.5 |
| LightGBM | 0.830 ± 0.013 | 0.909 ± 0.019 | 0.906 ± 0.022 | 19.0% ± 2.6 |

Mean ± standard deviation over 5 folds; domain overlap per fold: [0, 0, 0, 0, 0]. The majority-class baseline predicts whichever class is larger in each training fold, so in a fold where phishing is the majority it flags everything.

## External sets (never in training)

| Set | URLs | Truth | Primary flags (app threshold) | Each model flags (raw 0.5) | Overlap with training |
|---|---|---|---|---|---|
| official_brand_test_half | 136 | legitimate | 40.4% | Logistic Regression 30.1%, Random Forest 31.6%, XGBoost 36.8%, LightGBM 37.5% | 0 domain / 0 host |
| official_brand_validation_half | 162 | legitimate | 42.0% | Logistic Regression 31.5%, Random Forest 30.2%, XGBoost 37.0%, LightGBM 39.5% | 0 domain / 0 host |
| phishstats | 284 | phishing | 81.0% | Logistic Regression 74.6%, Random Forest 75.7%, XGBoost 78.9%, LightGBM 80.6% | 36 domain / 0 host |

For legitimate sets the flag rate is the false-alarm rate; for phishing sets it is recall. The overlap column counts URLs whose registered domain or exact host also appears in the training rows. The official-brand validation half was used to break model-selection ties, so quote the test half.

## Dashboard verdicts (ML-only (reinforcement=False): no live page capture)

| Set | URLs | likely_phishing | uncertain | likely_legitimate | Rate | Layer 1 flag rate |
|---|---|---|---|---|---|---|
| official_brand_test_half | 136 | 0 | 136 | 0 | false alarms 0.0% | 40.4% |
| phishstats | 284 | 69 | 215 | 0 | detected 24.3% | 81.0% |
| hard_legit | 15 | 0 | 15 | 0 | false alarms 0.0% | 86.7% |
| suite:obvious_legit | 6 | 0 | 6 | 0 | false alarms 0.0% | 100.0% |
| suite:tricky_legit | 6 | 0 | 6 | 0 | false alarms 0.0% | 100.0% |
| suite:obvious_phish | 2 | 2 | 0 | 0 | detected 100.0% | 100.0% |
| suite:hard_phishing | 4 | 4 | 0 | 0 | detected 100.0% | 100.0% |
| heldout_test_sample_legit | 200 | 18 | 182 | 0 | false alarms 9.0% | 26.0% |
| heldout_test_sample_phishing | 200 | 47 | 153 | 0 | detected 23.5% | 87.5% |

Without live capture the adjudication layer treats missing page evidence as a reason for `uncertain`, not as proof of safety.

## Latency

| Path | p50 | p95 | Context |
|---|---|---|---|
| Layer 1 only (features + model + calibration) | 15.7 ms | 22.7 ms | 400 held-out URLs; Intel(R) Xeon(R) Processor @ 2.80GHz; warm process, one URL at a time, includes 4-model agreement |
| Dashboard, ML-only (all rules + EAL) | 157.8 ms | 200.5 ms | same URLs |

## Full system with live page capture

From `phishguard evaluate-live` (GitHub Actions, commit `0014d21fdc`, 2026-09-30T19:07:59+00:00). Each page was captured once with Playwright and analyzed twice (legitimacy rescue on and off). Dummy login interaction: on. Raw values: `metrics/results/live_capture.json`.

| Set | URLs | Captured OK | likely_phishing | uncertain | likely_legitimate | Pass rate | Pass rate (captured OK) | likely_phishing with rescue on / off |
|---|---|---|---|---|---|---|---|---|
| eal_edge_cases | 15 | 14 (93.3%) | 3 | 8 | 4 | 86.7% | 85.7% | 3 / 3 (0 changed) |
| url_suites | 18 | 14 (77.8%) | 7 | 4 | 7 | 94.4% | 92.9% | 7 / 7 (0 changed) |
| hard_legit | 15 | 15 (100.0%) | 2 | 6 | 7 | 86.7% | 86.7% | 2 / 2 (0 changed) |
| official_brand | 136 | 134 (98.5%) | 8 | 28 | 100 | 94.1% | 94.8% | 8 / 8 (0 changed) |
| phishing_feed_live | 60 | 59 (98.3%) | 22 | 13 | 25 | 36.7% | 35.6% | 22 / 22 (0 changed) |
| tranco_live | 60 | 53 (88.3%) | 14 | 12 | 34 | 76.7% | 86.8% | 14 / 14 (0 changed) |

Fresh phishing URLs came from: phishstats. Popular-homepage sample: Tranco list K9QPW, 17 infrastructure domains without a website skipped.

Pass means: legitimate sets not labeled likely_phishing; phishing sets labeled likely_phishing; the edge cases follow their own expected outcome (`data/evaluation/eal_edge_cases.json`). Live phishing pages are often already taken down, so check the captured-OK column before reading a phishing pass rate.

Full-path latency (capture + analysis): p50 6.8 s, p95 15.0 s over 304 URLs (capture (Playwright) + full analysis, one URL at a time, GitHub-hosted runner).

Edge cases that did not pass:

| URL | Expected | Verdict | Captured OK |
|---|---|---|---|
| `https://mrbslink.weebly.com/` | phishing | uncertain | True |
| `https://gghdgsyttetyeyy72.weebly.com/` | phishing | uncertain | True |


## Rule tuning on a frozen live snapshot

Pre-registered in `docs/rebuild/TUNING_PLAN.md`. Every number below replays the same saved captures (`data/evaluation/frozen/live_snapshot_20260930.jsonl.gz`, sha256 `517d4560b4c3`, captured 2026-09-30) through the full analysis with no network access. Rules were changed using the val split only. The test split was scored once with the old rules and once with the final rules; one more test run, made with a stale model by mistake, is logged and discarded (`metrics/results/test_access_log.jsonl`). Model: the shipped Layer 1 bundle.

| Split | Rules | Legitimate pages labeled likely_phishing | of which official brand | Fresh phishing labeled likely_phishing | Fresh phishing labeled likely_legitimate |
|---|---|---|---|---|---|
| val | before | 41 / 273 (15.0%) | 19 / 162 | 25 / 109 (22.9%) | 41 / 109 |
| val | after | 28 / 273 (10.3%) | 11 / 162 | 25 / 109 (22.9%) | 41 / 109 |
| test | before | 22 / 212 (10.4%) | 7 / 136 | 39 / 99 (39.4%) | 40 / 99 |
| test | after | 19 / 212 (9.0%) | 4 / 136 | 39 / 99 (39.4%) | 40 / 99 |

Legitimate pages: official brand URLs, popular homepages (Tranco), hard legitimate and curated legitimate URLs (all captures, failed ones included). Fresh phishing: newest PhishStats URLs whose capture succeeded and whose host is not itself a top-10K site. Feed labels mean reported as phishing; some pages were already replaced by harmless content when captured.


## Layer 1 against a free local LLM (same URLs, same machine)

From `phishguard baseline-llm`, which replays saved answers with no network and no API key. The LLM is a yardstick, not part of the product. `llama3.2:3b` (2.0 GB download, run with Ollama 0.35.1) was asked once, on 2026-10-05, about a seeded random sample of 2,000 held-out test URLs (929 phishing). Zero-shot, temperature 0, one fixed prompt that was not edited after seeing answers. Hardware for both: Intel(R) Xeon(R) Processor @ 2.10GHz (2 cores), CPU only. Recording: `data/evaluation/frozen/llm_baseline_llama3_2_3b.jsonl.gz` (sha256 `ac2b37194db4`). Raw values: `metrics/results/llm_baseline.json`.

|  | Precision | Recall | F1 | F1 95% CI | FPR | Time per URL p50 | p95 |
|---|---|---|---|---|---|---|---|
| Layer 1 (app threshold) | 0.752 | 0.860 | 0.802 | 0.783 to 0.822 | 24.6% | 11.4 ms | 15.5 ms |
| llama3.2:3b, zero-shot | 0.868 | 0.501 | 0.635 | 0.606 to 0.664 | 6.6% | 741.6 ms | 1,447.3 ms |

F1 difference (Layer 1 minus LLM): +0.167, 95% CI +0.134 to +0.202 (paired bootstrap, 1,000 resamples). The LLM took 65 times as long per URL (medians). Both right on 1,169 URLs, only Layer 1 right on 437, only the LLM right on 296, both wrong on 98. LLM answers that were neither word: 8 (counted as legitimate).

This is one small open model with one prompt. It says nothing about larger hosted models, which published work puts higher on other datasets.

## Layer 1 as a first filter (cascade)

From `phishguard cascade`: arithmetic on the saved scores of all 147,702 held-out test URLs (`metrics/results/heldout_scores.csv.gz`), no network. Layer 1 settles a URL when its calibrated score is outside the band; URLs inside the band go to a slow second step. Every band is shown and none is picked as the operating point, because picking one from these rows would be choosing on the test set. Raw values: `metrics/results/cascade.json`.

| Unsure band | Sent to slow step | Slow calls avoided | Layer 1 wrong on what it settles | Phishing waved through | Legitimate blocked outright | Time per URL with page capture | Time per URL with local LLM |
|---|---|---|---|---|---|---|---|
| none (Layer 1 decides all) | 0.0% | 100.0% | 19.5% | 14.1% | 24.1% | 16 ms | 16 ms |
| 0.45 to 0.55 | 1.3% | 98.7% | 19.2% | 13.0% | 24.1% | 104 ms | 25 ms |
| 0.40 to 0.60 | 2.1% | 97.9% | 18.9% | 13.0% | 23.4% | 155 ms | 31 ms |
| 0.35 to 0.65 | 13.2% | 86.8% | 15.4% | 10.6% | 15.8% | 908 ms | 114 ms |
| 0.30 to 0.70 | 19.9% | 80.1% | 13.9% | 9.2% | 12.8% | 1,359 ms | 163 ms |
| 0.25 to 0.75 | 27.1% | 72.9% | 12.9% | 6.1% | 12.3% | 1,848 ms | 217 ms |
| 0.20 to 0.80 | 30.1% | 69.9% | 12.5% | 4.9% | 12.0% | 2,050 ms | 239 ms |
| 0.15 to 0.85 | 33.5% | 66.5% | 12.2% | 3.6% | 12.0% | 2,282 ms | 264 ms |
| 0.10 to 0.90 | 53.8% | 46.2% | 9.7% | 1.8% | 6.8% | 3,659 ms | 415 ms |
| 0.05 to 0.95 | 67.0% | 33.0% | 9.9% | 0.9% | 5.3% | 4,552 ms | 513 ms |

Time per URL = Layer 1 p50 (15.7 ms) + share sent on x slow step p50 (page capture 6,767 ms, capture (Playwright) + full analysis, one URL at a time, GitHub-hosted runner; local LLM 742 ms, llama3.2:3b on Intel(R) Xeon(R) Processor @ 2.10GHz (2 cores), CPU only). An estimate from measured medians. "Phishing waved through" and "legitimate blocked outright" are shares of all phishing / all legitimate test URLs that Layer 1 gets wrong without a second look.

**Measured end to end with the recorded LLM** (llama3.2:3b, 2,000 URLs): Layer 1 answers first and the LLM's saved answer is used only inside the band. LLM on every URL: F1 0.635 at 842 ms per URL (mean).

| Unsure band | LLM calls avoided | Precision | Recall | F1 | FPR | Time per URL (mean, measured) | Faster than LLM on everything |
|---|---|---|---|---|---|---|---|
| none (Layer 1 decides all) | 100.0% | 0.752 | 0.860 | 0.802 | 24.6% | 12 ms | 71.3x |
| 0.45 to 0.55 | 98.6% | 0.752 | 0.864 | 0.804 | 24.7% | 22 ms | 37.5x |
| 0.40 to 0.60 | 97.8% | 0.756 | 0.860 | 0.805 | 24.1% | 29 ms | 29.2x |
| 0.35 to 0.65 | 87.1% | 0.795 | 0.801 | 0.798 | 17.9% | 115 ms | 7.3x |
| 0.30 to 0.70 | 81.6% | 0.806 | 0.770 | 0.787 | 16.1% | 156 ms | 5.4x |
| 0.25 to 0.75 | 73.9% | 0.805 | 0.766 | 0.785 | 16.2% | 215 ms | 3.9x |
| 0.20 to 0.80 | 70.0% | 0.808 | 0.774 | 0.791 | 16.0% | 248 ms | 3.4x |
| 0.15 to 0.85 | 65.9% | 0.802 | 0.781 | 0.792 | 16.7% | 280 ms | 3.0x |
| 0.10 to 0.90 | 45.2% | 0.839 | 0.667 | 0.743 | 11.1% | 438 ms | 1.9x |
| 0.05 to 0.95 | 32.2% | 0.833 | 0.600 | 0.697 | 10.5% | 544 ms | 1.5x |

## Earlier baseline

The pre-rebuild audit (leaky split, scheme artifact, stale calibrator) is kept for comparison in `metrics/audit_baseline/` and at git tag `audit-baseline-2026-09-29`. Its numbers describe the old code and must not be quoted for the current system.
