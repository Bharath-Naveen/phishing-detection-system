"""`phishguard curves`: the per-row data that charts need and `evaluate` does not keep.

`phishguard evaluate` stores summary numbers (F1, ROC-AUC, confusion counts). Drawing a ROC curve, a
precision-recall curve, a calibration plot or a feature ranking needs the scores behind them. This
command re-scores the same held-out test rows with the same trained run and writes:

  metrics/results/curves.json

  roc / pr          201 points per model (interpolated on an even grid, so the file stays small)
  calibration       10 equal-width probability bins for the primary model, raw and calibrated
  score_histogram   primary-model calibrated score, 20 bins, per true class
  feature_contributions   mean absolute XGBoost path contribution per feature (log-odds), all test rows

Before writing, every model's ROC-AUC and PR-AUC is compared with `metrics/results/evaluation.json`.
If they differ by more than the tolerance, the command stops: the curves would describe a different
run than the published numbers.

    phishguard train --full
    phishguard evaluate --with-tests
    phishguard curves
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from phishguard.evaluation.evaluate import MODEL_NAMES, _phish_proba, _sha256
from phishguard.paths import outputs_dir, processed_dir, project_root

GRID = 201
AUC_TOLERANCE = 1e-6
# A refit logistic regression lands within about 0.002 of the published run, not on it (see REBUILD_LOG,
# Phase 4b). The tree models and the shipped model files reproduce exactly.
REFIT_TOLERANCE = {"logistic_regression": 5e-3}


def roc_points(y: np.ndarray, p: np.ndarray, n: int = GRID) -> Dict[str, List[float]]:
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0.0, 1.0, n)
    return {"fpr": [round(float(v), 5) for v in grid],
            "tpr": [round(float(v), 5) for v in np.interp(grid, fpr, tpr)]}


def pr_points(y: np.ndarray, p: np.ndarray, n: int = GRID) -> Dict[str, List[float]]:
    prec, rec, _ = precision_recall_curve(y, p)
    order = np.argsort(rec, kind="stable")
    grid = np.linspace(0.0, 1.0, n)
    return {"recall": [round(float(v), 5) for v in grid],
            "precision": [round(float(v), 5) for v in np.interp(grid, rec[order], prec[order])]}


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = which == b
        k = int(m.sum())
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k,
                    "mean_predicted": float(p[m].mean()) if k else None,
                    "observed_phishing_rate": float(y[m].mean()) if k else None})
    return out


def score_histogram(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> Dict[str, Any]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return {"edges": [float(e) for e in edges],
            "legitimate": [int(v) for v in np.histogram(p[y == 0], bins=edges)[0]],
            "phishing": [int(v) for v in np.histogram(p[y == 1], bins=edges)[0]]}


def feature_contributions(pipeline, X: pd.DataFrame, cols: List[str]) -> Optional[Dict[str, Any]]:
    """Mean |contribution| per feature from XGBoost's own per-row path contributions (log-odds)."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    clf = pipeline.steps[-1][1]
    if not hasattr(clf, "get_booster"):
        return None
    prep = pipeline.steps[0][1]
    Xt = prep.transform(X)
    names = [str(n).split("__", 1)[-1] for n in prep.get_feature_names_out()]
    contrib = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True, approx_contribs=True)[:, :-1]
    mean_abs, mean_signed = np.abs(contrib).mean(axis=0), contrib.mean(axis=0)
    rows = sorted(({"feature": names[i], "mean_abs_log_odds": float(mean_abs[i]), "mean_signed_log_odds": float(mean_signed[i])}
                   for i in range(len(names))), key=lambda r: -r["mean_abs_log_odds"])
    assert sorted(names) == sorted(cols), "transformed feature names differ from the bundle's feature list"
    return {"method": "XGBoost pred_contribs (approx_contribs=True), mean over all held-out test rows; "
                      "positive = pushes toward phishing", "n_rows": int(len(X)), "features": rows}


def build_curves(results_dir: Optional[Path] = None, check: bool = True, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    from phishguard.models.train import _exclude_layer1_only, _feature_matrix

    out_dir = results_dir or (project_root() / "metrics" / "results")
    mdir = Path(models_dir) if models_dir else outputs_dir() / "models"
    if not models_dir and not (mdir / "layer1_bundle.joblib").is_file():
        mdir = project_root() / "models" / "layer1"  # the shipped, evaluated models
    if not (mdir / "layer1_bundle.joblib").is_file():
        raise SystemExit(f"No model bundle in {mdir}. Run `phishguard train` first, or pass --models-dir models/layer1.")
    bundle = joblib.load(mdir / "layer1_bundle.joblib")
    test_csv = processed_dir() / "kaggle_test.csv"
    te = pd.read_csv(test_csv, dtype=str, low_memory=False)
    te = te.loc[te["url_length"].notna() & (te["url_length"].astype(str).str.len() > 0)].reset_index(drop=True)
    X, y, _, _ = _feature_matrix(te, _exclude_layer1_only(te, True, include_dns=False))
    cols = bundle["feature_columns"]
    assert list(X.columns) == cols, "feature columns differ from the bundle; use the run that produced it"

    try:
        mdir_rel = str(mdir.resolve().relative_to(project_root()))
    except ValueError:
        mdir_rel = str(mdir)
    out: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "phishguard curves",
        "test_csv_sha256": _sha256(test_csv), "test_rows": int(len(y)), "test_phishing_rows": int(y.sum()),
        "models_dir": mdir_rel, "bundle_sha256": _sha256(mdir / "layer1_bundle.joblib"),
        "bundle_train_csv_sha256": bundle.get("train_csv_sha256"),
        "grid_points": GRID, "models": {},
    }
    for m in MODEL_NAMES:
        f = mdir / f"{m}.joblib"
        if not f.is_file():
            continue
        p = _phish_proba(joblib.load(f), X)
        out["models"][m] = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                            "roc": roc_points(y, p), "pr": pr_points(y, p)}
        print("curves", m, flush=True)

    primary = bundle["model_name"]
    cal = bundle.get("calibrator")
    p_raw = _phish_proba(bundle["pipeline"], X)
    p_cal = np.clip(cal["model"].predict(p_raw), 0, 1) if cal and cal.get("type") == "isotonic" else p_raw
    out["primary"] = {
        "model": primary,
        "calibration": {"raw": calibration_bins(y, p_raw), "calibrated": calibration_bins(y, p_cal),
                        "brier_raw": float(np.mean((p_raw - y) ** 2)), "brier_calibrated": float(np.mean((p_cal - y) ** 2))},
        "score_histogram_calibrated": score_histogram(y, p_cal),
        "feature_contributions": feature_contributions(bundle["pipeline"], X, cols),
    }

    ev_path = out_dir / "evaluation.json"
    if check and ev_path.is_file():
        ev = json.loads(ev_path.read_text())
        diffs = {}
        for m, c in out["models"].items():
            ref = ev["held_out_test"]["models"].get(m, {})
            for k in ("roc_auc", "pr_auc"):
                if k in ref:
                    diffs[f"{m}.{k}"] = abs(c[k] - ref[k])
        ref_cal = ev["primary"]["calibration"]
        diffs["brier_raw"] = abs(out["primary"]["calibration"]["brier_raw"] - ref_cal["brier_raw"])
        diffs["brier_calibrated"] = abs(out["primary"]["calibration"]["brier_calibrated"] - ref_cal["brier_calibrated"])
        diffs["test_rows"] = abs(out["test_rows"] - ev["held_out_test"]["test_rows"])
        bad = {k: v for k, v in diffs.items() if v > REFIT_TOLERANCE.get(k.split(".")[0], AUC_TOLERANCE)}
        out["check_against_evaluation_json"] = {"tolerance": AUC_TOLERANCE, "tolerance_exceptions": REFIT_TOLERANCE,
                                                "max_abs_difference": max(diffs.values()),
                                                "differences": diffs,
                                                "evaluation_test_csv_sha256": ev["data"].get("test_csv_sha256")}
        if bad:
            raise SystemExit(f"curves do not match metrics/results/evaluation.json (different run?): {bad}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "curves.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", out_dir / "curves.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Save ROC/PR curve points, calibration bins and feature contributions for charts.")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--no-check", action="store_true", help="skip the comparison with evaluation.json")
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="folder with layer1_bundle.joblib and the four model files (default: outputs/models if you "
                         "trained a run, otherwise the shipped, evaluated models in models/layer1)")
    a = ap.parse_args()
    build_curves(results_dir=a.results_dir, check=not a.no_check, models_dir=a.models_dir)


if __name__ == "__main__":
    main()
