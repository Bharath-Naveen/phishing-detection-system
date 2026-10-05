"""`phishguard cascade`: how much slow, expensive checking Layer 1 saves when it goes first.

Idea: Layer 1 answers in milliseconds. Let it settle the URLs it is sure about and send only the ones in
an "unsure" band of scores to a slow second step (opening the page in a browser, or asking an LLM).
This command measures, on the held-out test rows, for a range of band widths:

  * what share of URLs Layer 1 settles alone, and how often it is wrong on those;
  * what share still goes to the slow step, and the time per URL that follows from our measured latencies;
  * when the LLM recording exists (`phishguard baseline-llm`), the real end-to-end result of
    "Layer 1 first, LLM only for the unsure ones" on the URLs the LLM actually answered.

It is arithmetic on saved scores: no network, no model call, a few seconds.

    phishguard cascade              uses metrics/results/heldout_scores.csv.gz (shipped in the repo)
    phishguard cascade --rescore    rebuilds that file from a trained run (`phishguard train --full`)

Every band is reported. None is picked as "the" operating point, because picking one by looking at
these rows would be choosing on the test set.

Result: metrics/results/cascade.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from phishguard.paths import processed_dir, project_root

# The unsure band is [lo, 1 - lo): lo = 0.5 means no band at all (Layer 1 decides everything).
BAND_LOWS = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
SCORES_NAME = "heldout_scores.csv.gz"
BRIER_TOLERANCE = 1e-6


def results_dir() -> Path:
    return project_root() / "metrics" / "results"


# --------------------------------------------------------------------------- the saved scores
def write_scores(path: Path, y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:  # no timestamp: same rows, same bytes
        gz.write(b"label,p_raw,p_cal\n")
        gz.write("".join(f"{int(a)},{b:.6f},{c:.6f}\n" for a, b, c in zip(y, p_raw, p_cal)).encode("ascii"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def read_scores(path: Path) -> Dict[str, np.ndarray]:
    with gzip.open(path, "rt", encoding="ascii") as f:
        d = np.loadtxt(f, delimiter=",", skiprows=1)
    return {"y": d[:, 0].astype(int), "p_raw": d[:, 1], "p_cal": d[:, 2]}


def rescore(out_path: Path, check: bool = True) -> Dict[str, Any]:
    """Score the held-out rows of a trained run with the app's model bundle and save label + scores."""
    import joblib
    import pandas as pd

    from phishguard.app.ml_layer1 import runtime_models_dir
    from phishguard.evaluation.evaluate import _phish_proba, _sha256
    from phishguard.models.train import _exclude_layer1_only, _feature_matrix

    test_csv = processed_dir() / "kaggle_test.csv"
    if not test_csv.is_file():
        raise SystemExit(f"{test_csv} not found. Run `phishguard train --full` first, or drop --rescore to use the shipped scores.")
    mdir = runtime_models_dir()
    bundle = joblib.load(mdir / "layer1_bundle.joblib")
    te = pd.read_csv(test_csv, dtype=str, low_memory=False)
    te = te.loc[te["url_length"].notna() & (te["url_length"].astype(str).str.len() > 0)].reset_index(drop=True)
    X, y, _, _ = _feature_matrix(te, _exclude_layer1_only(te, True, include_dns=False))
    assert list(X.columns) == bundle["feature_columns"], "feature columns differ from the bundle; use the run that produced it"
    p_raw = _phish_proba(bundle["pipeline"], X)
    cal = bundle.get("calibrator")
    p_cal = np.clip(cal["model"].predict(p_raw), 0, 1) if cal and cal.get("type") == "isotonic" else p_raw
    y = np.asarray(y).astype(int)
    info = {"test_csv_sha256": _sha256(test_csv), "bundle_sha256": _sha256(mdir / "layer1_bundle.joblib"),
            "rows": int(len(y)), "brier_raw": float(np.mean((p_raw - y) ** 2)), "brier_calibrated": float(np.mean((p_cal - y) ** 2))}
    ev_path = results_dir() / "evaluation.json"
    if check and ev_path.is_file():
        ev = json.loads(ev_path.read_text())
        ref = ev["primary"]["calibration"]
        diffs = {"brier_raw": abs(info["brier_raw"] - ref["brier_raw"]), "brier_calibrated": abs(info["brier_calibrated"] - ref["brier_calibrated"]),
                 "test_rows": abs(info["rows"] - ev["held_out_test"]["test_rows"])}
        if any(v > BRIER_TOLERANCE for v in diffs.values()):
            raise SystemExit(f"scores do not match metrics/results/evaluation.json (different run?): {diffs}")
        info["check_against_evaluation_json"] = diffs
    write_scores(out_path, y, p_raw, p_cal)
    out_path.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("wrote", out_path)
    return info


# --------------------------------------------------------------------------- the sweep
def band_row(y: np.ndarray, p: np.ndarray, lo: float, second: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """One band [lo, 1 - lo). `second` = the slow step's 0/1 answers for the same rows, when we have them."""
    from phishguard.evaluation.evaluate import _counts

    hi = 1.0 - lo
    routed = (p >= lo) & (p < hi)
    settled = ~routed
    pred_l1 = (p >= 0.5).astype(int)
    n, ns = len(y), int(settled.sum())
    c = _counts(y[settled], pred_l1[settled]) if ns else None
    row: Dict[str, Any] = {
        "band": [lo, hi], "share_sent_to_slow_step": float(routed.mean()), "share_settled_by_layer1": float(settled.mean()),
        "settled": None if c is None else {
            "n": ns, "error_rate": float(1 - c["accuracy"]), "precision": c["precision"], "recall": c["recall"],
            "false_positive_rate": c["false_positive_rate"],
            "legitimate_called_phishing": int(c["fp"]), "phishing_called_legitimate": int(c["fn"])},
        "share_of_all_phishing_waved_through": float(np.sum(settled & (y == 1) & (pred_l1 == 0)) / max(int((y == 1).sum()), 1)),
        "share_of_all_legitimate_blocked_outright": float(np.sum(settled & (y == 0) & (pred_l1 == 1)) / max(int((y == 0).sum()), 1)),
    }
    if second is not None:
        final = np.where(routed, second, pred_l1)
        e = _counts(y, final)
        row["end_to_end"] = {k: e[k] for k in ("precision", "recall", "f1", "false_positive_rate", "accuracy")}
        row["end_to_end"]["n"] = n
        if routed.any():
            r = _counts(y[routed], second[routed])
            row["slow_step_on_its_share"] = {"n": int(routed.sum()), "accuracy": r["accuracy"], "f1": r["f1"]}
    return row


def time_per_url(share_routed: float, l1_ms: float, slow_ms: float) -> Dict[str, float]:
    t = l1_ms + share_routed * slow_ms
    return {"ms_per_url": float(t), "times_faster_than_slow_step_on_everything": float(slow_ms / t),
            "slow_calls_avoided": float(1 - share_routed)}


def _slow_steps() -> Dict[str, Dict[str, Any]]:
    """Measured latencies of the slow steps we actually have numbers for."""
    out: Dict[str, Dict[str, Any]] = {}
    lv = results_dir() / "live_capture.json"
    if lv.is_file():
        lat = json.loads(lv.read_text())["latency_seconds_full_path"]
        out["live_page_capture"] = {"p50_ms": lat["p50"] * 1000, "source": "metrics/results/live_capture.json", "note": lat.get("note")}
    lb = results_dir() / "llm_baseline.json"
    if lb.is_file():
        d = json.loads(lb.read_text())
        out["local_llm"] = {"p50_ms": d["llm"]["latency_ms"]["p50"], "source": "metrics/results/llm_baseline.json",
                            "note": f"{d['recording'].get('model')} on {d['recording'].get('hardware')}"}
    return out


def build(scores_path: Optional[Path] = None, out_dir: Optional[Path] = None, llm_recording: Optional[Path] = None) -> Dict[str, Any]:
    from phishguard.evaluation import llm_baseline
    from phishguard.evaluation.evaluate import _sha256

    out_dir = out_dir or results_dir()
    scores_path = scores_path or (results_dir() / SCORES_NAME)
    if not scores_path.is_file():
        raise SystemExit(f"{scores_path} not found. Run `phishguard cascade --rescore` after `phishguard train --full`.")
    s = read_scores(scores_path)
    y, p = s["y"], s["p_cal"]
    ev_path = results_dir() / "evaluation.json"
    l1_ms = json.loads(ev_path.read_text())["latency_ms"]["layer1_only"]["p50"] if ev_path.is_file() else None
    slow = _slow_steps()

    out: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": "phishguard cascade",
        "scores_file": scores_path.name, "scores_sha256": _sha256(scores_path),
        "rows": int(len(y)), "phishing_rows": int(y.sum()),
        "score_used": "calibrated Layer 1 probability (what the app thresholds at 0.5)",
        "how_to_read": "Layer 1 settles a URL when its score is below the band (legitimate) or at/above it (phishing); "
                       "URLs inside the band go to the slow step. Band [0.5, 0.5) = Layer 1 decides everything.",
        "latencies_used": {"layer1_p50_ms": l1_ms, "layer1_source": "metrics/results/evaluation.json", "slow_steps": slow,
                           "note": "time per URL = Layer 1 p50 + (share sent on) x slow step p50; an estimate from measured medians"},
        "bands": [],
    }
    for lo in BAND_LOWS:
        row = band_row(y, p, lo)
        if l1_ms is not None:
            row["time"] = {name: time_per_url(row["share_sent_to_slow_step"], l1_ms, st["p50_ms"]) for name, st in slow.items()}
        out["bands"].append(row)

    rec = llm_recording or llm_baseline.recording_path()
    if rec.is_file():
        meta, rows = llm_baseline.read_recording(rec)
        a = llm_baseline.arrays(rows)
        section: Dict[str, Any] = {
            "what": "Layer 1 first, the recorded LLM answer only for URLs in the band; every number is measured on the same URLs",
            "recording_sha256": _sha256(rec), "model": meta.get("model"), "hardware": meta.get("hardware"), "n": int(len(a["y"])),
            "llm_on_everything": {"f1": llm_baseline._metrics(a["y"], a["llm"])["f1"], "ms_per_url_mean": float(a["llm_ms"].mean())},
            "bands": [],
        }
        for lo in BAND_LOWS:
            row = band_row(a["y"], a["p_cal"], lo, second=a["llm"])
            routed = (a["p_cal"] >= lo) & (a["p_cal"] < 1 - lo)
            ms = float((a["l1_ms"].sum() + a["llm_ms"][routed].sum()) / len(a["y"]))
            row["measured_time"] = {"ms_per_url_mean": ms, "times_faster_than_llm_on_everything": float(a["llm_ms"].mean() / ms),
                                    "llm_calls_avoided": float(1 - routed.mean())}
            section["bands"].append(row)
        out["with_recorded_llm"] = section

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cascade.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{out['rows']} held-out URLs")
    print("  band            sent on   settled wrong   phishing waved through")
    for r in out["bands"]:
        st = r["settled"]
        print(f"  [{r['band'][0]:.2f}, {r['band'][1]:.2f})   {r['share_sent_to_slow_step']:6.1%}   {st['error_rate']:6.1%}          {r['share_of_all_phishing_waved_through']:6.1%}")
    print("wrote", out_dir / "cascade.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="How much slow checking Layer 1 saves when it goes first (offline arithmetic on saved scores).")
    ap.add_argument("--rescore", action="store_true", help="rebuild the saved scores from a trained run before the sweep")
    ap.add_argument("--no-check", action="store_true", help="with --rescore: skip the comparison with evaluation.json")
    ap.add_argument("--results-dir", type=Path, default=None)
    a = ap.parse_args()
    if a.rescore:
        rescore(results_dir() / SCORES_NAME, check=not a.no_check)
    build(out_dir=a.results_dir)


if __name__ == "__main__":
    main()
