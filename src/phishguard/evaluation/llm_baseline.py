"""`phishguard baseline-llm`: Layer 1 against a free, local, open-source LLM on the same held-out URLs.

The LLM is a yardstick, not part of the product. It is asked once ("record"); its raw answers are saved
in the repo; everyone else replays that file. So the comparison costs nothing, needs no API key and no
network, and gives the same numbers on any PC.

    phishguard baseline-llm                 replay the committed recording (default; offline, seconds)
    phishguard baseline-llm --record        ask a local Ollama model again (needs `phishguard train --full`
                                            for the held-out rows, and Ollama running on this machine)

Recording: data/evaluation/frozen/llm_baseline_<model>.jsonl.gz   (first line = how it was made)
Result:    metrics/results/llm_baseline.json

Rules fixed before the first recording (so nothing is tuned on test rows):
  * the sample is a seeded simple random sample of the held-out test rows (no hand-picking);
  * one zero-shot prompt (below), temperature 0, fixed seed, never edited after seeing answers;
  * an answer that is neither "phishing" nor "legitimate" counts as "legitimate" and is reported;
  * Layer 1 is judged at the app's threshold (calibrated probability >= 0.5), as everywhere else.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from phishguard.config import SEED
from phishguard.paths import outputs_dir, processed_dir, project_root

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_N = 2000
MAX_URL_CHARS = 500
SYSTEM_PROMPT = ("You are a security analyst. You will be shown one URL. Decide whether it is a phishing URL or a "
                 "legitimate URL, judging only from the URL text. Answer with exactly one word: phishing or legitimate.")
USER_TEMPLATE = "URL: {url}"
OPTIONS = {"temperature": 0, "seed": SEED, "num_predict": 8}
N_BOOT = 1000


def frozen_dir() -> Path:
    return project_root() / "data" / "evaluation" / "frozen"


def recording_path(model: str = DEFAULT_MODEL) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in model)
    return frozen_dir() / f"llm_baseline_{slug}.jsonl.gz"


def parse_answer(text: Optional[str]) -> Optional[int]:
    """1 = phishing, 0 = legitimate, None = neither word (or both) in the answer."""
    t = (text or "").lower()
    phish, legit = "phish" in t, "legit" in t
    if phish == legit:
        return None
    return 1 if phish else 0


# --------------------------------------------------------------------------- recording file
def write_recording(path: Path, meta: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> None:
    """Gzip with no timestamp, so the same content always gives the same bytes."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write((json.dumps({"meta": meta}, sort_keys=True) + "\n").encode("utf-8"))
        for r in rows:
            gz.write((json.dumps(r, sort_keys=True) + "\n").encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def read_recording(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[0]["meta"], lines[1:]


# --------------------------------------------------------------------------- scoring a recording
def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    from phishguard.evaluation.evaluate import point_metrics

    m = point_metrics(y, None, pred=pred)
    return {k: m[k] for k in ("precision", "recall", "f1", "false_positive_rate", "accuracy", "confusion_matrix", "n")}


def _paired_f1_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """95% interval for F1(a) - F1(b), resampling the same rows for both (paired bootstrap)."""
    from phishguard.evaluation.evaluate import _counts

    rng = np.random.default_rng(SEED)
    n, diffs, fa, fb = len(y), [], [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        ca, cb = _counts(y, a, w)["f1"], _counts(y, b, w)["f1"]
        fa.append(ca), fb.append(cb), diffs.append(ca - cb)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"f1_a_ci95": q(fa), "f1_b_ci95": q(fb), "difference_ci95": q(diffs), "n_boot": n_boot}


def _pctl(v: np.ndarray) -> Dict[str, float]:
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}


def arrays(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    parsed = [parse_answer(r.get("response")) for r in rows]
    return {
        "y": np.array([int(r["label"]) for r in rows]),
        "p_cal": np.array([float(r["p_cal"]) for r in rows]),
        "p_raw": np.array([float(r["p_raw"]) for r in rows]),
        "llm": np.array([0 if p is None else p for p in parsed]),
        "unparsed": np.array([p is None for p in parsed]),
        "l1_ms": np.array([float(r["l1_ms"]) for r in rows]),
        "llm_ms": np.array([float(r["llm_ms"]) for r in rows]),
    }


def score_recording(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    a = arrays(rows)
    y, l1 = a["y"], (a["p_cal"] >= 0.5).astype(int)
    lat_l1, lat_llm = _pctl(a["l1_ms"]), _pctl(a["llm_ms"])
    paired = _paired_f1_ci(y, l1, a["llm"])
    return {
        "recording": meta,
        "sample": {"n": int(len(y)), "phishing": int(y.sum()), "legitimate": int((y == 0).sum())},
        "layer1": {"rule": "calibrated probability >= 0.5 (the app's threshold)", **_metrics(y, l1),
                   "f1_ci95": paired["f1_a_ci95"], "latency_ms": lat_l1},
        "llm": {"rule": "zero-shot, one word; unparsed answers count as legitimate", **_metrics(y, a["llm"]),
                "f1_ci95": paired["f1_b_ci95"], "latency_ms": lat_llm,
                "unparsed_answers": int(a["unparsed"].sum()),
                "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens") or 0 for r in rows])),
                "answer_tokens_mean": float(np.mean([r.get("eval_tokens") or 0 for r in rows]))},
        "layer1_minus_llm_f1": {"difference": float(_metrics(y, l1)["f1"] - _metrics(y, a["llm"])["f1"]),
                                "ci95": paired["difference_ci95"], "n_boot": paired["n_boot"],
                                "method": "paired bootstrap over the same URLs"},
        "speed": {"same_machine": True, "hardware": meta.get("hardware"),
                  "llm_over_layer1_p50": float(lat_llm["p50"] / lat_l1["p50"]),
                  "llm_over_layer1_mean": float(lat_llm["mean"] / lat_l1["mean"]),
                  "note": "both timed in the recording run, one URL at a time, warm, CPU only"},
        "agreement": {"both_right": int(np.sum((l1 == y) & (a["llm"] == y))), "only_layer1_right": int(np.sum((l1 == y) & (a["llm"] != y))),
                      "only_llm_right": int(np.sum((l1 != y) & (a["llm"] == y))), "both_wrong": int(np.sum((l1 != y) & (a["llm"] != y)))},
    }


def replay(model: str = DEFAULT_MODEL, results_dir: Optional[Path] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    from phishguard.evaluation.evaluate import _sha256

    path = path or recording_path(model)
    if not path.is_file():
        raise SystemExit(f"No recording at {path}. Run `phishguard baseline-llm --record` (needs Ollama), or pull the repo's copy.")
    meta, rows = read_recording(path)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": "phishguard baseline-llm",
           "recording_file": str(path.relative_to(project_root())) if path.is_relative_to(project_root()) else str(path),
           "recording_sha256": _sha256(path), **score_recording(meta, rows)}
    out_dir = results_dir or (project_root() / "metrics" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_baseline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    l1, llm = out["layer1"], out["llm"]
    print(f"{out['sample']['n']} held-out URLs, model {meta.get('model')}")
    print(f"  Layer 1  F1 {l1['f1']:.3f}  precision {l1['precision']:.3f}  recall {l1['recall']:.3f}  FPR {l1['false_positive_rate']:.1%}  p50 {l1['latency_ms']['p50']:.1f} ms")
    print(f"  LLM      F1 {llm['f1']:.3f}  precision {llm['precision']:.3f}  recall {llm['recall']:.3f}  FPR {llm['false_positive_rate']:.1%}  p50 {llm['latency_ms']['p50']:.0f} ms")
    print("wrote", out_dir / "llm_baseline.json")
    return out


# --------------------------------------------------------------------------- recording (needs Ollama)
def _post(host: str, route: str, payload: Dict[str, Any], timeout: float = 600) -> Dict[str, Any]:
    req = urllib.request.Request(host.rstrip("/") + route, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(host: str, route: str) -> Dict[str, Any]:
    with urllib.request.urlopen(host.rstrip("/") + route, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def ask_llm(host: str, model: str, url: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    d = _post(host, "/api/chat", {"model": model, "stream": False, "options": OPTIONS, "keep_alive": "30m",
                                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                               {"role": "user", "content": USER_TEMPLATE.format(url=url[:MAX_URL_CHARS])}]})
    return {"response": d.get("message", {}).get("content", ""), "llm_ms": (time.perf_counter() - t0) * 1000,
            "prompt_tokens": d.get("prompt_eval_count"), "eval_tokens": d.get("eval_count")}


def draw_sample(n: int) -> Dict[str, Any]:
    """Seeded simple random sample of the held-out test rows, with Layer 1's scores for them."""
    import joblib
    import pandas as pd

    from phishguard.app.ml_layer1 import runtime_models_dir
    from phishguard.evaluation.evaluate import _phish_proba, _sha256
    from phishguard.models.train import _exclude_layer1_only, _feature_matrix

    test_csv = processed_dir() / "kaggle_test.csv"
    if not test_csv.is_file():
        raise SystemExit(f"{test_csv} not found. Run `phishguard train --full` first (it writes the held-out rows).")
    mdir = runtime_models_dir()
    bundle = joblib.load(mdir / "layer1_bundle.joblib")
    te = pd.read_csv(test_csv, dtype=str, low_memory=False)
    te = te.loc[te["url_length"].notna() & (te["url_length"].astype(str).str.len() > 0)].reset_index(drop=True)
    pick = np.sort(np.random.default_rng(SEED).choice(len(te), size=min(n, len(te)), replace=False))
    sub = te.iloc[pick].reset_index(drop=True)
    X, y, _, _ = _feature_matrix(sub, _exclude_layer1_only(sub, True, include_dns=False))
    assert list(X.columns) == bundle["feature_columns"], "feature columns differ from the bundle"
    p_raw = _phish_proba(bundle["pipeline"], X)
    cal = bundle.get("calibrator")
    p_cal = np.clip(cal["model"].predict(p_raw), 0, 1) if cal and cal.get("type") == "isotonic" else p_raw
    return {"urls": sub["canonical_url"].astype(str).tolist(), "y": [int(v) for v in y], "p_raw": p_raw, "p_cal": p_cal,
            "test_rows": int(len(te)), "test_csv_sha256": _sha256(test_csv),
            "bundle_sha256": _sha256(mdir / "layer1_bundle.joblib"), "primary_model": bundle["model_name"]}


def time_layer1(urls: List[str]) -> List[float]:
    """The app's own scoring call, warm, one URL at a time (same method as `phishguard evaluate`)."""
    from phishguard.app.ml_layer1 import predict_layer1

    for u in urls[:20]:
        predict_layer1(u)
    out = []
    for u in urls:
        t0 = time.perf_counter()
        predict_layer1(u)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def record(model: str, n: int, host: str) -> Path:
    from phishguard.evaluation.evaluate import provenance

    try:
        version = _get(host, "/api/version").get("version")
        tags = {m["name"]: m for m in _get(host, "/api/tags").get("models", [])}
    except Exception as e:
        raise SystemExit(f"Could not reach Ollama at {host} ({e}). Install it from https://ollama.com, then run "
                         f"`ollama pull {model}` and try again.")
    if model not in tags:
        raise SystemExit(f"Model {model} is not downloaded. Run `ollama pull {model}` first.")
    s = draw_sample(n)
    print(f"sample: {len(s['urls'])} of {s['test_rows']} held-out rows", flush=True)
    l1_ms = time_layer1(s["urls"])
    print(f"layer 1 timed: p50 {np.percentile(l1_ms, 50):.1f} ms", flush=True)

    work = outputs_dir() / "llm_baseline" / (recording_path(model).name.replace(".jsonl.gz", f"_n{len(s['urls'])}.partial.jsonl"))
    work.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[int, Dict[str, Any]] = {}
    if work.is_file():
        for line in work.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["url"] == s["urls"][r["i"]]:
                done[r["i"]] = r
        print(f"resuming: {len(done)} answers already saved", flush=True)
    ask_llm(host, model, "https://example.com/")  # load the model before timing anything
    with open(work, "a", encoding="utf-8") as f:
        for i, u in enumerate(s["urls"]):
            if i in done:
                continue
            r = {"i": i, "url": u, **ask_llm(host, model, u)}
            done[i] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (len(done) % 50) == 0:
                print(f"{len(done)}/{len(s['urls'])}", flush=True)

    prov = provenance()
    meta = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": f"phishguard baseline-llm --record --n {n} --model {model}",
        "model": model, "model_digest": tags[model].get("digest"), "model_size_bytes": tags[model].get("size"),
        "model_details": tags[model].get("details"), "ollama_version": version, "options": OPTIONS,
        "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE, "max_url_chars": MAX_URL_CHARS,
        "sample": f"seeded simple random sample (seed {SEED}) of the held-out test rows", "n": len(s["urls"]),
        "test_rows": s["test_rows"], "test_csv_sha256": s["test_csv_sha256"], "bundle_sha256": s["bundle_sha256"],
        "primary_model": s["primary_model"], "hardware": f"{prov['environment']['cpu']} ({prov['environment']['cpu_count']} cores), CPU only",
        "git_commit": prov["git_commit"],
    }
    rows = [{"i": i, "url": u, "label": s["y"][i], "p_raw": round(float(s["p_raw"][i]), 6), "p_cal": round(float(s["p_cal"][i]), 6),
             "l1_ms": round(l1_ms[i], 3), "response": done[i]["response"], "llm_ms": round(done[i]["llm_ms"], 1),
             "prompt_tokens": done[i]["prompt_tokens"], "eval_tokens": done[i]["eval_tokens"]} for i, u in enumerate(s["urls"])]
    path = recording_path(model)
    write_recording(path, meta, rows)
    print("wrote", path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 vs a free local LLM on the same held-out URLs (replays a saved recording by default).")
    ap.add_argument("--record", action="store_true", help="ask a local Ollama model again instead of replaying the saved answers")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="sample size when recording")
    ap.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama address (your own machine)")
    ap.add_argument("--results-dir", type=Path, default=None)
    a = ap.parse_args()
    if a.record:
        record(a.model, a.n, a.host)
    replay(a.model, results_dir=a.results_dir)


if __name__ == "__main__":
    main()
