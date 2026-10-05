"""`phishguard cascade` and `phishguard baseline-llm`: the arithmetic, the file formats, and that the
committed result files are exactly what the committed inputs produce."""

import json

import numpy as np
import pytest

from phishguard.evaluation import cascade, llm_baseline
from phishguard.paths import project_root

RESULTS = project_root() / "metrics" / "results"
VOLATILE = {"generated_utc"}


def _strip(d):
    return {k: v for k, v in d.items() if k not in VOLATILE}


def _rows():
    # label, calibrated score, LLM answer
    spec = [(1, 0.95, "Phishing"), (1, 0.60, "phishing."), (1, 0.40, "legitimate"), (1, 0.05, "PHISHING"),
            (0, 0.02, "legitimate"), (0, 0.45, "Legitimate"), (0, 0.55, "legitimate"), (0, 0.97, "I cannot tell")]
    return [{"i": i, "url": f"http://site{i}.example/", "label": y, "p_raw": p, "p_cal": p, "l1_ms": 10.0 + i,
             "response": r, "llm_ms": 1000.0 + 100 * i, "prompt_tokens": 60, "eval_tokens": 2} for i, (y, p, r) in enumerate(spec)]


def test_parse_answer_accepts_only_a_clear_single_label():
    assert llm_baseline.parse_answer("Phishing") == 1
    assert llm_baseline.parse_answer(" legitimate.\n") == 0
    assert llm_baseline.parse_answer("This looks like PHISHING to me") == 1
    assert llm_baseline.parse_answer("phishing or legitimate") is None
    assert llm_baseline.parse_answer("unsure") is None
    assert llm_baseline.parse_answer("") is None and llm_baseline.parse_answer(None) is None


def test_recording_round_trips_and_is_byte_stable(tmp_path):
    meta, rows = {"model": "toy", "hardware": "test"}, _rows()
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    llm_baseline.write_recording(a, meta, rows)
    llm_baseline.write_recording(b, meta, rows)
    assert a.read_bytes() == b.read_bytes()
    meta2, rows2 = llm_baseline.read_recording(a)
    assert meta2 == meta and rows2 == rows


def test_score_recording_counts_by_hand():
    out = llm_baseline.score_recording({"model": "toy", "hardware": "test"}, _rows())
    assert out["sample"] == {"n": 8, "phishing": 4, "legitimate": 4}
    # Layer 1 at 0.5: flags rows 0, 1, 6, 7  ->  tp 2, fp 2, fn 2, tn 2
    assert out["layer1"]["confusion_matrix"] == {"tn": 2, "fp": 2, "fn": 2, "tp": 2}
    assert out["layer1"]["f1"] == pytest.approx(0.5)
    # LLM: phishing on rows 0, 1, 3; row 7 is unparsed and counts as legitimate  ->  tp 3, fp 0, fn 1, tn 4
    assert out["llm"]["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 1, "tp": 3}
    assert out["llm"]["unparsed_answers"] == 1
    assert out["layer1_minus_llm_f1"]["difference"] == pytest.approx(0.5 - 6 / 7)
    assert out["agreement"] == {"both_right": 4, "only_layer1_right": 0, "only_llm_right": 3, "both_wrong": 1}
    assert out["speed"]["llm_over_layer1_mean"] == pytest.approx(1350.0 / 13.5)


def test_band_row_routes_only_the_unsure_scores():
    a = llm_baseline.arrays(_rows())
    none = cascade.band_row(a["y"], a["p_cal"], 0.5, second=a["llm"])
    assert none["share_sent_to_slow_step"] == 0.0 and none["settled"]["n"] == 8
    assert none["end_to_end"]["f1"] == pytest.approx(0.5)  # nothing routed: plain Layer 1
    band = cascade.band_row(a["y"], a["p_cal"], 0.3, second=a["llm"])  # [0.3, 0.7): rows 1, 2, 5, 6
    assert band["share_sent_to_slow_step"] == 0.5
    # settled rows 0, 3, 4, 7: right on 0 and 4, wrong on 3 (missed phishing) and 7 (false alarm)
    assert band["settled"]["error_rate"] == pytest.approx(0.5)
    assert band["settled"]["phishing_called_legitimate"] == 1 and band["settled"]["legitimate_called_phishing"] == 1
    assert band["share_of_all_phishing_waved_through"] == pytest.approx(0.25)
    # final answers: L1 on 0,3,4,7 = 1,0,0,1 ; LLM on 1,2,5,6 = 1,0,0,0  ->  tp 2, fp 1, fn 2, tn 3
    assert band["end_to_end"]["precision"] == pytest.approx(2 / 3) and band["end_to_end"]["recall"] == pytest.approx(0.5)
    everything = cascade.band_row(a["y"], a["p_cal"], 0.0, second=a["llm"])
    assert everything["share_sent_to_slow_step"] == 1.0 and everything["settled"] is None
    assert everything["end_to_end"]["f1"] == pytest.approx(6 / 7)  # everything routed: plain LLM


def test_time_per_url():
    t = cascade.time_per_url(0.25, l1_ms=20.0, slow_ms=4000.0)
    assert t["ms_per_url"] == pytest.approx(1020.0)
    assert t["times_faster_than_slow_step_on_everything"] == pytest.approx(4000.0 / 1020.0)
    assert t["slow_calls_avoided"] == 0.75


def test_scores_file_round_trips_and_is_byte_stable(tmp_path):
    y, p = np.array([0, 1, 1]), np.array([0.125, 0.5, 0.987654])
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    cascade.write_scores(a, y, p, p)
    cascade.write_scores(b, y, p, p)
    assert a.read_bytes() == b.read_bytes()
    s = cascade.read_scores(a)
    assert list(s["y"]) == [0, 1, 1] and np.allclose(s["p_cal"], p, atol=1e-6)


@pytest.mark.skipif(not llm_baseline.recording_path().is_file() or not (RESULTS / "llm_baseline.json").is_file(),
                    reason="no committed LLM recording")
def test_committed_llm_result_is_what_the_recording_gives(tmp_path):
    out = llm_baseline.replay(results_dir=tmp_path)
    assert _strip(out) == _strip(json.loads((RESULTS / "llm_baseline.json").read_text()))
    meta = out["recording"]
    assert meta["options"]["temperature"] == 0 and meta["system_prompt"] == llm_baseline.SYSTEM_PROMPT
    assert out["sample"]["n"] == meta["n"]


@pytest.mark.skipif(not (RESULTS / cascade.SCORES_NAME).is_file() or not (RESULTS / "cascade.json").is_file(),
                    reason="no committed held-out scores")
def test_committed_cascade_result_is_what_the_scores_give(tmp_path):
    out = cascade.build(out_dir=tmp_path)
    assert _strip(out) == _strip(json.loads((RESULTS / "cascade.json").read_text()))
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    assert out["rows"] == ev["held_out_test"]["test_rows"]
    # band [0.5, 0.5) is plain Layer 1 at the app threshold: must equal the published confusion matrix
    cm = ev["primary"]["app_threshold_calibrated_ge_0_5"]["confusion_matrix"]
    s0 = out["bands"][0]["settled"]
    assert (s0["legitimate_called_phishing"], s0["phishing_called_legitimate"]) == (cm["fp"], cm["fn"])
