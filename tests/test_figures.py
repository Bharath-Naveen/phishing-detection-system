"""`phishguard curves` helpers and `phishguard figures` (charts drawn from the result files)."""

import json
import re

import numpy as np
import pytest

from phishguard.evaluation import curves
from phishguard.paths import project_root

RESULTS = project_root() / "metrics" / "results"


def test_roc_and_pr_points_keep_the_shape_of_the_curve():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = np.clip(y * 0.35 + rng.random(4000) * 0.65, 0, 1)
    roc = curves.roc_points(y, p)
    assert len(roc["fpr"]) == len(roc["tpr"]) == curves.GRID and roc["tpr"][-1] == 1.0
    assert all(b >= a for a, b in zip(roc["tpr"], roc["tpr"][1:]))
    from sklearn.metrics import roc_auc_score

    assert abs(np.trapz(roc["tpr"], roc["fpr"]) - roc_auc_score(y, p)) < 2e-3
    pr = curves.pr_points(y, p)
    assert len(pr["recall"]) == curves.GRID and all(0 <= v <= 1 for v in pr["precision"])


def test_calibration_bins_count_every_row_once():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.0, 0.05, 0.5, 0.95, 1.0, 0.31])
    bins = curves.calibration_bins(y, p)
    assert sum(b["n"] for b in bins) == len(y)
    assert bins[0]["n"] == 2 and bins[0]["observed_phishing_rate"] == 0.0
    assert bins[9]["n"] == 2 and bins[9]["observed_phishing_rate"] == 1.0
    assert bins[1]["n"] == 0 and bins[1]["mean_predicted"] is None
    hist = curves.score_histogram(y, p)
    assert sum(hist["legitimate"]) == 3 and sum(hist["phishing"]) == 3


@pytest.mark.skipif(not (RESULTS / "evaluation.json").is_file(), reason="no evaluation.json")
def test_figures_render_in_both_themes_and_match_the_results(tmp_path):
    pytest.importorskip("matplotlib")
    from phishguard.evaluation import figures

    res = figures.render(out_dir=tmp_path)
    assert res["drawn"], res
    for name in res["drawn"]:
        for theme in ("light", "dark"):
            svg = (tmp_path / f"{name}-{theme}.svg").read_text(encoding="utf-8")
            assert svg.lstrip().startswith("<?xml") and "</svg>" in svg
    # anything skipped must be for a missing input file, never an error in the drawing code
    assert all(v.startswith("missing input") for v in res["skipped"].values())

    index = (tmp_path / "README.md").read_text(encoding="utf-8")
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    x = ev["held_out_test"]["models"]["xgboost"]
    cm = x["confusion_matrix"]
    for value in (f"{x['f1']:.3f}", f"{x['roc_auc']:.3f}", f"{cm['tp']:,}", f"{cm['fp']:,}"):
        assert value in index, value
    # the table under each chart is generated: no hand-typed placeholder may survive
    assert not re.search(r"TODO|TBD|\bnan\b", index)


@pytest.mark.skipif(not (RESULTS / "curves.json").is_file(), reason="no curves.json")
def test_curves_file_agrees_with_evaluation_json():
    c = json.loads((RESULTS / "curves.json").read_text())
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    assert c["test_rows"] == ev["held_out_test"]["test_rows"]
    for m, v in c["models"].items():
        ref = ev["held_out_test"]["models"][m]
        tol = curves.REFIT_TOLERANCE.get(m, curves.AUC_TOLERANCE)
        assert abs(v["roc_auc"] - ref["roc_auc"]) <= tol
        assert abs(v["pr_auc"] - ref["pr_auc"]) <= tol
        assert len(v["roc"]["fpr"]) == len(v["roc"]["tpr"]) == c["grid_points"]
