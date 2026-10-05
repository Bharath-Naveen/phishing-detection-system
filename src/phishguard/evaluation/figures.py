"""`phishguard figures`: every chart in the README and on the website, drawn from the result files.

No number is typed here. Each figure reads the JSON that `phishguard evaluate`, `phishguard curves`,
`phishguard replay` and the pre-rebuild audit wrote, so the pictures cannot drift from the tables:

  metrics/results/evaluation.json                   held-out metrics, external sets, dashboard verdicts
  metrics/results/curves.json                       ROC / PR points, calibration bins, feature contributions
  metrics/results/tuning/{val,test}_{before,after}.json   full system on the frozen live snapshot
  metrics/audit_baseline/results/06_full_run.json   the same models before the rebuild

Output: docs/figures/<name>-light.svg and <name>-dark.svg, plus docs/figures/README.md with the
plotted values as tables (the text twin of every chart). Only matplotlib and numpy are needed.

    phishguard figures            # all figures
    phishguard figures --only roc_pr_curves confusion_matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from phishguard.paths import project_root  # noqa: E402

# --------------------------------------------------------------------------- style
# Colors were checked for color-vision-deficiency separation and contrast on both surfaces.
THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
        "red": "#e34948", "neutral": "#c3c2b7", "before": "#a9a79f",
        "seq": ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500",
        "red": "#e66767", "neutral": "#5a5955", "before": "#6f6d68",
        "seq": ["#0d366b", "#184f95", "#2a78d6", "#86b6ef"],
    },
}
# A model keeps its color in every figure.
MODEL_COLOR = {"xgboost": "blue", "lightgbm": "orange", "random_forest": "aqua", "logistic_regression": "yellow"}
MODEL_LABEL = {"xgboost": "XGBoost (primary)", "lightgbm": "LightGBM", "random_forest": "Random Forest",
               "logistic_regression": "Logistic Regression", "majority_class_baseline": "Majority-class baseline"}
MODEL_ORDER = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
VERDICTS = [("likely_phishing", "red", "Called phishing"), ("uncertain", "neutral", "Uncertain"),
            ("likely_legitimate", "blue", "Called legitimate")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "phishguard",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


class Fig:
    """One figure with a title block on top, a source line at the bottom and axes placed in inches."""

    def __init__(self, theme: str, w: float, h: float, title: str, subtitle: str, source: str):
        self.t = THEMES[theme]
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h), facecolor=self.t["surface"])
        self.fig.text(0.35 / w, 1 - 0.38 / h, title, fontsize=13.5, fontweight="bold", color=self.t["ink"], va="baseline")
        n_sub = subtitle.count("\n") + 1
        self.fig.text(0.35 / w, 1 - 0.62 / h, subtitle, fontsize=9, color=self.t["ink2"], va="top", linespacing=1.45)
        self.top = 0.62 + 0.19 * n_sub + 0.25  # inches used by the title block
        self.fig.text(0.35 / w, 0.16 / h, source, fontsize=7, color=self.t["muted"], va="baseline")

    def axes(self, left: float, top: float, width: float, height: float, grid: Optional[str] = "x"):
        """Axes placed in inches; `top` is measured from the bottom of the title block."""
        ax = self.fig.add_axes([left / self.w, 1 - (self.top + top + height) / self.h, width / self.w, height / self.h])
        t = self.t
        ax.set_facecolor(t["surface"])
        for s in ax.spines.values():
            s.set_color(t["axis"])
            s.set_linewidth(0.8)
        ax.tick_params(colors=t["muted"], labelsize=8, length=0, pad=4)
        if grid:
            ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
        return ax

    def panel_title(self, ax, text: str, note: str = ""):
        ax.text(0, 1.0, text, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=self.t["ink"], va="bottom",
                ha="left", clip_on=False, zorder=5, bbox=dict(facecolor="none", edgecolor="none", pad=0),
                position=(0, 1.0 + (0.34 if note else 0.12) / (ax.get_position().height * self.h)))
        if note:
            ax.text(0, 1.0 + 0.10 / (ax.get_position().height * self.h), note, transform=ax.transAxes, fontsize=8,
                    color=self.t["ink2"], va="bottom", ha="left", clip_on=False)

    def legend(self, items: List[Tuple[str, str, str]], x: float, y: float, gap: float = 0.0):
        """Legend row in figure inches from the top-left; items are (color key, label, 'box' or 'line')."""
        cx = x
        for color, label, kind in items:
            yy = 1 - y / self.h
            if kind == "dot":
                self.fig.add_artist(plt.Line2D([(cx + 0.07) / self.w], [yy], marker="o", markersize=7.5, color=self.t[color],
                                               markeredgecolor=self.t["surface"], linestyle="none"))
                cx -= 0.08
            elif kind == "line":
                self.fig.add_artist(plt.Line2D([cx / self.w, (cx + 0.22) / self.w], [yy, yy], color=self.t[color],
                                               linewidth=2, solid_capstyle="round"))
            else:
                self.fig.patches.append(FancyBboxPatch((cx / self.w, yy - 0.05 / self.h), 0.14 / self.w, 0.10 / self.h,
                                                       boxstyle="round,pad=0,rounding_size=0.002", transform=self.fig.transFigure,
                                                       facecolor=self.t[color], edgecolor="none"))
                cx -= 0.08
            txt = self.fig.text((cx + 0.30) / self.w, yy, label, fontsize=8.5, color=self.t["ink2"], va="center")
            self.fig.canvas.draw()
            cx += 0.30 + txt.get_window_extent().width / self.fig.dpi + 0.28 + gap

    def save(self, out_dir: Path, name: str, theme: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{name}-{theme}.svg"
        self.fig.savefig(p, format="svg", facecolor=self.t["surface"], metadata={"Date": None, "Creator": "phishguard figures"})
        plt.close(self.fig)
        return p


def hbar(ax, y: float, x0: float, x1: float, height: float, color: str, round_end: bool = True):
    """Horizontal bar from x0 to x1, square at the start, rounded at the data end."""
    fig = ax.figure
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_position().width * fig.get_figwidth() * fig.dpi)
    y_per_px = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_position().height * fig.get_figheight() * fig.dpi)
    r = 4.0 if round_end else 0.0
    rx, ry = min(r * x_per_px, max(x1 - x0, 0) / 2), min(r * y_per_px, height / 2)
    lo, hi = y - height / 2, y + height / 2
    verts = [(x0, lo), (x1 - rx, lo), (x1, lo), (x1, lo + ry), (x1, hi - ry), (x1, hi), (x1 - rx, hi), (x0, hi), (x0, lo)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def vbar(ax, x: float, y1: float, width: float, color: str):
    """Vertical bar from 0 to y1, square at the baseline, rounded at the top."""
    fig = ax.figure
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_position().width * fig.get_figwidth() * fig.dpi)
    y_per_px = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_position().height * fig.get_figheight() * fig.dpi)
    rx, ry = min(4.0 * x_per_px, width / 2), min(4.0 * y_per_px, y1 / 2)
    lo, hi = x - width / 2, x + width / 2
    verts = [(lo, 0), (lo, y1 - ry), (lo, y1), (lo + rx, y1), (hi - rx, y1), (hi, y1), (hi, y1 - ry), (hi, 0), (lo, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def stacked(ax, y: float, counts: List[int], colors: List[str], height: float, t: Dict[str, str],
            labels: bool = True):
    """100% stacked horizontal bar with a 2px surface gap between segments and counts inside when they fit."""
    total = float(sum(counts))
    gap = 2 * 100.0 / (ax.get_position().width * ax.figure.get_figwidth() * ax.figure.dpi)
    live = [i for i, c in enumerate(counts) if c > 0]
    x = 0.0
    for i in live:
        w = 100.0 * counts[i] / total
        a, b = x + (gap / 2 if i != live[0] else 0), x + w - (gap / 2 if i != live[-1] else 0)
        hbar(ax, y, a, max(b, a), height, t[colors[i]], round_end=(i == live[-1]))
        seg_in = (w / 100.0) * ax.get_position().width * ax.figure.get_figwidth()
        if labels and seg_in >= len(str(counts[i])) * 0.075 + 0.08:
            ink = "#ffffff" if colors[i] in ("red", "blue") else t["ink"]
            ax.text(x + w / 2, y, f"{counts[i]}", ha="center", va="center", fontsize=8.5, color=ink, zorder=4)
        x += w


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self, root: Path):
        r = root / "metrics" / "results"
        self.root = root
        self.ev = json.loads((r / "evaluation.json").read_text())
        self.curves = json.loads((r / "curves.json").read_text()) if (r / "curves.json").is_file() else None
        self.tuning = {k: json.loads((r / "tuning" / f"{k}.json").read_text())
                       for k in ("val_before", "val_after", "test_before", "test_after")
                       if (r / "tuning" / f"{k}.json").is_file()}
        b = root / "metrics" / "audit_baseline" / "results" / "06_full_run.json"
        self.baseline = json.loads(b.read_text()) if b.is_file() else None
        self.commit = str(self.ev["provenance"]["git_commit"])[:7]

    def src(self, *files: str) -> str:
        return "Source: " + ", ".join(files) + f"  ·  evaluated at commit {self.commit}  ·  drawn by `phishguard figures`"


Table = Tuple[List[str], List[List[str]]]
FIGURES: Dict[str, Tuple[str, Callable[[Data, str, Path], Table]]] = {}


def figure(name: str, caption: str):
    def deco(fn):
        FIGURES[name] = (caption, fn)
        return fn
    return deco


# --------------------------------------------------------------------------- 1. before / after the rebuild
@figure("rebuild_before_after", "Held-out scores before and after the evaluation was fixed")
def fig_rebuild(d: Data, theme: str, out: Path) -> Table:
    if d.baseline is None:
        raise FileNotFoundError("metrics/audit_baseline/results/06_full_run.json")
    before, after = d.baseline["held_out_test_threshold_0_5_raw"], d.ev["held_out_test"]["models"]
    f = Fig(theme, 9.2, 5.0, "Fixing the evaluation lowered every score, and that is the point",
            "Same four models, full-data run, before and after the rebuild. The rebuild removed three shortcuts: an http/https\n"
            "artifact in the data, evaluation websites sitting in the training set, and a validation set that shared sites with training.",
            d.src("metrics/audit_baseline/results/06_full_run.json", "metrics/results/evaluation.json"))
    t = f.t
    rows: List[List[str]] = []
    for k, (metric, label) in enumerate((("roc_auc", "ROC-AUC"), ("f1", "F1 at threshold 0.5"))):
        ax = f.axes(1.75 + k * 3.75, 0.55, 3.2, 2.05)
        ax.set_xlim(0.70, 1.0)
        ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
        ax.set_xticks([0.70, 0.80, 0.90, 1.00])
        ax.set_xticklabels(["0.70", "0.80", "0.90", "1.00"])
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_LABEL[m].replace(" (primary)", "") for m in MODEL_ORDER] if k == 0 else [])
        ax.tick_params(axis="y", labelsize=8.5, labelcolor=t["ink2"])
        ax.spines["left"].set_visible(False)
        f.panel_title(ax, label)
        for i, m in enumerate(MODEL_ORDER):
            b, a = before[m][metric], after[m][metric]
            ax.plot([a, b], [i, i], color=t["axis"], linewidth=2, zorder=2, solid_capstyle="round")
            ax.scatter([b], [i], s=62, color=t["before"], edgecolor=t["surface"], linewidth=1.5, zorder=3)
            ax.scatter([a], [i], s=62, color=t["blue"], edgecolor=t["surface"], linewidth=1.5, zorder=3)
            ax.text(b + 0.012, i, f"{b:.3f}", va="center", ha="left", fontsize=8, color=t["ink2"])
            ax.text(a - 0.012, i, f"{a:.3f}", va="center", ha="right", fontsize=8, color=t["ink"])
            rows.append([MODEL_LABEL[m], label, f"{b:.3f}", f"{a:.3f}", f"{a - b:+.3f}"])
    f.legend([("before", "Before the rebuild (flawed evaluation)", "dot"), ("blue", "After the rebuild (verified)", "dot")], 0.43, f.top + 0.02)
    nb, na = d.baseline["split"]["n_test"], d.ev["held_out_test"]["test_rows"]
    f.fig.text(0.35 / f.w, 0.42 / f.h,
               f"The two test sets are not identical: {nb:,} URLs before, {na:,} after (rows on evaluation websites were removed).\n"
               "Read the gap as direction, not as an exact measurement.", fontsize=8, color=t["ink2"], va="bottom", linespacing=1.4)
    f.save(out, "rebuild_before_after", theme)
    return ["Model", "Metric", "Before rebuild", "After rebuild", "Change"], rows


# --------------------------------------------------------------------------- 2. layers on real brand sites
@figure("layers_on_brand_sites", "What each layer does to 136 real brand URLs")
def fig_layers(d: Data, theme: str, out: Path) -> Table:
    ext = d.ev["external_sets"]["official_brand_test_half"]
    n = ext["n"]
    flagged = int(round(ext["primary_flag_rate_app_threshold"] * n))
    ml = d.ev["dashboard_ml_only"]["sets"]["official_brand_test_half"]["verdicts"]
    full = d.tuning["test_after"]["summary"]["per_set"]["official_brand"]
    assert full["n"] == n
    fv = full["verdicts"]
    f = Fig(theme, 9.2, 3.65, f"The layers stop false alarms on real brand sites: {flagged} of {n} flagged becomes {fv['likely_phishing']}",
            f"{n} official URLs from PayPal, Microsoft, Chase, Amazon and others. None of their websites were in the training data.\n"
            "Each bar is the same set of URLs at a later stage of the system.",
            d.src("metrics/results/evaluation.json", "metrics/results/tuning/test_after.json"))
    t = f.t
    ax = f.axes(2.75, 0.25, 6.0, 1.75, grid=None)
    ax.set_xlim(0, 100)
    ax.set_ylim(2.6, -0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    stages = [
        ("URL model alone", "score of 0.5 or more counts as flagged", [flagged, 0, n - flagged]),
        ("Plus the adjudication layer", "no page capture: it declines to accuse", [ml["likely_phishing"], ml["uncertain"], ml["likely_legitimate"]]),
        ("Full system", "page captured and inspected", [fv["likely_phishing"], fv["uncertain"], fv["likely_legitimate"]]),
    ]
    rows = []
    for i, (name, note, counts) in enumerate(stages):
        stacked(ax, i, counts, [c for _, c, _ in VERDICTS], 0.44, t)
        ax.text(-2.5, i - 0.10, name, ha="right", va="center", fontsize=9.5, color=t["ink"], fontweight="bold")
        ax.text(-2.5, i + 0.20, note, ha="right", va="center", fontsize=7.8, color=t["ink2"])
        rows.append([name, str(counts[0]), str(counts[1]), str(counts[2]), pct(counts[0] / n)])
    ax.set_yticks([])
    f.legend([("red", "Flagged or called phishing (a false alarm here)", "box"), ("neutral", "Uncertain", "box"),
              ("blue", "Not flagged or called legitimate", "box")], 0.43, f.top + 0.02)
    f.save(out, "layers_on_brand_sites", theme)
    return ["Stage", "Flagged / phishing", "Uncertain", "Not flagged / legitimate", "False-alarm rate"], rows


# --------------------------------------------------------------------------- 3. model comparison
@figure("model_comparison", "Four models and a baseline on the held-out test set")
def fig_models(d: Data, theme: str, out: Path) -> Table:
    held = d.ev["held_out_test"]["models"]
    n = d.ev["held_out_test"]["test_rows"]
    order = MODEL_ORDER + ["majority_class_baseline"]
    half = max((held[m]["bootstrap_95ci"][k][1] - held[m]["bootstrap_95ci"][k][0]) / 2
               for m in MODEL_ORDER for k in ("f1", "roc_auc", "pr_auc") if "bootstrap_95ci" in held[m])
    f = Fig(theme, 9.2, 4.45, "Tree models beat the linear model, and every model beats the baseline",
            f"{n:,} held-out URLs from websites never seen in training. Bootstrap 95% intervals are within ±{half:.3f} of every\n"
            "value, too narrow to draw. XGBoost was chosen on validation data, not on these test scores.",
            d.src("metrics/results/evaluation.json"))
    t = f.t
    rows = []
    for k, (metric, label) in enumerate((("f1", "F1 at threshold 0.5"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"))):
        ax = f.axes(1.95 + k * 2.42, 0.45, 2.0, 2.05)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1.0"])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in order] if k == 0 else [])
        ax.tick_params(axis="y", labelsize=8.5, labelcolor=t["ink2"])
        ax.spines["left"].set_color(t["axis"])
        f.panel_title(ax, label)
        for i, m in enumerate(order):
            v = held[m][metric]
            color = t[MODEL_COLOR[m]] if m in MODEL_COLOR else t["before"]
            if v > 0:
                hbar(ax, i, 0, v, 0.46, color)
            ax.text(v + 0.03, i, f"{v:.3f}", va="center", ha="left", fontsize=8.2, color=t["ink"])
    for m in order:
        ci = held[m].get("bootstrap_95ci", {}).get("f1")
        rows.append([MODEL_LABEL[m], f"{held[m]['f1']:.3f}", f"{ci[0]:.3f} to {ci[1]:.3f}" if ci else "n/a",
                     f"{held[m]['roc_auc']:.3f}", f"{held[m]['pr_auc']:.3f}", f"{held[m]['precision']:.3f}",
                     f"{held[m]['recall']:.3f}", pct(held[m]["false_positive_rate"])])
    f.save(out, "model_comparison", theme)
    return ["Model", "F1", "F1 95% CI", "ROC-AUC", "PR-AUC", "Precision", "Recall", "False-positive rate"], rows


# --------------------------------------------------------------------------- 4. ROC and PR curves
@figure("roc_pr_curves", "ROC and precision-recall curves on the held-out test set")
def fig_curves(d: Data, theme: str, out: Path) -> Table:
    if d.curves is None:
        raise FileNotFoundError("metrics/results/curves.json (run `phishguard curves`)")
    held = d.ev["held_out_test"]["models"]
    n = d.curves["test_rows"]
    prev = d.curves["test_phishing_rows"] / n
    f = Fig(theme, 9.2, 5.9, "The three tree models are close; logistic regression trails throughout",
            f"{n:,} held-out URLs from unseen websites. The dot marks XGBoost at the 0.5 threshold. Legend numbers: ROC-AUC / PR-AUC.",
            d.src("metrics/results/curves.json", "metrics/results/evaluation.json"))
    t = f.t
    rows = []
    ax1 = f.axes(0.9, 0.72, 3.5, 3.15, grid="both")
    ax2 = f.axes(5.3, 0.72, 3.5, 3.15, grid="both")
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.plot([0, 1], [0, 1], color=t["axis"], linewidth=1.2, zorder=2)
    ax1.text(0.87, 0.82, "chance", color=t["muted"], fontsize=8, rotation=42, ha="center", va="center")
    ax2.plot([0, 1], [prev, prev], color=t["axis"], linewidth=1.2, zorder=2)
    ax2.text(0.02, prev - 0.035, f"baseline: {pct(prev)} of test URLs are phishing", color=t["muted"], fontsize=8, va="top")
    for m in reversed(MODEL_ORDER):
        c = d.curves["models"][m]
        ax1.plot([0.0, *c["roc"]["fpr"]], [0.0, *c["roc"]["tpr"]], color=t[MODEL_COLOR[m]], linewidth=2, solid_capstyle="round", zorder=3)
        ax2.plot(c["pr"]["recall"][1:], c["pr"]["precision"][1:], color=t[MODEL_COLOR[m]], linewidth=2, solid_capstyle="round", zorder=3)
    for m in MODEL_ORDER:
        rows.append([MODEL_LABEL[m], f"{held[m]['roc_auc']:.3f}", f"{held[m]['pr_auc']:.3f}",
                     f"{d.curves['models'][m]['roc_auc']:.4f}", f"{d.curves['models'][m]['pr_auc']:.4f}"])
    x = held["xgboost"]
    ax1.scatter([x["false_positive_rate"]], [x["recall"]], s=70, color=t["blue"], edgecolor=t["surface"], linewidth=2, zorder=5)
    ax1.annotate(f"catches {pct(x['recall'])} of phishing,\nflags {pct(x['false_positive_rate'])} of legitimate",
                 (x["false_positive_rate"], x["recall"]), xytext=(0.42, 0.56), fontsize=8, color=t["ink2"],
                 arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=0.8, shrinkA=2, shrinkB=6), va="top")
    ax2.scatter([x["recall"]], [x["precision"]], s=70, color=t["blue"], edgecolor=t["surface"], linewidth=2, zorder=5)
    ax2.annotate(f"precision {x['precision']:.3f}\nat recall {x['recall']:.3f}", (x["recall"], x["precision"]), xytext=(0.52, 0.985),
                 fontsize=8, color=t["ink2"], va="top",
                 arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=0.8, shrinkA=2, shrinkB=6))
    ax1.set_xlabel("False-positive rate (legitimate URLs flagged)", color=t["ink2"], fontsize=8.5, labelpad=6)
    ax1.set_ylabel("Recall (phishing URLs caught)", color=t["ink2"], fontsize=8.5, labelpad=6)
    ax2.set_xlabel("Recall (phishing URLs caught)", color=t["ink2"], fontsize=8.5, labelpad=6)
    ax2.set_ylabel("Precision (flags that are right)", color=t["ink2"], fontsize=8.5, labelpad=6)
    f.panel_title(ax1, "ROC curve")
    f.panel_title(ax2, "Precision-recall curve")
    f.legend([(MODEL_COLOR[m], f"{MODEL_LABEL[m].replace(' (primary)', '')}  {held[m]['roc_auc']:.3f} / {held[m]['pr_auc']:.3f}", "line")
              for m in MODEL_ORDER], 0.43, f.top + 0.02, gap=-0.1)
    f.save(out, "roc_pr_curves", theme)
    return ["Model", "ROC-AUC (evaluation.json)", "PR-AUC (evaluation.json)", "ROC-AUC of the drawn curve", "PR-AUC of the drawn curve"], rows


# --------------------------------------------------------------------------- 5. confusion matrix
@figure("confusion_matrix", "Confusion matrix of the primary model on the held-out test set")
def fig_confusion(d: Data, theme: str, out: Path) -> Table:
    x = d.ev["held_out_test"]["models"][d.ev["primary"]["model"]]
    cm = x["confusion_matrix"]
    n = x["n"]
    f = Fig(theme, 7.0, 4.9, f"XGBoost catches {pct(x['recall'])} of phishing with {pct(x['false_positive_rate'])} false alarms",
            f"{n:,} held-out URLs from unseen websites, threshold 0.5 on the raw model probability.\n"
            "Percentages are shares of each row (of what the URL really was).",
            d.src("metrics/results/evaluation.json"))
    t = f.t
    ax = f.axes(1.95, 0.55, 4.6, 2.6, grid=None)
    cells = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    names = [["Correctly passed", "False alarm"], ["Missed", "Correctly caught"]]
    vmax = max(max(r) for r in cells)
    seq = t["seq"]
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    for i in range(2):
        tot = sum(cells[i])
        for j in range(2):
            frac = cells[i][j] / tot
            step = min(int(cells[i][j] / vmax * len(seq)), len(seq) - 1)
            ax.add_patch(FancyBboxPatch((j + 0.02, i + 0.03), 0.96, 0.94, boxstyle="round,pad=0,rounding_size=0.03",
                                        facecolor=seq[step], edgecolor="none", mutation_aspect=0.62))
            light_cell = (theme == "light" and step < 2) or (theme == "dark" and step == 3)
            ink = "#0b0b0b" if light_cell else "#ffffff"
            ax.text(j + 0.5, i + 0.40, f"{cells[i][j]:,}", ha="center", va="center", fontsize=17, color=ink, fontweight="bold")
            ax.text(j + 0.5, i + 0.66, f"{names[i][j]} · {pct(frac)}", ha="center", va="center", fontsize=8.5, color=ink)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Model says legitimate", "Model says phishing"])
    ax.xaxis.tick_top()
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([f"Really legitimate\n{cm['tn'] + cm['fp']:,} URLs", f"Really phishing\n{cm['fn'] + cm['tp']:,} URLs"])
    ax.tick_params(axis="both", labelsize=9, labelcolor=t["ink2"], length=0, pad=6)
    f.save(out, "confusion_matrix", theme)
    rows = [["Really legitimate", f"{cm['tn']:,}", f"{cm['fp']:,}"], ["Really phishing", f"{cm['fn']:,}", f"{cm['tp']:,}"]]
    return ["", "Model says legitimate", "Model says phishing"], rows


# --------------------------------------------------------------------------- 6. verdicts by evaluation set
@figure("verdicts_by_set", "Final verdicts per evaluation set, with and without page capture")
def fig_verdicts(d: Data, theme: str, out: Path) -> Table:
    ta = d.tuning["test_after"]["summary"]
    ps = ta["per_set"]
    rec = ta["phishing_recall"]
    fresh = [rec["likely_phishing"], rec["n"] - rec["likely_phishing"] - rec["likely_legitimate"], rec["likely_legitimate"]]
    ml = d.ev["dashboard_ml_only"]["sets"]

    def v(x):
        return [x["verdicts"]["likely_phishing"], x["verdicts"]["uncertain"], x["verdicts"]["likely_legitimate"]]

    groups = [
        ("With live page capture (frozen snapshot, test split)", [
            (f"Official brand URLs ({ps['official_brand']['n']})", "should be legitimate", v(ps["official_brand"])),
            (f"Popular homepages ({ps['tranco']['n']})", "should be legitimate", v(ps["tranco"])),
            (f"Fresh phishing feed ({rec['n']})", "should be phishing", fresh),
        ]),
        ("URL only, no page capture", [
            (f"Official brand URLs ({ml['official_brand_test_half']['n']})", "should be legitimate", v(ml["official_brand_test_half"])),
            (f"PhishStats phishing URLs ({ml['phishstats']['n']})", "should be phishing", v(ml["phishstats"])),
        ]),
    ]
    tr_fp = [r for r in d.tuning["test_after"]["rows"] if r["set"] == "tranco" and r["verdict"] == "likely_phishing"]
    tr_failed = sum(1 for r in tr_fp if not r["capture_ok"])
    assert len(tr_fp) == ps["tranco"]["verdicts"]["likely_phishing"]
    f = Fig(theme, 9.2, 6.0, f"Brand sites are mostly cleared; fresh phishing is the weak spot ({rec['likely_phishing']} of {rec['n']} caught)",
            f"Final verdict of the whole system on each evaluation set. {rec['likely_legitimate']} of {rec['n']} fresh phishing pages were called legitimate.\n"
            f"Of the {len(tr_fp)} popular homepages called phishing, {tr_failed} were pages that failed to load. Without a capture the system mostly answers uncertain.",
            d.src("metrics/results/tuning/test_after.json", "metrics/results/evaluation.json"))
    t = f.t
    rows = []
    top = 0.62
    for title, sets in groups:
        hgt = 0.48 * len(sets)
        ax = f.axes(2.95, top, 5.8, hgt, grid=None)
        ax.set_xlim(0, 100)
        ax.set_ylim(len(sets) - 0.5, -0.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        f.fig.text(0.35 / f.w, 1 - (f.top + top - 0.16) / f.h, title, fontsize=9.5, fontweight="bold", color=t["ink"], va="bottom")
        for i, (name, want, counts) in enumerate(sets):
            stacked(ax, i, counts, [c for _, c, _ in VERDICTS], 0.5, t)
            ax.text(-2.5, i - 0.14, name, ha="right", va="center", fontsize=9, color=t["ink"])
            ax.text(-2.5, i + 0.20, want, ha="right", va="center", fontsize=7.8, color=t["muted"])
            rows.append([title, name, want, *[str(c) for c in counts]])
        top += hgt + 0.62
    f.legend([(c, lab, "box") for _, c, lab in VERDICTS], 0.43, f.top + 0.02)
    f.fig.text(0.35 / f.w, 0.42 / f.h,
               "Fresh phishing counts pages whose capture succeeded and whose host is not a top-10K site. Feed labels mean reported as phishing;\n"
               "some pages had already been replaced by harmless content when captured.", fontsize=7.8, color=t["ink2"], linespacing=1.4, va="bottom")
    f.save(out, "verdicts_by_set", theme)
    return ["Mode", "Set", "Expected", "Called phishing", "Uncertain", "Called legitimate"], rows


# --------------------------------------------------------------------------- 7. rule tuning
@figure("rule_tuning", "Rule tuning on a frozen live snapshot, before and after")
def fig_tuning(d: Data, theme: str, out: Path) -> Table:
    tu = d.tuning
    f = Fig(theme, 9.2, 4.3, "Rule tuning cut false alarms without touching phishing detection",
            "The same saved page captures replayed with the old and the new rules. Rules were changed using the validation split only;\n"
            "the test split was scored once with each rule set.",
            d.src("metrics/results/tuning/{val,test}_{before,after}.json"))
    t = f.t
    rows = []
    panels = [("legit_false_alarms", "Legitimate pages called phishing", "lower is better"),
              ("phishing_recall", "Fresh phishing pages caught", "higher is better")]
    for k, (key, label, note) in enumerate(panels):
        ax = f.axes(1.35 + k * 4.05, 0.75, 3.3, 1.75)
        ax.set_xlim(0, 60)
        ax.set_ylim(1.62, -0.62)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xticklabels(["0%", "20%", "40%", "60%"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Validation", "Test"] if k == 0 else [])
        ax.tick_params(axis="y", labelsize=9, labelcolor=t["ink2"])
        f.panel_title(ax, label, note)
        for i, split in enumerate(("val", "test")):
            for j, (when, color) in enumerate((("before", "before"), ("after", "blue"))):
                s = tu[f"{split}_{when}"]["summary"][key]
                y = i + (j - 0.5) * 0.36
                hbar(ax, y, 0, 100 * s["rate"], 0.30, t[color])
                ax.text(100 * s["rate"] + 1.2, y, f"{s['likely_phishing']} of {s['n']}  ({pct(s['rate'])})", va="center",
                        fontsize=8.2, color=t["ink"] if when == "after" else t["ink2"])
                rows.append([label, "Validation" if split == "val" else "Test", when, f"{s['likely_phishing']} of {s['n']}", pct(s["rate"])])
    f.legend([("before", "Old rules", "box"), ("blue", "New rules", "box")], 0.43, f.top + 0.02)
    f.save(out, "rule_tuning", theme)
    return ["Measure", "Split", "Rules", "Count", "Rate"], rows


# --------------------------------------------------------------------------- 8. feature contributions
@figure("feature_contributions", "Which URL features move the primary model's score most")
def fig_features(d: Data, theme: str, out: Path) -> Table:
    fc = (d.curves or {}).get("primary", {}).get("feature_contributions")
    if not fc:
        raise FileNotFoundError("feature contributions in metrics/results/curves.json (run `phishguard curves`)")
    feats = fc["features"]
    top = feats[:12]
    total = sum(x["mean_abs_log_odds"] for x in feats)
    share = sum(x["mean_abs_log_odds"] for x in top) / total
    f = Fig(theme, 9.2, 5.6, f"What the URL model looks at: top {len(top)} of {len(feats)} features",
            f"Average size of each feature's push on XGBoost's score (log-odds), over {fc['n_rows']:,} held-out URLs.\n"
            f"These {len(top)} features carry {pct(share, 0)} of the total. Size only: a feature can push either way depending on its value.",
            d.src("metrics/results/curves.json"))
    t = f.t
    ax = f.axes(3.0, 0.1, 5.5, 3.3)
    vmax = top[0]["mean_abs_log_odds"]
    ax.set_xlim(0, vmax * 1.14)
    ax.set_ylim(len(top) - 0.4, -0.6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([x["feature"] for x in top], family="DejaVu Sans Mono")
    ax.tick_params(axis="y", labelsize=8.3, labelcolor=t["ink2"])
    ax.set_xlabel("Mean absolute contribution (log-odds)", color=t["ink2"], fontsize=8.5, labelpad=6)
    for i, x in enumerate(top):
        hbar(ax, i, 0, x["mean_abs_log_odds"], 0.5, t["blue"])
        ax.text(x["mean_abs_log_odds"] + vmax * 0.015, i, f"{x['mean_abs_log_odds']:.3f}", va="center", fontsize=8.2, color=t["ink"])
    f.save(out, "feature_contributions", theme)
    return ["Rank", "Feature", "Mean absolute contribution (log-odds)", "Mean signed contribution"], \
        [[str(i + 1), f"`{x['feature']}`", f"{x['mean_abs_log_odds']:.4f}", f"{x['mean_signed_log_odds']:+.4f}"] for i, x in enumerate(feats)]


# --------------------------------------------------------------------------- 9. calibration
@figure("calibration", "Calibration of the primary model's probability")
def fig_calibration(d: Data, theme: str, out: Path) -> Table:
    cal = (d.curves or {}).get("primary", {}).get("calibration")
    if not cal:
        raise FileNotFoundError("calibration bins in metrics/results/curves.json (run `phishguard curves`)")
    br, bc = cal["brier_raw"], cal["brier_calibrated"]
    verdict = "does not help here" if bc >= br else "helps"
    f = Fig(theme, 6.8, 6.5, f"Calibration: the isotonic step {verdict}",
            f"When the model says 70%, how often is it phishing? {d.curves['test_rows']:,} held-out URLs in 10 probability bins.\n"
            f"Brier score (lower is better): {br:.4f} raw, {bc:.4f} calibrated. An open item, reported as measured.",
            d.src("metrics/results/curves.json"))
    t = f.t
    ax = f.axes(1.0, 0.6, 5.4, 2.8, grid="both")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.plot([0, 1], [0, 1], color=t["axis"], linewidth=1.2, zorder=2)
    ax.text(0.30, 0.345, "perfect", color=t["muted"], fontsize=8, rotation=27.4, ha="center", va="center")
    rows = []
    for key, color, label in (("raw", "blue", "Raw"), ("calibrated", "orange", "Calibrated")):
        pts = [(b["mean_predicted"], b["observed_phishing_rate"]) for b in cal[key] if b["n"]]
        ax.plot(*zip(*pts), color=t[color], linewidth=2, zorder=3, solid_capstyle="round")
        ax.scatter(*zip(*pts), s=42, color=t[color], edgecolor=t["surface"], linewidth=1.5, zorder=4)
        for b in cal[key]:
            rows.append([label, f"{b['lo']:.1f} to {b['hi']:.1f}", f"{b['n']:,}",
                         f"{b['mean_predicted']:.3f}" if b["n"] else "n/a", f"{b['observed_phishing_rate']:.3f}" if b["n"] else "n/a"])
    top_bin = cal["raw"][-1]
    if top_bin["n"] and top_bin["mean_predicted"] - top_bin["observed_phishing_rate"] > 0.05:
        ax.annotate(f"Overconfident at the top: says {pct(top_bin['mean_predicted'], 0)},\nis right {pct(top_bin['observed_phishing_rate'], 0)} of the time",
                    (top_bin["mean_predicted"], top_bin["observed_phishing_rate"]), xytext=(0.50, 0.30), fontsize=8, color=t["ink2"],
                    va="top", arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=0.8, shrinkA=2, shrinkB=6))
    ax.set_ylabel("Share that really is phishing", color=t["ink2"], fontsize=8.5, labelpad=6)
    f.legend([("blue", f"Raw model probability (Brier {br:.4f})", "line"), ("orange", f"After isotonic calibration (Brier {bc:.4f})", "line")],
             0.43, f.top + 0.02)
    # how many URLs fall in each raw-probability bin, on the same x-axis, so sparse bins are visible
    ax2 = f.axes(1.0, 0.6 + 2.8 + 0.22, 5.4, 0.72, grid=None)
    counts = [b["n"] for b in cal["raw"]]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, max(counts) * 1.08)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_yticks([0, max(counts)])
    ax2.set_yticklabels(["0", f"{max(counts):,}"])
    for b in cal["raw"]:
        if b["n"]:
            vbar(ax2, (b["lo"] + b["hi"]) / 2, b["n"], 0.07, t["before"])
    ax2.set_xlabel("Predicted probability of phishing (bars: URLs per raw-probability bin)", color=t["ink2"], fontsize=8.5, labelpad=6)
    f.save(out, "calibration", theme)
    return ["Probability", "Bin", "URLs", "Mean predicted", "Observed phishing rate"], rows


# --------------------------------------------------------------------------- 10. architecture
@figure("architecture", "How a URL moves through the system")
def fig_architecture(d: Data, theme: str, out: Path) -> Table:
    nfeat = d.ev["data"]["n_model_features"]
    lat = d.ev["latency_ms"]["layer1_only"]["p50"]
    f = Fig(theme, 11.0, 4.3, "How a URL becomes a verdict",
            "Five layers. The first needs only the URL text; the next two look at the live page; the last two weigh brand, trust and evidence.\n"
            "No external AI service is called, so the same input always gives the same verdict.",
            d.src("metrics/results/evaluation.json", "docs/rebuild/HOW_IT_WORKS.md"))
    t = f.t
    ax = f.axes(0.35, -0.12, 10.3, 2.85, grid=None)
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 2.85)
    ax.axis("off")
    steps = [
        ("1", "URL model", f"XGBoost scores {nfeat}\nfeatures of the URL\ntext. Three more\nmodels vote. About\n{lat:.0f} ms, no network.", "blue"),
        ("2", "Live capture", "A headless browser\nloads the page:\nredirects, form\ntargets, TLS state.", "neutral"),
        ("3", "Page analysis", "Login harvesters,\nwrapper pages,\nforms posting to\nanother site.", "neutral"),
        ("4", "Brand and trust", "Does the brand\nmatch the domain?\nOfficial-domain\nregistry as a weak\nprior.", "blue"),
        ("5", "Adjudication", "Adds up phishing,\nlegitimacy and\nambiguity signals.\nHard blockers. Same\nrules every time.", "blue"),
    ]
    bw, bh, gap, x0, y0 = 1.5, 1.72, 0.22, 0.74, 0.42
    box_edge = t["axis"]
    ax.text(0.27, y0 + bh / 2, "URL", ha="center", va="center", fontsize=10, fontweight="bold", color=t["ink"],
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8", facecolor=t["surface"], edgecolor=box_edge, linewidth=1))

    def arrow(xa, xb, y):
        ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>", mutation_scale=9, color=t["muted"], linewidth=1.1,
                                     shrinkA=0, shrinkB=0))

    arrow(0.56, x0 - 0.03, y0 + bh / 2)
    for i, (num, name, text, color) in enumerate(steps):
        x = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.08", facecolor=t["surface"],
                                    edgecolor=box_edge, linewidth=1))
        ax.add_patch(FancyBboxPatch((x, y0 + bh - 0.07), bw, 0.07, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=t[color], edgecolor="none"))
        ax.text(x + 0.13, y0 + bh - 0.31, f"{num}  {name}", fontsize=8.8, fontweight="bold", color=t["ink"], va="center")
        ax.text(x + 0.13, y0 + bh - 0.54, text, fontsize=7.7, color=t["ink2"], va="top", linespacing=1.5)
        if i < len(steps) - 1:
            arrow(x + bw + 0.03, x + bw + gap - 0.03, y0 + bh / 2)
    # the path when no page is captured
    xa, xb = x0 + bw / 2, x0 + 3 * (bw + gap) + bw / 2
    ax.add_patch(FancyArrowPatch((xa, y0 + bh + 0.03), (xb, y0 + bh + 0.03), connectionstyle="bar,fraction=-0.07",
                                 arrowstyle="-|>", mutation_scale=9, color=t["muted"], linewidth=1.1))
    ax.text((xa + xb) / 2, y0 + bh + 0.47, "URL-only mode: no page is captured, so layers 2 and 3 are skipped", fontsize=8,
            color=t["ink2"], ha="center", va="bottom")
    xe = x0 + 5 * (bw + gap) - gap
    ys = [y0 + bh / 2 + 0.5, y0 + bh / 2, y0 + bh / 2 - 0.5]
    for (key, color, _), y, lab in zip(VERDICTS, ys, ("likely phishing", "uncertain", "likely legitimate")):
        ax.add_patch(FancyArrowPatch((xe + 0.03, y0 + bh / 2), (xe + 0.26, y), arrowstyle="-", color=t["muted"], linewidth=1.1))
        ax.scatter([xe + 0.36], [y], s=60, color=t[color], zorder=3)
        ax.text(xe + 0.49, y, lab, fontsize=8.6, color=t["ink"], va="center")
    xl, xr = x0 + bw + gap, x0 + 3 * (bw + gap) - gap
    ax.plot([xl, xr], [y0 - 0.09, y0 - 0.09], color=t["axis"], linewidth=1)
    ax.text((xl + xr) / 2, y0 - 0.24, "need the live page", fontsize=8, color=t["muted"], va="center", ha="center")
    f.save(out, "architecture", theme)
    return ["Layer", "Name", "What it does"], [[n, name, " ".join(text.split())] for n, name, text, _ in steps]


# --------------------------------------------------------------------------- driver
def render(root: Optional[Path] = None, out_dir: Optional[Path] = None, only: Optional[List[str]] = None) -> Dict[str, Any]:
    root = root or project_root()
    out = out_dir or (root / "docs" / "figures")
    d = Data(root)
    done: Dict[str, Table] = {}
    skipped: Dict[str, str] = {}
    for name, (_, fn) in FIGURES.items():
        if only and name not in only:
            continue
        try:
            for theme in THEMES:
                table = fn(d, theme, out)
            done[name] = table
            print("drew", name, flush=True)
        except (FileNotFoundError, KeyError) as e:
            plt.close("all")
            skipped[name] = f"missing input: {e}"
            print("skipped", name, "-", skipped[name], flush=True)
    if not only:
        write_index(d, out, done)
    return {"drawn": sorted(done), "skipped": skipped, "out_dir": str(out)}


def write_index(d: Data, out: Path, tables: Dict[str, Table]) -> None:
    lines = ["# Figures", "",
             "Every figure here is drawn by `phishguard figures` from the result files in `metrics/results/` "
             "(and the pre-rebuild audit in `metrics/audit_baseline/`). Nothing is typed by hand. "
             f"The numbers were measured at commit `{d.commit}`. "
             "Each chart comes in a light and a dark version; the table under it holds the plotted values.", ""]
    for name, (caption, _) in FIGURES.items():
        if name not in tables:
            continue
        head, rows = tables[name]
        lines += [f"## {caption}", "",
                  "<picture>",
                  f'  <source media="(prefers-color-scheme: dark)" srcset="{name}-dark.svg">',
                  f'  <img alt="{caption}" src="{name}-light.svg">',
                  "</picture>", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        lines.append("")
        if name == "roc_pr_curves" and d.curves:
            chk = d.curves.get("check_against_evaluation_json", {})
            off = {k: v for k, v in chk.get("differences", {}).items() if v > chk.get("tolerance", 0)}
            if off:
                lines += ["The curves are re-scored from the model files in `" + str(d.curves.get("models_dir")) + "`. "
                          "They match `evaluation.json` exactly except: "
                          + ", ".join(f"{k} differs by {v:.4f}" for k, v in off.items())
                          + ". The chart legend shows the `evaluation.json` values.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw every chart from the result files (light and dark SVG).")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: docs/figures")
    ap.add_argument("--only", nargs="*", default=None, help=f"figure names: {', '.join(FIGURES)}")
    a = ap.parse_args()
    res = render(out_dir=a.out_dir, only=a.only)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
