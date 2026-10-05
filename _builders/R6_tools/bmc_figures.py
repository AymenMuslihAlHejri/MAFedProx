#!/usr/bin/env python3
"""bmc_figures.py - provenance-safe figure generation for the frozen R6 results (blueprint v2).

Place in  _builders\\R6_tools\\ . Reads ONLY a verified analysis directory written by bmc_analyze.py;
writes ONLY under R6_EXPERIMENTS\\_evaluation\\figures\\<UTC>\\ (write-once), with a figure manifest
recording the analysis-manifest hash, every input-table hash, the plotted values, the code hash
and every output hash. It computes no new statistic: plotted values are read from the frozen
tables, and the mean/SD are re-checked against the frozen bmc_stats.seed_summary.

Implemented so far: Figure 2 (main model performance comparison), Figure 3 (confirmatory forest plot), Figure 4 (cross-client heterogeneity), Figure 5 (external generalization). Further figures are added
one at a time after audit.

    python bmc_figures.py --figure 2 --analysis-dir "<...\\_evaluation\\analysis\\<UTC>>"
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BUILDERS = HERE.parent
for p in (BUILDERS / "R5_training", BUILDERS, BUILDERS / "R4_evaluation", HERE):
    if p.is_dir():
        sys.path.insert(0, str(p))

SEEDS = (101, 202, 303)
MAIN_ORDER = [("proposed", "Proposed"), ("fedavg", "FedAvg"), ("fedprox", "FedProx"),
              ("local", "Local-only"), ("centralised", "Centralised")]


class FigureError(RuntimeError):
    pass


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- input gate
def verified_analysis(analysis_dir: Path) -> dict:
    """Every table listed in analysis_manifest.json must exist and match its SHA-256."""
    analysis_dir = Path(analysis_dir).resolve()
    if analysis_dir.parent.name != "analysis" or analysis_dir.parent.parent.name != "_evaluation":
        raise FigureError("the analysis directory must be _evaluation/analysis/<UTC>")
    am_p = analysis_dir / "analysis_manifest.json"
    if not am_p.is_file():
        raise FigureError("analysis_manifest.json missing")
    am = json.loads(am_p.read_text(encoding="utf-8"))
    for name, digest in am["output_sha256"].items():
        p = analysis_dir / name
        if not p.is_file() or sha_file(p) != digest:
            raise FigureError(f"analysis output missing or altered: {name}")
    return {"dir": analysis_dir, "manifest": am, "manifest_sha256": sha_file(am_p),
            "eval_root": analysis_dir.parent.parent}


def read_table(gate: dict, name: str) -> list:
    if name not in gate["manifest"]["output_sha256"]:
        raise FigureError(f"{name} is not a listed analysis output")
    return list(csv.DictReader((gate["dir"] / name).open(encoding="utf-8")))


# ---------------------------------------------------------------- output (write-once)
class FigureRun:
    def __init__(self, gate: dict, figure_id: str):
        self.gate = gate
        self.root = (gate["eval_root"] / "figures").resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.dir = self.root / f"{figure_id}__{stamp}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.outputs, self.inputs = {}, {}

    def path(self, name: str) -> Path:
        p = (self.dir / name).resolve()
        if p.parent != self.dir or p.exists():
            raise FigureError(f"refusing to write {p} (outside the figure directory, or already exists)")
        return p

    def record(self, p: Path):
        self.outputs[p.name] = sha_file(p)

    def write_csv(self, name: str, rows: list):
        p = self.path(name)
        with p.open("x", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
        self.record(p)

    def finish(self, figure_id: str, spec: dict):
        from bmc_repro import env_versions
        man = {"figure": figure_id, "blueprint": "Final Results Presentation Blueprint v2", "spec": spec,
               "analysis_dir": str(self.gate["dir"]), "analysis_manifest_sha256": self.gate["manifest_sha256"],
               "input_tables_sha256": self.inputs, "code_sha256": sha_file(Path(__file__)),
               "output_sha256": self.outputs, "environment": env_versions(),
               "created_at": datetime.now(timezone.utc).isoformat()}
        with self.path("figure_manifest.json").open("x", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, sort_keys=True, default=str)
        return self.dir


# ---------------------------------------------------------------- Figure 2
def figure2_values(gate: dict) -> list:
    import bmc_stats as ST
    rows = read_table(gate, "main_descriptive.csv")
    out = []
    for metric in ("roc_auc", "dice"):
        for key, label in MAIN_ORDER:
            r = [x for x in rows if x["configuration"] == key and x["metric"] == metric and x["threshold"] == "primary"]
            if len(r) != 1:
                raise FigureError(f"main_descriptive.csv: expected one row for {key}/{metric}, found {len(r)}")
            r = r[0]; seeds = {s: float(r[f"seed_{s}"]) for s in SEEDS}
            chk = ST.seed_summary(seeds)                       # frozen R4 summary: consistency check only
            if abs(chk["mean"] - float(r["mean"])) > 1e-12 or abs(chk["sd"] - float(r["sd"])) > 1e-12 or int(r["ddof"]) != 1:
                raise FigureError(f"mean/SD in main_descriptive.csv do not match the frozen seed_summary for {key}/{metric}")
            out.append({"metric": metric, "configuration": key, "label": label, **{f"seed_{s}": seeds[s] for s in SEEDS},
                        "mean": float(r["mean"]), "sd": float(r["sd"]), "ddof": 1})
    return out


def figure2(analysis_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    gate = verified_analysis(analysis_dir)
    vals = figure2_values(gate)
    run = FigureRun(gate, "figure2")
    run.inputs["main_descriptive.csv"] = gate["manifest"]["output_sha256"]["main_descriptive.csv"]
    run.write_csv("figure2_plotted_values.csv", vals)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    seed_style = {101: "o", 202: "s", 303: "^"}
    offsets = {101: -0.17, 202: 0.0, 303: 0.17}
    panels = [("roc_auc", "A", "ROC-AUC"), ("dice", "B", "Dice")]
    ylabels = [lab for _, lab in MAIN_ORDER]
    for ax, (metric, letter, title) in zip(axes, panels):
        sub = [v for v in vals if v["metric"] == metric]
        for i, v in enumerate(sub):
            y = len(sub) - 1 - i
            ax.errorbar(v["mean"], y, xerr=v["sd"], fmt="D", ms=5.5, color="#1f2a44", ecolor="#1f2a44",
                        elinewidth=1.3, capsize=3.5, zorder=3)
            for s in SEEDS:
                ax.plot(v[f"seed_{s}"], y + offsets[s], seed_style[s], ms=3.6, mfc="white", mec="#6b7280",
                        mew=0.9, zorder=4)
        allv = [v[f"seed_{s}"] for v in sub for s in SEEDS] + [v["mean"] - v["sd"] for v in sub] + [v["mean"] + v["sd"] for v in sub]
        pad = 0.08 * (max(allv) - min(allv))
        ax.set_xlim(min(allv) - pad, max(allv) + pad)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(ylabels[::-1])
        ax.set_xlabel(f"{title} (internal test, client-macro)")
        ax.set_title(f"{letter}", loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="x", color="#e5e7eb", lw=0.6, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], marker="D", color="#1f2a44", lw=1.3, ms=5.5,
                      label="Mean across seeds ± across-seed SD (n = 3; descriptive, not a CI)")] + \
              [Line2D([0], [0], marker=seed_style[s], ls="none", mfc="white", mec="#6b7280", ms=4, label=f"Seed {s}")
               for s in SEEDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        p = run.path(f"Figure2_main_model_comparison.{ext}")
        fig.savefig(p, bbox_inches="tight", **kw); run.record(p)
    plt.close(fig)
    spec = {"panels": ["A: ROC-AUC", "B: Dice"], "methods_order": [lab for _, lab in MAIN_ORDER],
            "aggregation": "client-macro per seed (frozen bmc_stats.summarise) -> mean and SD (ddof = 1) across 3 seeds",
            "points": "three individual seed values (open markers) plus mean (filled diamond)",
            "error_bars": "across-seed SD, n = 3, descriptive; NOT confidence intervals",
            "threshold": "tau = 0.5 not involved (ROC-AUC and Dice are threshold-free / fixed segmentation threshold)",
            "statistical_tests_shown": "none (confirmatory tests are Figure 3)",
            "local_only_note": "client-macro over eight independently trained client models"}
    return run.finish("Figure 2", spec)


# ---------------------------------------------------------------- Figure 3
CONFIRMATORY_SET = [("Proposed vs fedavg", "Proposed \u2212 FedAvg"), ("Proposed vs fedprox", "Proposed \u2212 FedProx"),
                    ("Proposed vs local", "Proposed \u2212 Local-only")]


def figure3_values(gate: dict) -> list:
    rows = read_table(gate, "confirmatory_tests.csv")
    want = {(c, m) for c, _ in CONFIRMATORY_SET for m in ("roc_auc", "dice")}
    have = [(r["comparison"], r["metric"]) for r in rows]
    if len(rows) != 6 or set(have) != want:
        raise FigureError(f"confirmatory_tests.csv must contain exactly the six prespecified tests; found {have}")
    if gate["manifest"].get("holm_family_size") != 6:
        raise FigureError("analysis manifest does not record a six-test Holm family")
    out = []
    for metric in ("roc_auc", "dice"):
        for comp, label in CONFIRMATORY_SET:
            r = [x for x in rows if x["comparison"] == comp and x["metric"] == metric][0]
            if int(r["B_attempted"]) != 2000 or int(r["bootstrap_seed"]) != 20260930:
                raise FigureError(f"{comp}/{metric}: bootstrap contract not met")
            out.append({"metric": metric, "comparison": comp, "label": label, "delta_hat": float(r["delta_hat"]),
                        "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]), "p_holm": float(r["p_holm"]),
                        "p_raw": float(r["p_raw"]), "reject": r["reject"], "B_valid": int(r["B_valid"]),
                        "B_attempted": int(r["B_attempted"])})
    return out


def _fmt_p(p: float) -> str:
    return f"Holm-adjusted p = {p:.4f}"


def figure3(analysis_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gate = verified_analysis(analysis_dir)
    vals = figure3_values(gate)
    run = FigureRun(gate, "figure3")
    run.inputs["confirmatory_tests.csv"] = gate["manifest"]["output_sha256"]["confirmatory_tests.csv"]
    run.write_csv("figure3_plotted_values.csv", vals)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.3))
    for ax, (metric, letter, title) in zip(axes, [("roc_auc", "A  Classification (ROC-AUC)", "ROC-AUC"), ("dice", "B  Segmentation (Dice)", "Dice")]):
        sub = [v for v in vals if v["metric"] == metric]
        lo = min(min(v["ci_low"] for v in sub), 0.0); hi = max(max(v["ci_high"] for v in sub), 0.0)
        pad = 0.08 * (hi - lo)                                   # same fixed-margin rule as Figure 2
        ax.set_xlim(lo - pad, hi + pad)
        ax.axvline(0.0, color="#374151", lw=1.1, ls="--", zorder=1)
        for i, v in enumerate(sub):
            y = len(sub) - 1 - i
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color="#1f2a44", lw=1.4, solid_capstyle="butt", zorder=2)
            for x in (v["ci_low"], v["ci_high"]):
                ax.plot([x, x], [y - 0.09, y + 0.09], color="#1f2a44", lw=1.4, zorder=2)
            ax.plot(v["delta_hat"], y, "o", ms=5.5, color="#1f2a44", zorder=3)
            ax.annotate(_fmt_p(v["p_holm"]), xy=(1.02, y), xycoords=("axes fraction", "data"), va="center", fontsize=7.8)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels([v["label"] for v in sub][::-1])
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_xlabel(f"\u0394 {title} (Proposed \u2212 comparator)")
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="x", color="#e5e7eb", lw=0.6, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.74, top=0.94, bottom=0.11, hspace=0.62)
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        p = run.path(f"Figure3_confirmatory_forest.{ext}")
        fig.savefig(p, bbox_inches="tight", **kw); run.record(p)
    plt.close(fig)
    spec = {"tests": "exactly the six prespecified confirmatory tests (Proposed vs FedAvg/FedProx/Local-only x ROC-AUC/Dice)",
            "point": "delta_hat = mean over seeds of (Proposed - comparator), from confirmatory_tests.csv",
            "interval": "marginal paired cluster-bootstrap percentile 95% CI; NOT multiplicity-adjusted",
            "multiplicity": "Holm-adjusted p-values across the six-test family (printed per row)",
            "reference_line": "vertical dashed line at delta = 0",
            "B_valid": {f"{v['comparison']}/{v['metric']}": v["B_valid"] for v in vals},
            "excluded": "Centralised, N1-N4, external, tau_val, secondary metrics, any additional test",
            "axis_rule": "independent per panel; data-driven from CI limits and 0, fixed 8% margin"}
    return run.finish("Figure 3", spec)


# ---------------------------------------------------------------- Figure 4
CLIENT_ORDER = ["BUSI", "BUS-UC", "BrEaST", "BUS-BRA", "INbreast", "DMID", "CBIS-DDSM", "AIISLAB"]   # client IDs 0-7
COLOUR_LIMITS = {"roc_auc": (0.5, 1.0), "dice": (0.0, 1.0)}   # natural metric bounds, fixed before plotting


def figure4_values(gate: dict) -> list:
    import bmc_stats as ST
    rows = read_table(gate, "per_client_supplementary.csv")
    out = []
    for metric in ("roc_auc", "dice"):
        for key, label in MAIN_ORDER:
            for ds in CLIENT_ORDER:
                r = {int(x["seed"]): x for x in rows if x["configuration"] == key and x["dataset"] == ds}
                if sorted(r) != list(SEEDS):
                    raise FigureError(f"per_client_supplementary.csv: {key}/{ds} lacks one row per seed")
                ns = {r[s]["n"] for s in SEEDS}
                if len(ns) != 1:
                    raise FigureError(f"{key}/{ds}: test n differs across seeds")
                seeds = {s: float(r[s][metric]) for s in SEEDS}
                summ = ST.seed_summary(seeds)                         # frozen R4 mean / SD (ddof = 1)
                out.append({"metric": metric, "configuration": key, "label": label, "dataset": ds, "n": int(ns.pop()),
                            **{f"seed_{s}": seeds[s] for s in SEEDS}, "mean": summ["mean"], "sd": summ["sd"], "ddof": summ["ddof"]})
    ref = {(v["dataset"], v["n"]) for v in out}
    if len(ref) != len(CLIENT_ORDER):
        raise FigureError("test n differs across methods for the same client")
    return out


def figure4(analysis_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    gate = verified_analysis(analysis_dir)
    vals = figure4_values(gate)
    run = FigureRun(gate, "figure4")
    run.inputs["per_client_supplementary.csv"] = gate["manifest"]["output_sha256"]["per_client_supplementary.csv"]
    run.write_csv("figure4_plotted_values.csv", vals)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    n_of = {v["dataset"]: v["n"] for v in vals}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    for ax, (metric, title) in zip(axes, [("roc_auc", "A  Classification (ROC-AUC)"), ("dice", "B  Segmentation (Dice)")]):
        M = np.array([[next(v["mean"] for v in vals if v["metric"] == metric and v["configuration"] == k and v["dataset"] == ds)
                       for ds in CLIENT_ORDER] for k, _ in MAIN_ORDER])
        lo, hi = COLOUR_LIMITS[metric]
        im = ax.imshow(M, cmap="viridis", vmin=lo, vmax=hi, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                frac = (M[i, j] - lo) / (hi - lo)
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.4,
                        color="black" if frac > 0.62 else "white")
        ax.set_xticks(range(len(CLIENT_ORDER))); ax.set_xticklabels([f"{ds}\n(n = {n_of[ds]})" for ds in CLIENT_ORDER], fontsize=7.4)
        ax.set_yticks(range(len(MAIN_ORDER))); ax.set_yticklabels([lab for _, lab in MAIN_ORDER])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5, pad=17)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(3.5, color="white", lw=2.4, zorder=3)                 # subtle modality separator
        ax.axvline(3.5, color="#4b5563", lw=0.7, zorder=4)
        for x, txt in ((1.5, "Ultrasound"), (5.5, "Mammography")):
            ax.text(x, -0.62, txt, ha="center", va="bottom", fontsize=8.0, color="#374151")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.08, hspace=0.55)
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        p = run.path(f"Figure4_client_heterogeneity.{ext}")
        fig.savefig(p, bbox_inches="tight", **kw); run.record(p)
    plt.close(fig)
    spec = {"panels": ["A: ROC-AUC", "B: Dice"], "rows": [lab for _, lab in MAIN_ORDER], "columns": CLIENT_ORDER,
            "cell_value": "mean across the three seeds (frozen bmc_stats.seed_summary) of the frozen client-level value",
            "colour_scale": {k: list(v) for k, v in COLOUR_LIMITS.items()},
            "colour_scale_rule": "absolute natural metric bounds fixed before plotting; same scale for all methods; no row normalisation",
            "column_labels": "client test n", "modality_groups": {"Ultrasound": CLIENT_ORDER[:4], "Mammography": CLIENT_ORDER[4:]},
            "separator": "subtle vertical line between BUS-BRA and INbreast", "local_only_note": "each client's own independently trained model",
            "statistical_tests_shown": "none"}
    return run.finish("Figure 4", spec)


# ---------------------------------------------------------------- Figure 5
EXTERNAL_ORDER = [("BLUIS", "BLUIS\n(ultrasound)"), ("BUS-UCLM", "BUS-UCLM\n(ultrasound)"),
                  ("CDD-CESM", "CDD-CESM\n(mammography,\nlow-energy CESM)")]
EXTERNAL_METHODS = [("proposed", "Proposed"), ("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("centralised", "Centralised")]
METHOD_STYLE = {"proposed": ("#0072B2", "o"), "fedavg": ("#E69F00", "s"), "fedprox": ("#009E73", "^"), "centralised": ("#CC79A7", "D")}


def figure5_values(gate: dict) -> list:
    import bmc_stats as ST
    rows = read_table(gate, "external_descriptive.csv")
    if any(r["configuration"] == "local" for r in rows):
        raise FigureError("external_descriptive.csv contains Local-only rows; protocol violation")
    out = []
    for metric in ("roc_auc", "dice"):
        for ds, _ in EXTERNAL_ORDER:
            for key, label in EXTERNAL_METHODS:
                r = [x for x in rows if x["configuration"] == key and x["dataset"] == ds and x["metric"] == metric and x["threshold"] == "primary"]
                if len(r) != 1:
                    raise FigureError(f"external_descriptive.csv: expected one row for {key}/{ds}/{metric}, found {len(r)}")
                r = r[0]; seeds = {s: float(r[f"seed_{s}"]) for s in SEEDS}
                chk = ST.seed_summary(seeds)
                if abs(chk["mean"] - float(r["mean"])) > 1e-12 or abs(chk["sd"] - float(r["sd"])) > 1e-12:
                    raise FigureError(f"mean/SD do not match the frozen seed_summary for {key}/{ds}/{metric}")
                out.append({"metric": metric, "dataset": ds, "configuration": key, "label": label,
                            **{f"seed_{s}": seeds[s] for s in SEEDS}, "mean": float(r["mean"]), "sd": float(r["sd"]), "ddof": 1})
    return out


def figure5(analysis_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    gate = verified_analysis(analysis_dir)
    vals = figure5_values(gate)
    run = FigureRun(gate, "figure5")
    run.inputs["external_descriptive.csv"] = gate["manifest"]["output_sha256"]["external_descriptive.csv"]
    run.write_csv("figure5_plotted_values.csv", vals)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    offs = {k: o for (k, _), o in zip(EXTERNAL_METHODS, (-0.27, -0.09, 0.09, 0.27))}
    seed_dx = {101: -0.035, 202: 0.0, 303: 0.035}
    for ax, (metric, title, ylab) in zip(axes, [("roc_auc", "A  Classification (ROC-AUC)", "ROC-AUC"),
                                                ("dice", "B  Segmentation (Dice)", "Dice")]):
        sub = [v for v in vals if v["metric"] == metric]
        for j, (ds, _) in enumerate(EXTERNAL_ORDER):
            for key, _ in EXTERNAL_METHODS:
                v = next(x for x in sub if x["dataset"] == ds and x["configuration"] == key)
                col, mk = METHOD_STYLE[key]; x0 = j + offs[key]
                ax.errorbar(x0, v["mean"], yerr=v["sd"], fmt=mk, ms=5, color=col, ecolor=col, elinewidth=1.2, capsize=2.8, zorder=3)
                for s in SEEDS:
                    ax.plot(x0 + seed_dx[s], v[f"seed_{s}"], mk, ms=2.6, mfc="white", mec=col, mew=0.7, zorder=4)
        allv = [v[f"seed_{s}"] for v in sub for s in SEEDS] + [v["mean"] - v["sd"] for v in sub] + [v["mean"] + v["sd"] for v in sub]
        pad = 0.08 * (max(allv) - min(allv))
        ax.set_ylim(min(allv) - pad, max(allv) + pad)
        ax.set_xticks(range(len(EXTERNAL_ORDER))); ax.set_xticklabels([lab for _, lab in EXTERNAL_ORDER], fontsize=7.6)
        ax.set_xlim(-0.55, len(EXTERNAL_ORDER) - 0.45)
        for x in (0.5, 1.5):
            ax.axvline(x, color="#e5e7eb", lw=0.8, zorder=0)
        ax.set_ylabel(f"{ylab} (external test)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
        ax.grid(axis="y", color="#f0f0f0", lw=0.6, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], marker=METHOD_STYLE[k][1], color=METHOD_STYLE[k][0], lw=1.2, ms=5, label=lab) for k, lab in EXTERNAL_METHODS]
    handles.append(Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#6b7280", ms=3.5, label="Individual seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, -0.085, "Filled markers: mean across seeds; bars: across-seed SD (n = 3; descriptive, not a CI)",
             ha="center", fontsize=7.4, color="#374151")
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.5)
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        p = run.path(f"Figure5_external_generalization.{ext}")
        fig.savefig(p, bbox_inches="tight", **kw); run.record(p)
    plt.close(fig)
    spec = {"panels": ["A: ROC-AUC", "B: Dice"], "datasets": [d for d, _ in EXTERNAL_ORDER], "methods": [lab for _, lab in EXTERNAL_METHODS],
            "excluded": "Local-only (protocol), N1-N4 (blueprint v2)", "pooling": "none; each external dataset separate",
            "aggregation": "dataset-level value per seed -> mean and SD (ddof = 1) across 3 seeds (frozen tables)",
            "points": "individual seed values (open markers) plus mean (filled) with across-seed SD bars",
            "axis_rule": "independent per panel; data-driven with fixed 8% margin",
            "method_colours": "Okabe-Ito colour-blind-safe palette; equal visual weight for all methods",
            "n_not_shown": "external test sizes are not in external_descriptive.csv; reported in Tables 2 and 11",
            "statistical_tests_shown": "none"}
    return run.finish("Figure 5", spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", required=True, choices=["2", "3", "4", "5"]); ap.add_argument("--analysis-dir", required=True)
    a = ap.parse_args(argv)
    out = {"2": figure2, "3": figure3, "4": figure4, "5": figure5}[a.figure](Path(a.analysis_dir))
    print("figure written to:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
