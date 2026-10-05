#!/usr/bin/env python3
"""bmc_figure6_render.py - render Figure 6 from the FROZEN selection (blueprint v2). No new inference.

Place in  _builders\\R6_tools\\ . Order of operations (each step must pass before the next):
  1. frozen selection: figure6_selection.csv must match its selection manifest (SHA-256) and the
     manifest must record the frozen rule and the same evaluation manifest;
  2. input gate: bmc_analyze.load_inputs re-hashes every ledger; the ledgers the selection used must
     still have the recorded hashes;
  3. per displayed case and method: the saved predicted mask PNG written by the frozen R4 writer is
     read as stored (binary {0,255}; no thresholding, interpolation, morphology or post-processing),
     Dice is recomputed with the R4 formula 2|P&G|/(|P|+|G|) against the R3 ground truth and must equal
     the frozen ledger value exactly; input, GT and every PNG are fingerprinted;
  4. rendering at the 256 x 256 evaluation resolution (pixel-replicating display, no smoothing).

    python bmc_figure6_render.py --manifest "<...\\evaluation_manifest__*.json>" --selection-dir "<...\\figures\\figure6_selection__*>"
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bmc_analyze as AN                                        # input gate (sets the R4/R5 import paths)
import bmc_figure6_select as SEL

COLUMNS = [("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("proposed", "Proposed"), ("local", "Local-only"), ("centralised", "Centralised")]
QLABEL = {"25th": "lower (25th pct.)", "50th": "median (50th pct.)", "75th": "upper (75th pct.)"}


class RenderError(RuntimeError):
    pass


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_name(record_id: str) -> str:                           # identical to the frozen R4 writer
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(record_id))


def frozen_selection(sel_dir: Path, eval_root: Path, manifest_sha: str) -> list:
    sel_dir = Path(sel_dir).resolve()
    if sel_dir.parent != (Path(eval_root) / "figures").resolve() or not sel_dir.name.startswith("figure6_selection__"):
        raise RenderError("the selection directory must be _evaluation/figures/figure6_selection__<UTC>")
    sm = json.loads((sel_dir / "figure6_selection_manifest.json").read_text(encoding="utf-8"))
    tbl = sel_dir / "figure6_selection.csv"
    if AN.sha_file(tbl) != sm["output_sha256"]["figure6_selection.csv"]:
        raise RenderError("figure6_selection.csv does not match its frozen selection manifest")
    if sm["rule"]["seed"] != SEL.SEED or sm["evaluation_manifest_sha256"] != manifest_sha:
        raise RenderError("selection manifest does not record the frozen rule / the same evaluation manifest")
    rows = list(csv.DictReader(tbl.open(encoding="utf-8")))
    if [(r["modality"], r["target_quantile"]) for r in rows] != [(m, q) for m in ("ultrasound", "mammography") for q in ("25th", "50th", "75th")]:
        raise RenderError("the frozen selection must contain exactly six cases in the frozen order")
    return rows, sm


_LEDGER_CACHE: dict = {}


def _ledger_row(path: Path, record_id: str) -> dict:
    """The frozen ledger row for one record (the ledger file itself was re-hashed by the input gate)."""
    if path not in _LEDGER_CACHE:
        _LEDGER_CACHE[path] = {r["record_id"]: r for r in csv.DictReader(Path(path).open(encoding="utf-8"))}
    return _LEDGER_CACHE[path][record_id]


def verify_case(inp, row, prep_root: Path, manifest_rows: list) -> dict:
    import numpy as np
    from PIL import Image
    import bmc_data as BD
    ds, rid = row["dataset"], row["record_id"]
    o = BD.BMCSplitDataset(ds, "test", prep_root=prep_root, manifest_rows=manifest_rows, **BD.APPROVED_R3)
    idx = [i for i, r in enumerate(o.rows) if r["canonical_record_id"] == rid]
    if len(idx) != 1:
        raise RenderError(f"{rid}: not found exactly once in the frozen {ds} test split")
    src = o.rows[idx[0]]; item = o[idx[0]]
    img = np.asarray(item["image"][0], dtype=np.float32); gt = np.asarray(item["mask"][0]).astype(np.uint8).astype(bool)
    if img.shape != (256, 256) or gt.shape != (256, 256) or not gt.any():
        raise RenderError(f"{rid}: unexpected R3 input/GT")
    case = {"dataset": ds, "record_id": rid, "image": img, "gt": gt,
            "source_image_sha256": AN.sha_file(Path(src["canonical_image_abspath"])),
            "source_mask_sha256": AN.sha_file(Path(src["canonical_mask_abspath"])),
            "r3_input_array_sha256": sha_bytes(img.tobytes()), "r3_gt_array_sha256": sha_bytes(gt.astype(np.uint8).tobytes()),
            "methods": {}}
    for key, label in COLUMNS:
        L = inp.ledgers[(key, SEL.SEED)]["test"][ds]
        lp = Path(L.path); png = lp.parent / (lp.stem + "__pred_masks") / f"{safe_name(rid)}.png"
        if not png.is_file():
            raise RenderError(f"{label}/{rid}: saved predicted mask missing")
        arr = np.asarray(Image.open(png))
        if arr.shape != (256, 256) or not set(np.unique(arr).tolist()) <= {0, 255}:
            raise RenderError(f"{label}/{rid}: saved mask is not a 256x256 binary {{0,255}} image")
        pred = arr == 255
        inter, g, p = int(np.count_nonzero(pred & gt)), int(np.count_nonzero(gt)), int(np.count_nonzero(pred))
        dice = 2.0 * inter / (g + p)                                # frozen R4 formula (GT non-empty)
        ledger_dice = float(dict(zip(L.record_id.tolist(), L.dice.tolist()))[rid])
        if dice != ledger_dice or abs(float(row[f"dice_{label}"]) - ledger_dice) > 1e-12:
            raise RenderError(f"{label}/{rid}: recomputed Dice {dice!r} != frozen ledger {ledger_dice!r}")
        lrow = _ledger_row(lp, rid)                                # pixel counts recorded by the frozen R4 writer
        if (p, g, inter) != (int(lrow["pred_fg_px"]), int(lrow["gt_fg_px"]), int(lrow["intersection_px"])):
            raise RenderError(f"{label}/{rid}: pixel counts (pred {p}, gt {g}, inter {inter}) differ from the frozen ledger "
                              f"({lrow['pred_fg_px']}, {lrow['gt_fg_px']}, {lrow['intersection_px']})")
        case["methods"][key] = {"label": label, "mask": pred, "dice": ledger_dice, "recomputed_dice": dice,
                                "png": str(png), "png_sha256": AN.sha_file(png), "ledger_sha256": L.sha256,
                                "pred_fg_px": p, "gt_fg_px": g, "intersection_px": inter}
    return case


def render(manifest: Path, sel_dir: Path, prep_root: Path, *, expected_runs=AN.EXPECTED_RUNS) -> Path:
    import numpy as np
    eval_root = Path(prep_root) / "R6_EXPERIMENTS" / "_evaluation"
    rows, sm = frozen_selection(sel_dir, eval_root, AN.sha_file(manifest))
    inp = AN.load_inputs(manifest, eval_root, expected_runs=expected_runs)      # every ledger re-hashed
    for rel, h in sm["ledgers_used_sha256"].items():
        if AN.sha_file(eval_root / rel) != h:
            raise RenderError(f"ledger used by the selection changed: {rel}")
    import bmc_run as RN
    mrows = RN.read_manifest(prep_root)
    cases = [verify_case(inp, r, prep_root, mrows) for r in rows]                # all verified before any rendering
    out = (eval_root / "figures" / f"figure6__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = [{"modality": r["modality"], "target_quantile": r["target_quantile"], "dataset": c["dataset"], "record_id": c["record_id"],
              "method": m["label"], "ledger_dice": m["dice"], "recomputed_dice": m["recomputed_dice"], "exact_match": m["dice"] == m["recomputed_dice"],
              "pred_fg_px": m["pred_fg_px"], "gt_fg_px": m["gt_fg_px"], "intersection_px": m["intersection_px"],
              "pred_png_sha256": m["png_sha256"], "ledger_sha256": m["ledger_sha256"],
              "source_image_sha256": c["source_image_sha256"], "source_mask_sha256": c["source_mask_sha256"],
              "r3_input_array_sha256": c["r3_input_array_sha256"], "r3_gt_array_sha256": c["r3_gt_array_sha256"]}
             for r, c in zip(rows, cases) for m in (c["methods"][k] for k, _ in COLUMNS)]
    ap = out / "figure6_audit.csv"
    with ap.open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0])); w.writeheader(); [w.writerow(a) for a in audit]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    gt_cmap = ListedColormap([(0, 0, 0, 0), (0.0, 0.62, 0.45, 0.55)])           # GT: green, semi-transparent
    pr_cmap = ListedColormap([(0, 0, 0, 0), (0.80, 0.47, 0.65, 0.55)])           # predictions: one neutral colour for all methods
    ncol = 2 + len(COLUMNS)
    fig, axes = plt.subplots(len(cases), ncol, figsize=(7.2, 6.75))
    titles = ["Input", "Ground truth"] + [lab for _, lab in COLUMNS]
    for i, (r, c) in enumerate(zip(rows, cases)):
        for j in range(ncol):
            ax = axes[i, j]
            ax.imshow(c["image"], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            if j == 1:
                ax.imshow(c["gt"].astype(np.uint8), cmap=gt_cmap, vmin=0, vmax=1, interpolation="nearest")
            elif j >= 2:
                m = c["methods"][COLUMNS[j - 2][0]]
                ax.imshow(m["mask"].astype(np.uint8), cmap=pr_cmap, vmin=0, vmax=1, interpolation="nearest")
                ax.text(0.5, -0.04, f"Dice {m['dice']:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=6.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(titles[j], fontsize=7.6, fontweight="bold")
        axes[i, 0].set_ylabel(f"{r['modality'].capitalize()}\n{QLABEL[r['target_quantile']]}\n{c['dataset']}", fontsize=6.8, rotation=0,
                              ha="right", va="center", labelpad=4)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.965, bottom=0.03, wspace=0.04, hspace=0.30)
    outputs = {ap.name: AN.sha_file(ap)}
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {"dpi": 600}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        p = out / f"Figure6_qualitative_segmentation.{ext}"
        if p.exists():
            raise RenderError(f"refusing to overwrite {p}")
        fig.savefig(p, **kw); outputs[p.name] = AN.sha_file(p)
    plt.close(fig)
    from bmc_repro import env_versions
    man = {"figure": "Figure 6", "blueprint": "Final Results Presentation Blueprint v2",
           "selection_dir": str(Path(sel_dir).resolve()), "selection_manifest_sha256": AN.sha_file(Path(sel_dir) / "figure6_selection_manifest.json"),
           "selection_table_sha256": AN.sha_file(Path(sel_dir) / "figure6_selection.csv"), "evaluation_manifest_sha256": AN.sha_file(manifest),
           "record_ids": [r["record_id"] for r in rows], "columns": titles,
           "verification": "every displayed prediction: Dice recomputed (R4 formula) from the saved PNG and the R3 GT equals the frozen ledger value exactly, "
                           "and pred_fg_px, gt_fg_px and intersection_px equal the frozen ledger counts exactly",
           "display": "256 x 256 evaluation space; input on fixed [0,1] (R3 declared_range); masks shown as stored, nearest-neighbour display only",
           "not_performed": "no inference, thresholding, mask interpolation, morphology, post-processing, substitution or image-based selection",
           "selection_note": "the prespecified selection pooled records within modality; all ultrasound cases are from BUS-BRA and all mammography cases "
                             "from CBIS-DDSM; client-level heterogeneity is shown in Figure 4",
           "n_verified_predictions": len(audit), "code_sha256": AN.sha_file(Path(__file__)), "output_sha256": outputs,
           "environment": env_versions(), "created_at": datetime.now(timezone.utc).isoformat()}
    with (out / "figure_manifest.json").open("x", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True, default=str)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--selection-dir", required=True)
    ap.add_argument("--prep-root", default=str(AN.DEFAULT_PREP))
    a = ap.parse_args(argv)
    out = render(Path(a.manifest), Path(a.selection_dir), Path(a.prep_root))
    print("figure written to:", out)
    for r in csv.DictReader((out / "figure6_audit.csv").open(encoding="utf-8")):
        print(f"  {r['modality']:11s} {r['target_quantile']:4s} {r['method']:11s} ledger {float(r['ledger_dice']):.4f} = recomputed "
              f"{float(r['recomputed_dice']):.4f}  exact={r['exact_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
