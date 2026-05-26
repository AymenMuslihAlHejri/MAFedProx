"""
Bootstrap CI + Cohen's d + Sign Test
for MAFedProx Statistical Analysis
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

np.random.seed(42)
N_BOOTSTRAP = 10000

# ── Data from experiments ─────────────────────────────────
CLIENTS = ['BUSI','BUS_UC','BrEaST','BUS_BRA',
           'INbreast','DMID','CBIS_DDSM','AIISLAB']

DSC = {
    "MAFedProx": np.array([0.8064,0.9220,0.7624,0.9062,
                            0.8338,0.5215,0.2613,0.3884]),
    "Central":   np.array([0.7916,0.9111,0.8369,0.8949,
                            0.5516,0.6495,0.1951,0.6365]),
    "Config_B":  np.array([0.8237,0.9165,0.7514,0.8819,
                            0.8003,0.4162,0.2517,0.3131]),
    "Config_E":  np.array([0.7978,0.9177,0.7776,0.8848,
                            0.7874,0.5160,0.2560,0.3028]),
}

MODALITY = {
    'BUSI':'US','BUS_UC':'US','BrEaST':'US','BUS_BRA':'US',
    'INbreast':'MM','DMID':'MM','CBIS_DDSM':'MM','AIISLAB':'MM'
}

# ── Helpers ────────────────────────────────────────────────
def bootstrap_ci(data, n=N_BOOTSTRAP, ci=95):
    means = [np.mean(np.random.choice(data, len(data), replace=True))
             for _ in range(n)]
    lo = np.percentile(means, (100-ci)/2)
    hi = np.percentile(means, 100-(100-ci)/2)
    return float(lo), float(float(np.mean(data))), float(hi)

def bootstrap_diff_ci(a, b, n=N_BOOTSTRAP, ci=95):
    """Bootstrap CI on mean difference (a - b)."""
    diffs = []
    for _ in range(n):
        idx = np.random.choice(len(a), len(a), replace=True)
        diffs.append(np.mean(a[idx] - b[idx]))
    lo = np.percentile(diffs, (100-ci)/2)
    hi = np.percentile(diffs, 100-(100-ci)/2)
    return float(lo), float(np.mean(a-b)), float(hi)

def cohens_d(a, b):
    """Cohen's d for paired data."""
    diff = a - b
    return float(np.mean(diff) / np.std(diff, ddof=1))

def effect_label(d):
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"

def sign_test(a, b):
    """Exact sign test."""
    pos = np.sum(a > b)
    neg = np.sum(a < b)
    n   = pos + neg
    # Two-tailed binomial p
    from scipy.stats import binom_test
    try:
        p = binom_test(min(pos,neg), n, 0.5)
    except:
        from scipy.stats import binomtest
        p = binomtest(min(pos,neg), n, 0.5).pvalue
    return int(pos), int(neg), float(p)

# ── Main Analysis ──────────────────────────────────────────
print("\n" + "="*65)
print("  STATISTICAL ANALYSIS — MAFedProx")
print("="*65)

results = {}

# 1. Bootstrap CI per method
print("\n── Per-Method Bootstrap 95% CI (n=8 clients) ──────────────")
print(f"{'Method':20s}  {'Mean DSC':>10}  {'95% CI':>22}  {'US Mean':>8}  {'MM Mean':>8}")
print("-"*75)

us_idx = np.array([i for i,c in enumerate(CLIENTS) if MODALITY[c]=='US'])
mm_idx = np.array([i for i,c in enumerate(CLIENTS) if MODALITY[c]=='MM'])

ci_results = {}
for name, dsc in DSC.items():
    lo, mn, hi = bootstrap_ci(dsc)
    us_mn = float(np.mean(dsc[us_idx]))
    mm_mn = float(np.mean(dsc[mm_idx]))
    ci_results[name] = {"mean":mn,"ci_lo":lo,"ci_hi":hi,
                         "us_mean":us_mn,"mm_mean":mm_mn}
    print(f"{name:20s}  {mn:10.4f}  [{lo:.4f} — {hi:.4f}]  "
          f"{us_mn:8.4f}  {mm_mn:8.4f}")

results["bootstrap_ci"] = ci_results

# 2. Pairwise comparisons
print("\n── Pairwise Comparison vs MAFedProx ───────────────────────")
print(f"{'Comparison':30s}  {'Δ Mean':>8}  {'95% CI on Δ':>24}  "
      f"{'Cohen d':>8}  {'Effect':>10}  {'Sign(+/-)':>10}  {'p_sign':>8}")
print("-"*105)

comparisons = [
    ("vs Centralised",   "Central"),
    ("vs Config B",      "Config_B"),
    ("vs Config E",      "Config_E"),
]

pair_results = {}
for label, key in comparisons:
    a = DSC["MAFedProx"]
    b = DSC[key]

    lo, mn, hi = bootstrap_diff_ci(a, b)
    d          = cohens_d(a, b)
    pos, neg, p_sign = sign_test(a, b)

    # CI interpretation
    if lo > 0:
        sig_ci = "✅ CI>0 (favours MAFedProx)"
    elif hi < 0:
        sig_ci = "❌ CI<0 (favours baseline)"
    else:
        sig_ci = "⚠️  CI crosses 0"

    pair_results[label] = {
        "mean_diff": mn, "ci_lo": lo, "ci_hi": hi,
        "cohens_d": d, "effect_size": effect_label(d),
        "sign_pos": pos, "sign_neg": neg, "p_sign": p_sign,
        "ci_interpretation": sig_ci
    }

    print(f"{label:30s}  {mn:+8.4f}  [{lo:+.4f} — {hi:+.4f}]  "
          f"{d:8.4f}  {effect_label(d):>10}  "
          f"{pos:3d}/{neg:3d}      {p_sign:.4f}")
    print(f"  → {sig_ci}")

results["pairwise"] = pair_results

# 3. Per-client breakdown
print("\n── Per-Client: MAFedProx vs Centralised ───────────────────")
print(f"{'Client':15s}  {'MAFedProx':>10}  {'Central':>10}  "
      f"{'Diff':>8}  {'Winner':>12}")
print("-"*65)
per_client = {}
for i, c in enumerate(CLIENTS):
    diff = DSC["MAFedProx"][i] - DSC["Central"][i]
    winner = "MAFedProx ✅" if diff > 0 else "Central ❌"
    per_client[c] = {
        "mafedprox": float(DSC["MAFedProx"][i]),
        "central":   float(DSC["Central"][i]),
        "diff":      float(diff),
        "winner":    winner
    }
    print(f"{c:15s}  {DSC['MAFedProx'][i]:10.4f}  "
          f"{DSC['Central'][i]:10.4f}  {diff:+8.4f}  {winner}")

results["per_client_vs_central"] = per_client

# 4. Summary for paper
print("\n" + "="*65)
print("  SUMMARY FOR PAPER")
print("="*65)

mn_vs_c = pair_results["vs Centralised"]["mean_diff"]
lo_vs_c = pair_results["vs Centralised"]["ci_lo"]
hi_vs_c = pair_results["vs Centralised"]["ci_hi"]
d_vs_c  = pair_results["vs Centralised"]["cohens_d"]
ci_str  = pair_results["vs Centralised"]["ci_interpretation"]

print(f"""
MAFedProx vs Centralised:
  Mean DSC difference: {mn_vs_c:+.4f} (95% CI: {lo_vs_c:+.4f} to {hi_vs_c:+.4f})
  Cohen's d = {d_vs_c:.3f} ({effect_label(d_vs_c)} effect)
  {ci_str}
  MAFedProx wins on {pair_results['vs Centralised']['sign_pos']}/8 clients

MAFedProx vs Config B:
  Mean DSC difference: {pair_results['vs Config B']['mean_diff']:+.4f}
  95% CI: [{pair_results['vs Config B']['ci_lo']:+.4f}, {pair_results['vs Config B']['ci_hi']:+.4f}]
  Cohen's d = {pair_results['vs Config B']['cohens_d']:.3f} ({pair_results['vs Config B']['effect_size']})
  {pair_results['vs Config B']['ci_interpretation']}

MAFedProx vs Config E:
  Mean DSC difference: {pair_results['vs Config E']['mean_diff']:+.4f}
  95% CI: [{pair_results['vs Config E']['ci_lo']:+.4f}, {pair_results['vs Config E']['ci_hi']:+.4f}]
  Cohen's d = {pair_results['vs Config E']['cohens_d']:.3f} ({pair_results['vs Config E']['effect_size']})
  {pair_results['vs Config E']['ci_interpretation']}
""")

# Save
out = Path(r"C:\Users\Asus\Python code\FL_MuL_BC\results\statistical_analysis.json")
out.parent.mkdir(parents=True, exist_ok=True)

# Convert numpy types
def convert(o):
    if isinstance(o, (np.integer, np.floating)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, bool): return bool(o)
    raise TypeError(type(o))

with open(out, "w") as f:
    json.dump(results, f, indent=2, default=convert)
print(f"Saved → {out}")
