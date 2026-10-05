#!/usr/bin/env python3
"""bmc_cache.py - cache of the R3 DETERMINISTIC path (decode -> declared-range scale ->
direct resize, image bilinear / mask nearest), matrix v2 section P.

CachedBMCSplitDataset subclasses the locked R3 BMCSplitDataset and overrides ONLY
load_pair(); R3's __getitem__ (training augmentation, uint8 branch, normalisation) runs
unchanged on the cached arrays. A run may use the cache only after verify_cache() has
proven it bit-identical to R3's own load_pair() for every record.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # _builders/
from bmc_data import BMCSplitDataset                                   # locked R3


def _fname(rid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in rid) + ".npz"


def build_cache(ds: BMCSplitDataset, out_dir: Path) -> dict:
    out = Path(out_dir) / ds.dataset / ds.split
    out.mkdir(parents=True, exist_ok=True)
    idx = {}
    for i, r in enumerate(ds.rows):
        img, mask, _ = ds.load_pair(i)
        p = out / _fname(r["canonical_record_id"])
        np.savez(p, image=img, mask=mask)
        idx[r["canonical_record_id"]] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "cache_index.json").write_text(json.dumps(idx, indent=1, sort_keys=True))
    return idx


def verify_cache(ds: BMCSplitDataset, out_dir: Path) -> List[str]:
    """Bit-identical comparison with R3's deterministic path, for every record."""
    out = Path(out_dir) / ds.dataset / ds.split
    bad = []
    for i, r in enumerate(ds.rows):
        img, mask, _ = ds.load_pair(i)
        with np.load(out / _fname(r["canonical_record_id"])) as z:
            ci, cm = z["image"], z["mask"]
        if ci.dtype != img.dtype or cm.dtype != mask.dtype or not np.array_equal(ci, img) \
                or not np.array_equal(cm, mask):
            bad.append(r["canonical_record_id"])
    return bad


class CachedBMCSplitDataset(BMCSplitDataset):
    def __init__(self, *a, cache_dir: Path, **kw):
        super().__init__(*a, **kw)
        self.cache = Path(cache_dir) / self.dataset / self.split

    def load_pair(self, i):
        r = self.rows[i]
        with np.load(self.cache / _fname(r["canonical_record_id"])) as z:
            return z["image"].copy(), z["mask"].copy(), r
