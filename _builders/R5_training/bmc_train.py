#!/usr/bin/env python3
"""bmc_train.py - one engine for federated (fedavg/fedprox/proposed), centralised and
local-only training (matrix v2, sections C, D, H, I, M).

Data access is through ClientData objects (training draws + deterministic validation),
so the same engine runs on the verified R3 cache or on synthetic test data. Test and
external partitions are never reachable from this module: ClientData refuses them.
No checkpoint is ever loaded to initialise a model.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
import torch

from bmc_aggregation import aggregate
from bmc_losses import joint_loss_per_image, prox_term
from bmc_model import BMCModel, state_sha256
from bmc_repro import generator, seed_everything
from bmc_sampling import centralised_train_batches, client_train_batches, eval_order
from bmc_schedule import lr_at
from bmc_select import client_macro

ALLOWED_TRAIN_SPLITS = ("train",)
ALLOWED_SELECTION_SPLITS = ("validation",)


def task_loss(seg, gt, cls, y, lam_seg, lam_cls):
    """The single task-loss entry point used by every method (T38)."""
    return joint_loss_per_image(seg, gt, cls, y, lam_seg, lam_cls)


@dataclass
class ClientData:
    name: str
    code: int                               # 0 ultrasound, 1 mammography
    train_split: str
    val_split: str
    train_labels: List[int]
    get_train: Callable[[int], dict]        # index -> {"image","mask","label"} (augmented)
    val_items: List[dict]                   # deterministic, sorted

    def __post_init__(self):
        if self.train_split not in ALLOWED_TRAIN_SPLITS:
            raise ValueError(f"{self.name}: split {self.train_split!r} can never be used for training")
        if self.val_split not in ALLOWED_SELECTION_SPLITS:
            raise ValueError(f"{self.name}: split {self.val_split!r} can never be used for selection")

    @property
    def n_train(self) -> int:
        return len(self.train_labels)


def _collate(items: Sequence[dict], codes: Sequence[int], device):
    x = torch.stack([torch.as_tensor(i["image"]) for i in items]).to(device)
    m = torch.stack([torch.as_tensor(i["mask"]) for i in items]).to(device)
    y = torch.tensor([float(i["label"]) for i in items], device=device)
    c = torch.tensor(list(codes), dtype=torch.long, device=device)
    return x, m, y, c


def make_optimizer(model, cfg):
    o = cfg["optimizer"]
    return torch.optim.AdamW(model.parameters(), lr=o["lr0"], betas=tuple(o["betas"]),
                             eps=o["eps"], weight_decay=o["weight_decay"])


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def _autocast(device, cfg):
    use = cfg.get("mixed_precision") == "bf16_autocast" and torch.device(device).type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use else torch.autocast("cpu", enabled=False)


def train_step(model, opt, batch, cfg, device, mu: float = 0.0, ref=None) -> float:
    x, m, y, c = batch
    if x.shape[0] < cfg["batch_size"]:
        raise RuntimeError(f"training batch of {x.shape[0]} < {cfg['batch_size']} (drop_last violated)")
    model.train()
    with _autocast(device, cfg):
        seg, cls = model(x, c)
    loss = task_loss(seg, m, cls, y, cfg["loss"]["lambda_seg"], cfg["loss"]["lambda_cls"]).mean()
    if mu:
        loss = loss + prox_term(model, ref, mu)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def client_val_loss(model, client: ClientData, cfg, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    order = eval_order(len(client.val_items))
    for s in range(0, len(order), 64):
        items = [client.val_items[i] for i in order[s:s + 64]]
        x, m, y, c = _collate(items, [client.code] * len(items), device)
        with _autocast(device, cfg):
            seg, cls = model(x, c)
        l = task_loss(seg, m, cls, y, cfg["loss"]["lambda_seg"], cfg["loss"]["lambda_cls"])
        tot += float(l.sum()); n += len(items)
    return tot / n


def _record(history, idx, per_client, lr, epoch_eq):
    h = {"eval_index": idx, "split": "validation", "criterion": client_macro(per_client),
         "per_client": dict(per_client), "lr": lr, "epoch_equivalent": epoch_eq}
    history.append(h)
    return h


def new_model(cfg, seed: int, rgb_state=None):
    if "init_checkpoint" in cfg or "resume" in cfg:
        raise ValueError("loading a trained checkpoint is forbidden (R0 F10)")
    seed_everything(seed)
    return BMCModel(rgb_encoder_state=rgb_state)


def run_federated(cfg, clients: Sequence[ClientData], seed: int, device="cpu", rgb_state=None,
                  on_round: Optional[Callable] = None) -> dict:
    method = cfg["method"]
    agg_rule = "proposed" if method == "proposed" else "fedavg"
    mu = float(cfg["mu"]) if method in ("fedprox", "proposed") else 0.0
    model = new_model(cfg, seed, rgb_state).to(device)
    meta = {c.name: (c.code, c.n_train) for c in clients}
    gens = {c.name: generator(seed, i) for i, c in enumerate(clients)}
    history, best, best_idx, best_state = [], None, None, None
    for r in range(cfg["rounds"]):
        lr = lr_at(r)
        server = copy.deepcopy(model.state_dict())
        states = {}
        for c in clients:                                           # fixed order
            model.load_state_dict(server)
            ref = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt = make_optimizer(model, cfg)                        # reset every round (D5)
            set_lr(opt, lr)
            for _e in range(cfg["local_epochs"]):
                for b in client_train_batches(c.train_labels, gens[c.name]):
                    items = [c.get_train(i) for i in b]
                    train_step(model, opt, _collate(items, [c.code] * len(items), device), cfg, device, mu, ref)
            if on_round:
                on_round(r, c.name, opt, lr)
            states[c.name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        agg = cfg.get("aggregation", {})
        model.load_state_dict(aggregate(agg_rule, server, states, meta, cap=agg.get("cap", 0.4),
                                        alpha=agg.get("alpha", 0.5), capped=agg.get("capped", True)))
        per = {c.name: client_val_loss(model, c, cfg, device) for c in clients}
        h = _record(history, r, per, lr, cfg["local_epochs"] * (r + 1))
        if best is None or h["criterion"] < best:                   # strict: earliest tie
            best, best_idx = h["criterion"], r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"history": history, "best_index": best_idx, "best_state": best_state,
            "best_sha256": state_sha256(best_state)}


def run_centralised(cfg, clients: Sequence[ClientData], seed: int, device="cpu", rgb_state=None) -> dict:
    model = new_model(cfg, seed, rgb_state).to(device)
    by = {c.name: c for c in clients}
    gen = generator(seed, 999)
    opt = make_optimizer(model, cfg)                                # persists across epochs
    history, best, best_idx, best_state = [], None, None, None
    for blk in range(cfg["epochs"] // cfg["eval_every"]):
        lr = lr_at(blk); set_lr(opt, lr)
        for _e in range(cfg["eval_every"]):
            for b in centralised_train_batches({c.name: c.train_labels for c in clients}, gen):
                items = [by[k].get_train(i) for k, i in b]
                codes = [by[k].code for k, _ in b]
                train_step(model, opt, _collate(items, codes, device), cfg, device)
        per = {c.name: client_val_loss(model, c, cfg, device) for c in clients}
        h = _record(history, blk, per, lr, cfg["eval_every"] * (blk + 1))
        if best is None or h["criterion"] < best:
            best, best_idx = h["criterion"], blk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"history": history, "best_index": best_idx, "best_state": best_state,
            "best_sha256": state_sha256(best_state)}


def run_local(cfg, client: ClientData, seed: int, stream: int, device="cpu", rgb_state=None) -> dict:
    model = new_model(cfg, seed, rgb_state).to(device)
    gen = generator(seed, stream)
    opt = make_optimizer(model, cfg)
    history, best, best_idx, best_state = [], None, None, None
    for blk in range(cfg["epochs"] // cfg["eval_every"]):
        lr = lr_at(blk); set_lr(opt, lr)
        for _e in range(cfg["eval_every"]):
            for b in client_train_batches(client.train_labels, gen):
                items = [client.get_train(i) for i in b]
                train_step(model, opt, _collate(items, [client.code] * len(items), device), cfg, device)
        h = _record(history, blk, {client.name: client_val_loss(model, client, cfg, device)}, lr,
                    cfg["eval_every"] * (blk + 1))
        if best is None or h["criterion"] < best:
            best, best_idx = h["criterion"], blk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"history": history, "best_index": best_idx, "best_state": best_state,
            "best_sha256": state_sha256(best_state)}


def require_frozen_selection(run_dir: Path) -> Path:
    """Ordering safeguard (C8): test/external evaluation needs the frozen record first."""
    p = Path(run_dir) / "selection_record.json"
    if not p.is_file():
        raise RuntimeError("no frozen selection record: internal-test/external evaluation refused")
    return p
