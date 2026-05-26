"""
====================================================================
server.py  —  Final v3  (Fixed _NE _ND as instance variables)
Modality-Aware FedProx Aggregation Strategy

Fix: _NE and _ND moved from class variables to instance variables
     to prevent state sharing between runs in Ray simulation
====================================================================
"""

import numpy as np
from logging import INFO

import flwr as fl
from flwr.common import (FitRes, Parameters,
                         ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.common.logger import log

import config as C
from logger import FLLogger


# ════════════════════════════════════════════════════════════════
# Capped weighted average
# ════════════════════════════════════════════════════════════════
def _capped_avg(results, cap=C.MAX_CLIENT_WEIGHT):
    total   = sum(n for n, _ in results)
    weights = [min(n/total, cap) for n, _ in results]
    ws      = sum(weights)
    weights = [w/ws for w in weights]
    avg = None
    for w, (_, arrays) in zip(weights, results):
        if avg is None:
            avg = [a * w for a in arrays]
        else:
            avg = [acc + a*w for acc, a in zip(avg, arrays)]
    return avg


# ════════════════════════════════════════════════════════════════
# Strategy
# ════════════════════════════════════════════════════════════════
class ModalityAwareFedProx(fl.server.strategy.Strategy):

    def __init__(self,
                 num_rounds         = C.FL_ROUNDS,
                 fusion_alpha       = C.FUSION_ALPHA,
                 min_clients        = C.NUM_CLIENTS,
                 initial_parameters = None,
                 local_epochs       = C.LOCAL_EPOCHS,
                 max_client_weight  = C.MAX_CLIENT_WEIGHT,
                 out_dir            = C.OUT_FL):

        self.num_rounds         = num_rounds
        self.fusion_alpha       = fusion_alpha
        self.min_clients        = min_clients
        self.initial_parameters = initial_parameters
        self.local_epochs       = local_epochs
        self.max_client_weight  = max_client_weight
        self.fl_logger          = FLLogger(out_dir=out_dir)
        self.last_global        = None
        self.drift_log          = {}
        self.round_metrics      = []
        self._best_dice         = 0.0
        self._best_round        = 0

        # ── FIX: instance variables (not class variables) ────────
        # Prevents state sharing between runs in Ray simulation
        self._NE = None   # encoder boundary
        self._ND = None   # single decoder size

    # ── Flower required ──────────────────────────────────────────
    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        cfg     = {"local_epochs": self.local_epochs,
                   "server_round": server_round}
        clients = client_manager.sample(
            num_clients     = self.min_clients,
            min_num_clients = self.min_clients)
        return [(c, fl.common.FitIns(parameters, cfg))
                for c in clients]

    def configure_evaluate(self, server_round, parameters,
                           client_manager):
        clients = client_manager.sample(
            num_clients     = self.min_clients,
            min_num_clients = self.min_clients)
        return [(c, fl.common.EvaluateIns(parameters, {}))
                for c in clients]

    # ── Fit aggregation ──────────────────────────────────────────
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        us_r, mammo_r       = [], []
        fit_results_for_log = []

        for _, fit_res in results:
            arrays   = parameters_to_ndarrays(fit_res.parameters)
            modality = fit_res.metrics.get("modality", "ultrasound")
            entry    = (fit_res.num_examples, arrays)
            if modality == "ultrasound":
                us_r.append(entry)
            else:
                mammo_r.append(entry)
            fit_results_for_log.append(
                (fit_res.num_examples, fit_res.metrics))

        # Log
        self.fl_logger.log_fit_results(server_round,
                                       fit_results_for_log)

        # Drift
        drifts = [m.get("client_drift", 0.0)
                  for _, m in fit_results_for_log]
        if drifts:
            self.drift_log[server_round] = drifts

        # Aggregate
        if us_r and mammo_r:
            us_avg    = _capped_avg(us_r,    self.max_client_weight)
            mammo_avg = _capped_avg(mammo_r, self.max_client_weight)
            fused     = self._fuse(us_avg, mammo_avg)
        elif us_r:
            fused = _capped_avg(us_r,    self.max_client_weight)
        else:
            fused = _capped_avg(mammo_r, self.max_client_weight)

        self.last_global = fused
        return ndarrays_to_parameters(fused), {}

    # ── Cross-modal encoder fusion ────────────────────────────────
    def _fuse(self, us, mammo):
        """
        Fuse encoder; keep modality-specific decoders; avg cls head.

        Layout (ResNet-34, depth=5, dual decoders):
          [0 .. NE)          → shared encoder      (~55%)
          [NE .. NE+ND)      → US decoder + head   (~20%)
          [NE+ND .. NE+2*ND) → Mammo decoder + head (~20%)
          [NE+2*ND .. end)   → shared cls head     (~5%)
        """
        total = len(us)

        # ── FIX: use self._NE / self._ND (instance variables) ────
        if self._NE is None:
            self._NE = 216
            self._ND = 62
            log(INFO,
                f"[Fuse boundaries] "
                f"total={total} | "
                f"encoder=[0:{self._NE}] | "
                f"US-dec=[{self._NE}:{self._NE+self._ND}] | "
                f"Mammo-dec=[{self._NE+self._ND}:"
                f"{self._NE+2*self._ND}] | "
                f"cls=[{self._NE+2*self._ND}:{total}]")

        NE = self._NE
        ND = self._ND
        a  = self.fusion_alpha

        fused = []
        for i, (u, m) in enumerate(zip(us, mammo)):
            if   i < NE:          fused.append(a*u + (1-a)*m)  # encoder
            elif i < NE + ND:     fused.append(u)               # US dec
            elif i < NE + 2*ND:   fused.append(m)               # Mammo dec
            else:                 fused.append(0.5*u + 0.5*m)  # cls head
        return fused

    # ── Evaluate aggregation ─────────────────────────────────────
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        eval_log = [(r.loss, r.num_examples, r.metrics)
                    for _, r in results]
        self.fl_logger.log_eval_results(server_round, eval_log)

        total_ex = sum(r.num_examples for _, r in results)
        agg_loss = sum(r.loss * r.num_examples
                       for _, r in results) / total_ex

        def _mean(key):
            vals = [r.metrics[key] for _, r in results
                    if key in r.metrics]
            return float(np.mean(vals)) if vals else 0.0

        metrics = {
            "loss"       : agg_loss,
            "dice"       : _mean("dice"),
            "iou"        : _mean("iou"),
            "accuracy"   : _mean("accuracy"),
            "sensitivity": _mean("sensitivity"),
            "specificity": _mean("specificity"),
        }

        # Save all rounds
        self.round_metrics.append(
            {"round": server_round, **metrics})

        # Save best model
        if metrics["dice"] > self._best_dice:
            self._best_dice  = metrics["dice"]
            self._best_round = server_round

            if self.last_global is not None:
                import pickle
                save_path = f"{self.fl_logger.out_dir}/best_model.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(self.last_global, f)
                    self.fl_logger._log(f"  ★ New best model saved! "
                                        f"Round={server_round}  "
                    f"Dice={metrics['dice']:.4f}")

        return agg_loss, metrics

    # ── Centralised eval (disabled) ───────────────────────────────
    def evaluate(self, server_round, parameters):
        return None

    # ── Finalize ─────────────────────────────────────────────────
    def finalize(self):
        self.fl_logger._log(
            f"\n  Best Round : {self._best_round}"
            f"  Best Dice  : {self._best_dice:.4f}")
        return self.fl_logger.save_summary()