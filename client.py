"""
====================================================================
client.py  —  Final v2
FedProx client for 8 datasets
====================================================================
"""

from copy import deepcopy
import numpy as np
import torch
import flwr as fl

import config as C
from model   import MultiTaskBreastModel, MultiTaskLoss, all_metrics
from dataset import get_dataloader


class FedProxClient(fl.client.NumPyClient):

    def __init__(self, dataset_name, mu=C.MU):
        self.dataset_name = dataset_name
        self.mu           = mu
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        kw = dict(batch_size=C.BATCH_SIZE,
                  image_size=C.IMAGE_SIZE, seed=C.SEED)
        self.trainloader = get_dataloader(
            dataset_name, "train", use_balanced=True,  **kw)
        self.valloader   = get_dataloader(
            dataset_name, "val",   use_balanced=False, **kw)

        self.modality  = next(iter(self.trainloader))["modality"][0]
        self.model     = MultiTaskBreastModel().to(self.device)
        self.criterion = MultiTaskLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50)
        self.drift_history = []

    def get_parameters(self, config):
        return [v.cpu().numpy()
                for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), params):
            sd[k] = torch.tensor(v)
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        gm = deepcopy(self.model).to(self.device)
        for p in gm.parameters():
            p.requires_grad_(False)

        epochs   = config.get("local_epochs", C.LOCAL_EPOCHS)
        run_loss = 0.0
        self.model.train()

        for _ in range(epochs):
            for batch in self.trainloader:
                imgs   = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                sp, cp = self.model(imgs, modality=self.modality)
                loss, _, _ = self.criterion(sp, masks, cp, labels)

                prox = sum(torch.norm(lp-gp)**2
                           for lp, gp in zip(
                               self.model.parameters(),
                               gm.parameters()))
                loss = loss + (self.mu / 2.0) * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), C.GRAD_CLIP)
                self.optimizer.step()
                run_loss += loss.item()

        self.scheduler.step()
        drift = self._drift(gm)
        self.drift_history.append(drift)

        return self.get_parameters(config), \
               len(self.trainloader.dataset), \
               {"train_loss"  : run_loss / max(len(self.trainloader), 1),
                "modality"    : self.modality,
                "dataset"     : self.dataset_name,
                "client_drift": float(drift)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        buf = {k: [] for k in
               ["dice","iou","accuracy","sensitivity","specificity"]}

        with torch.no_grad():
            for batch in self.valloader:
                imgs   = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)
                sp, cp = self.model(imgs, modality=self.modality)
                loss, _, _ = self.criterion(sp, masks, cp, labels)
                total_loss += loss.item()
                for k, v in all_metrics(sp, masks, cp, labels).items():
                    buf[k].append(v)

        n = max(len(self.valloader), 1)
        metrics = {k: float(np.mean(v)) for k, v in buf.items()}
        metrics.update(modality=self.modality,
                       dataset=self.dataset_name)
        return float(total_loss/n), \
               len(self.valloader.dataset), metrics

    @torch.no_grad()
    def _drift(self, gm):
        total, n = 0.0, 0
        for lp, gp in zip(self.model.parameters(), gm.parameters()):
            total += torch.norm(lp - gp).item(); n += 1
        return total / max(n, 1)


def client_fn(cid, mu=C.MU):
    return FedProxClient(C.CLIENT_MAP[str(cid)], mu=mu)
