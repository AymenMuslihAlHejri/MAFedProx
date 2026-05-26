"""
====================================================================
model.py  —  Final v2
Multi-task model: Shared Encoder + Dual Decoders + Shared Cls Head
====================================================================
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import config as C


def _build_decoder(enc_ch):
    try:
        dec = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels = enc_ch,
            decoder_channels = (256, 128, 64, 32, 16),
            n_blocks         = 5,
            use_batchnorm    = True,
            attention_type   = None,
        )
    except TypeError:
        dec = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels = enc_ch,
            decoder_channels = (256, 128, 64, 32, 16),
            n_blocks         = 5,
            attention_type   = None,
        )
    head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())
    return dec, head


class MultiTaskBreastModel(nn.Module):
    def __init__(self, encoder_name=C.ENCODER_NAME,
                 pretrained=C.PRETRAINED):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=1,
            depth=C.ENCODER_DEPTH,
            weights="imagenet" if pretrained else None)
        enc_ch = self.encoder.out_channels

        self.us_dec,    self.us_head    = _build_decoder(enc_ch)
        self.mammo_dec, self.mammo_head = _build_decoder(enc_ch)

        deepest = enc_ch[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(deepest, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x, modality="ultrasound"):
        feat = self.encoder(x)
        if modality == "ultrasound":
            seg = self.us_head(self.us_dec(*feat))
        else:
            seg = self.mammo_head(self.mammo_dec(*feat))
        cls = self.cls_head(feat[-1])
        return seg, cls


# ── Loss ─────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__(); self.s = smooth
    def forward(self, p, t):
        p, t  = p.view(-1), t.view(-1)
        inter = (p * t).sum()
        return 1. - (2.*inter + self.s) / (p.sum() + t.sum() + self.s)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce  = nn.functional.binary_cross_entropy(
            pred, target, reduction="none")
        pt   = torch.exp(-bce)
        loss = self.alpha * (1-pt)**self.gamma * bce
        return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=C.LOSS_ALPHA, beta=C.LOSS_BETA):
        super().__init__()
        self.a     = alpha; self.b = beta
        self.dice  = DiceLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)  # ← بدل BCE
        self.bce_c = nn.BCELoss()

    def forward(self, sp, st, cp, ct):
        ls = .5*self.dice(sp,st) + .5*self.focal(sp,st)  # ← focal
        lc = self.bce_c(cp.squeeze(1), ct)
        return self.a*ls + self.b*lc, ls, lc


# ── Metrics ───────────────────────────────────────────────────────
@torch.no_grad()
def dice_score(p, t, thr=.5):
    p, t = (p>thr).float().view(-1), t.view(-1)
    return (2*(p*t).sum()/(p.sum()+t.sum()+1e-6)).item()

@torch.no_grad()
def iou_score(p, t, thr=.5):
    p, t  = (p>thr).float().view(-1), t.view(-1)
    inter = (p*t).sum()
    return (inter/(p.sum()+t.sum()-inter+1e-6)).item()

@torch.no_grad()
def accuracy(cp, ct, thr=.5):
    return ((cp.squeeze()>thr).float()==ct).float().mean().item()

@torch.no_grad()
def sensitivity(p, t, thr=.5):
    p, t = (p>thr).float().view(-1), t.view(-1)
    tp   = (p*t).sum(); fn = ((1-p)*t).sum()
    return (tp/(tp+fn+1e-6)).item()

@torch.no_grad()
def specificity(p, t, thr=.5):
    p, t = (p>thr).float().view(-1), t.view(-1)
    tn   = ((1-p)*(1-t)).sum(); fp = (p*(1-t)).sum()
    return (tn/(tn+fp+1e-6)).item()

def all_metrics(sp, st, cp, ct):
    return {
        "dice"       : dice_score(sp, st),
        "iou"        : iou_score(sp, st),
        "accuracy"   : accuracy(cp, ct),
        "sensitivity": sensitivity(sp, st),
        "specificity": specificity(sp, st),
    }


if __name__ == "__main__":
    m = MultiTaskBreastModel()
    x = torch.randn(2, 1, 256, 256)
    for mod in ["ultrasound", "mammogram"]:
        s, c = m(x, mod)
        print(f"{mod}: seg={tuple(s.shape)}  cls={tuple(c.shape)}")
