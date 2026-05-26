"""
====================================================================
config.py  —  Final v2
8 datasets: 4 US + 4 Mammo
====================================================================
"""

# ── Paths ─────────────────────────────────────────────────────────
US_ROOT    = r"H:\BC datasets\US"
MAMMO_ROOT = r"H:\BC datasets\Mammo"

# ── Image ─────────────────────────────────────────────────────────
IMAGE_SIZE   = 256
BATCH_SIZE   = 16
NUM_WORKERS  = 0
VAL_RATIO    = 0.15
TEST_RATIO   = 0.10
SEED         = 42

CENTRAL_EPOCHS = 50

# ── Clients ───────────────────────────────────────────────────────
CLIENT_MAP = {
    "0": "BUSI",
    #"1": "BLUIS",
    "1": "BUS_UC",
    "2": "BrEaST",
    "3": "BUS_BRA",
    "4": "INbreast",
    "5": "DMID",
    "6": "CBIS_DDSM",
    "7": "AIISLAB",
}
NUM_CLIENTS = len(CLIENT_MAP)

MODALITY_MAP = {
    "BUSI"     : "ultrasound",
    #"BLUIS"    : "ultrasound",
    "BUS_UC"   : "ultrasound", 
    "BrEaST"   : "ultrasound",
    "BUS_BRA"  : "ultrasound",
    "INbreast" : "mammogram",
    "DMID"     : "mammogram",
    "CBIS_DDSM": "mammogram",
    "AIISLAB"  : "mammogram",
}

US_DATASETS    = [k for k, v in MODALITY_MAP.items() if v == "ultrasound"]
MAMMO_DATASETS = [k for k, v in MODALITY_MAP.items() if v == "mammogram"]

# ── Model ─────────────────────────────────────────────────────────
ENCODER_NAME  = "resnet34"
ENCODER_DEPTH = 5
PRETRAINED    = True

# ── Loss ──────────────────────────────────────────────────────────
LOSS_ALPHA = 0.7
LOSS_BETA  = 0.3

# ── Training ──────────────────────────────────────────────────────
MU           = 0.02
LR           = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0

# ── FL ────────────────────────────────────────────────────────────
FL_ROUNDS         = 100
LOCAL_EPOCHS      = 5
FUSION_ALPHA      = 0.5
MAX_CLIENT_WEIGHT = 0.4

# ── Output ────────────────────────────────────────────────────────
OUT_FL       = "./results_exp2/fl"
OUT_CENTRAL  = "./results/centralized"
OUT_ABLATION = "./results/ablation"
