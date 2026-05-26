Requirements
Hardware

GPU: NVIDIA RTX 5080 Laptop (16 GB VRAM) or equivalent
RAM: ≥ 16 GB
Storage: ≥ 50 GB for all datasets

Software
bashconda create -n mafedprox python=3.10
conda activate mafedprox
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


Dataset Structure
datasets/
├── US/
│   ├── 01_BUSI/
│   ├── 02_BUS_UC/
│   ├── 03_BrEaST/
│   └── 04_BUS_BRA/
└── Mammo/
    ├── 01_INbreast/
    ├── 02_DMID/
    ├── 03_CBIS_DDSM/
    └── 04_AIISLAB/



How to Run

bash# Federated training
python main.py

# Centralised baseline
python baseline_centralized.py

# Ablation study
python ablation.py --rounds 15 --config A
python ablation.py --rounds 15 --config B
python ablation.py --rounds 15 --config C
python ablation.py --rounds 15 --config D
python ablation.py --rounds 15 --config E

# External validation
python external_validation_all.py
python optimal_threshold.py

# Figures
python visualize_results.py

# Statistics
python bootstrap_stats.py
python wilcoxon_eval.py


Project Structure
MAFedProx/
├── config.py
├── model.py
├── dataset.py
├── client.py
├── server.py
├── main.py
├── baseline_centralized.py
├── ablation.py
├── external_validation_all.py
├── optimal_threshold.py
├── visualize_results.py
├── bootstrap_stats.py
├── wilcoxon_eval.py
├── logger.py
└── requirements.txt


