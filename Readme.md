# 🚀 RateMamba | Domain-Adaptive Rate Prediction with Mamba for Wireless

<p align="center">
  <img src="docs/rateMamba.png" alt="RateMamba Logo" width="360"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg" alt="PyTorch"></a>
  <a href="https://github.com/username/RateMamba/actions"><img src="https://img.shields.io/github/actions/workflow/status/username/RateMamba/ci.yml?branch=main" alt="Build Status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/username/RateMamba/stargazers"><img src="https://img.shields.io/github/stars/username/RateMamba.svg" alt="Stars"></a>
</p>

<p align="center">
  <strong>Language:</strong>
  English |
  <a href="./README_zh.md">中文</a>
</p>

## 📖 Abstract
Targeting rate prediction in wireless systems with multi-antenna, multi-carrier, and beamforming, this work integrates domain adaptation and sequence modeling from the perspective of spatial/frequency-domain correlations. For Problem 1, we adopt the Exponential Effective SINR Mapping (EESM) with a Spearman-correlation objective and Nelder–Mead optimization of α and β, yielding a unified representation for non-TxBF conditions. Results show significant sub-carrier correlation, effective frequency-domain characterization by EESM, transferable fitted parameters across subsets, and overall stability with limited deviation under abrupt link changes. For Problem 2, we propose a Mamba-based sequence encoder that combines mean/attention pooling with FiLM-based domain-conditional affine modulation to mitigate regression-to-the-mean. Compared with FTTransformer, ResNet, and RNN baselines, the proposed model improves average F1 by ~10%, reduces mean bias, strengthens tail-case recognition, and achieves consistent gains in RMSE, MAE, and R². For Problem 3, we design a domain-adaptive architecture with a shared encoder, FiLM, and dual expert heads, where domain routing selects a dedicated prediction head to jointly model Non-TxBF and TxBF; this improves cross-domain overall accuracy by 7% without sacrificing task accuracy and reduces variance for higher stability. While some bias remains under extreme class imbalance, the approach substantially enhances predictive stability and generalization.

---
## 📸 Architecture & Results 📸 Architecture & Results 

**Mamba Architecture**

<p align="center">
  <img src="docs/structure.png" alt="Model Architecture" width="70%"/>
</p>
**Non-TxBF vs TxBF **

<p align="center">
  <img src="docs/non_txbf.png" alt="Non-TxBF" width="45%"/>
  <img src="docs/txbf.png" alt="TxBF" width="45%"/>
</p>

**Results Curves**
<p align="center">
  <img src="docs/q2_curve.png" alt="Q2 Curve" width="70%"/>
</p>
<p align="center">
  <img src="docs/q3_curve.png" alt="Q3 Curve" width="70%"/>
</p>

---


## ⚡ Quick Start
| Step | Command |
| ---- | ------- |
| 1    | `git clone https://github.com/MAXIMUM950814/RateMamba.git` |
| 2    | `cd RateMamba` |
| 3    | `pip install -r requirements.txt` |
---


## 📦 Installation
```bash
git clone https://github.com/MAXIMUM950814/RateMamba.git
cd RateMamba
pip install -r requirements.txt
```

## 📂 Project Structure

```
RateMamba/
├─ datasets/
	├─ feature_map
├─ docs/
├─ models/
├─ notebook/
├─ results/
	├─ q1_eesm
	├─ q2_mamba
	├─ q3_mamba
	├─ q3_mamba_non
├─ utils/
├─ main.py
├─ requirements.txt
└─ README.md
```


## 🚀 Experiment Workflow



### Experiment Workflow

**Step 1: Data Preprocessing ，SINR Calculation  **  
Run the following script to load the dataset and compute per-subcarrier SINR values:

```bash
python .\models\eesm.py
```

**Step 2: Effective SINR equivalent  Mapping with EESM**
 Apply the EESM model to map per-subcarrier SINR to an equivalent SINR and generate the `Valid_Predict` results:

```
python .\models\eesm.py \
  --train '/mnt/j/workspace/2025math/datasets/train_all_non_txbf.csv' \
  --valid '/mnt/j/workspace/2025math/datasets/valid_all_non_txbf.csv' \
  --per_sc_col sinr_per_sc_non \
  --label_col mcs \
  --outdir .results/eesm_out
```

**Step 3: Training the RateMamba Model**
 The RateMamba framework is used to train channel transmission rate prediction models. The workflow includes:

1. **Feature Embedding**

   ```
   python .\datasets\feature_embedding.py
   ```

2. **Model Training and Prediction**

   ```
   python main.py
   ```

Alternatively, the experiments can be reproduced step by step using the provided Jupyter notebooks:

```
notebook\mambular-baseline.ipynb 
notebook\mambular-nontxbf-2moe.ipynb 
notebook\mambular-txbf-2moe.ipynb 
notebook\mambular-txbf.ipynb
```

### 📊Results and Outputs

All experimental results are stored in the `results` directory:

- **Task 1 (Effective SINR Mapping):**
   Results, plots, and parameter files are saved in `\results\q1_eesm`.
   Prediction results:
   `valid_non_txbf_predict.csv`
- **Task 2 (Rate Prediction – Non-TxBF Scenario):**
   Results, plots, parameters, and trained model weights are also stored in `\results\q1_eesm`.
   Prediction results:
   `valid_pre_non_txbf_predict.csv`
- **Task 3 (Rate Prediction – TxBF and Non-TxBF Scenarios):**
   Results, plots, parameters, and trained model weights are saved in `\results\q3_mamba` and `\results\q3_mamba_non`.
   Prediction results:
  - `valid_pre_txbf_predict.csv`
  - `valid_pre_non_txbf_predict.csv`

## 👥 Contributors

Thanks to the following people who have contributed to this project:

<p align="left">
  <a href="https://github.com/Watch-A">Watch-A</a>
</p>
<p align="left">
  <a href="https://github.com/ArcadiaLin">ArcadiaLin</a>
</p>

## 📜LICENSE（MIT）

The entire codebase is under MIT license.