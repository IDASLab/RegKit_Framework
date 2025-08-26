# 🧱 RegKit: A Config-Driven Regression Framework for Tabular Data

RegKit is a **modular, extensible framework** for running regression experiments on tabular datasets.  
It unifies **classical ML models** (XGBoost, LightGBM, CatBoost, etc.) and **deep learning architectures** (DNN, TabNet, FT-Transformer, NODE) under a **single YAML-driven workflow**.

---

## 🏗️ Architecture Overview

The framework is structured for clarity and reusability:

- **Configs** (`configs/config.yaml`)  
  One YAML file defines dataset, experiment setup, models, hyperparameter search, and training.

- **Models** (`models/`)  
  Registry-based system. Includes adapters for classical ML and custom PyTorch implementations (TabNet, FT-Transformer, NODE, DNN).

- **Pipelines** (`pipelines/`)  
  - `make_pipeline.py` → preprocessing builder (scaling, OHE, cat handling).  
  - `train_strategies/` → strategy files for training each model family.  

- **Utils** (`utils/`)  
  - `optuna_search.py` → Optuna-based HPO (Hyperparameters Optimization) with CV (Cross Validation) + early stopping.  
  - `metrics.py` → R², MAE, RMSE, MAPE, overfitting detection.  
  - `logger.py` → Logs experiments to Excel/CSV.  

- **Runner** (`run_training.py`)  
  Loads config, builds pipeline, trains/evaluates models, logs results.

---

## 🌟 Key Features & Use Cases

### 🔑 Features
- **Unified Interface**: Classical ML + DL models under one config.  
- **Custom Implementations**: Includes PyTorch-based TabNet, FT-Transformer, NODE.  
- **Hyperparameter Optimization**: Optuna + K-Fold CV integrated.  
- **Reproducibility**: YAML configs + logged outputs ensure repeatable experiments.  
- **Overfitting Control**: R² gap penalty + early stopping.  
- **Cross-Model Comparisons**: Fair benchmarking across multiple models.  

### 🎯 Best Suited For Projects That Need:
- Predicting **continuous variables** (e.g., material strength, finance, energy, health).  
- Comparing **ML vs DL** methods on the same dataset.  
- **Reproducible research pipelines**.  
- Rapid prototyping of **new architectures** via adapters & registry.  

---

## ⚙️ Setup

### *1. Clone the repo*
```bash
git clone https://github.com/yourname/regkit.git
cd regkit
```
### *2. Install PyTorch manually*
Because PyTorch installation depends on your CUDA version, install it first:

👉 Find the correct command at [PyTorch.org](https://PyTorch.org)

Examples:
```bash
# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU-only (no GPU support)
pip install torch torchvision torchaudio
```

### *3. Install remaining requirements*

```bash
pip install -r requirements.txt
```
---

## ▶️ Running

Default run (uses `configs/config.yaml`):

```bash
python run_training.py
```

Custom config path:

```bash
python run_training.py --config path/to/your_config.yaml
```

---

## 🛠️ Configuration (`configs/config.yaml`)

The config drives everything. Here’s a breakdown:

### 📁 Global Settings
```yaml
data:
  csv_path: real_estate_cleaned.csv
  target_column: price_per_unit_area
```
- `csv_path`: path to your dataset.  
- `target_column`: the regression target.  

```yaml
experiment:
  num_runs: 10
  log_file: ./experiment_log.xlsx
  model_name: dnn
```
- `num_runs`: repeat runs for statistical stability.  
- `log_file`: where to log metrics.  
- `model_name`: active model to train (`dnn`, `tabnet`, `ft_transformer`, `node`, etc.).  

---

### 🧩 Per-Model Config
Each model has a section in `models:` block.

#### Example: DNN
```yaml
models:
  dnn:
    model:
      hidden_dims: [[256,128], [512,256,128]]
      dropout:
        min: 0.0001
        max: 0.6
      batch_norm: [true, false]
```
- `hidden_dims`: possible layer structures.  
- `dropout`: search range for dropout.  
- `batch_norm`: toggle batch normalization.  

#### Example: TabNet
```yaml
tabnet:
  model:
    n_d: { choices: [64] }
    n_a: { choices: [32] }
    n_steps: { min: 3, max: 3 }
```
- `n_d`, `n_a`: dimensionalities for decision/attention steps.  
- `n_steps`: number of decision steps.  

---

### 🔍 Optuna Settings
```yaml
optuna:
  n_trials: 50
  kfold_splits: 5
  direction: maximize
```
- `n_trials`: how many trials to run.  
- `kfold_splits`: folds for CV during search.  
- `direction`: maximize (R²) or minimize (loss).  

---

### 🏋️ Training Settings
```yaml
training:
  epochs: 300
  batch_size: [16, 32, 64]
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.0001
```
- `epochs`: max training iterations.  
- `batch_size`: batch sizes to try.  
- `early_stopping`: patience (rounds to wait before stopping).  

---

## 📊 Output

After training, results are saved in logs (`experiment_log.xlsx`) with fields:

- `Model`  
- `Train_R2`, `CV_R2`  
- `R2_Gap`, `Overfitting`  
- `MAE`, `RMSE`, `MAPE`  
- `Best_Params`  

---


## 📜 License
This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.