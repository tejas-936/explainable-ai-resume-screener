# XAI Resume Bias Detection

An Explainable AI (XAI) tool for auditing and detecting bias in resume representations using a Hierarchical Transformer Variational Autoencoder (HTVAE). This system maps resumes into a latent professional space, performs sentence-level sensitivity analyses using occlusion, and conducts counterfactual audits to identify demographic, ethnic, and gender biases.

## Features
- **Hierarchical Encoding**: Tokenizes resumes at both the word level (using DistilBERT embeddings) and the sentence level (using a Transformer Encoder).
- **Variational Representation**: Learns a smooth, continuous latent space representing candidate professional profiles.
- **Explainable AI (XAI)**: Estimates the importance of individual resume sentences by computing the displacement in latent space after occlusion.
- **Counterfactual Auditing**: Automatically mutates demographic markers (names, gendered pronouns) to audit models for statistical fairness and latent-space bias.
- **Dual Interfaces**: 
  - **CLI Tools**: Clean command-line utility for preprocessing, training, and auditing.
  - **Interactive Dashboard**: A premium Streamlit dashboard with responsive visualizations (Radar maps, probability density curves, highlighted sentence heatmaps, and bias metrics).

---

## Folder Structure

```
d:\CODING\Python\xAI_resume\
│
├── config.py             # Model hyperparameters, path configurations, and bias settings
├── dataset.py            # Custom PyTorch Dataset for hierarchical documents
├── models.py             # PyTorch implementation of PositionalEncoding, HierarchicalEncoder, HTVAE
├── explainability.py     # Attribution algorithms, counterfactual audits, and text mutator utilities
├── train.py              # Loss functions and mixed-precision training loop
├── main.py               # Main CLI orchestrator
├── app.py                # Streamlit dashboard
├── requirements.txt      # Project library requirements
└── README.md             # Project documentation
```

---

## Installation & Setup

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Place your raw resume dataset `Resume.csv` (containing at least the column `Resume_str`) in this folder.
   If you have a custom file path, you can set the environment variable:
   ```powershell
   $env:RAW_CSV_PATH="your_file.csv"
   ```

---

## Command Line Interface (CLI) Usage

The `main.py` entrypoint provides three main command sub-groups:

### 1. Preprocess Data
Cleans missing values and compiles the dataset:
```bash
python main.py preprocess
```

### 2. Train the HTVAE Model
Trains the autoencoder on CPU or GPU (mixed precision AMP is dynamically handled):
```bash
python main.py train --epochs 10 --batch-size 8 --lr 2e-5
```
This will automatically preprocess the raw dataset if not already done, and output a model state checkpoint `htvae_production_checkpoint.pth`.

### 3. Run Explainability Audits
Run diagnostic sentence attributions and bias checks on a resume:
* **Auditing a candidate from the dataset**:
  ```bash
  python main.py audit --bias-swap female
  ```
* **Auditing custom raw text**:
  ```bash
  python main.py audit --resume-text "Highly motivated Software Engineer. Led Mr. Smith's dev team." --bias-swap female
  ```
* **Auditing a text file**:
  ```bash
  python main.py audit --resume-file candidate.txt --bias-swap minority
  ```

---

## Interactive Web Dashboard (Streamlit)

Start the beautiful dashboard interface locally:
```bash
streamlit run app.py
```
> **Note**: The web interface has a built-in **Demo fallback mode**. If no trained checkpoint `htvae_production_checkpoint.pth` is found in the directory, the dashboard will dynamically spin up with mock simulations so you can test all features (Visual charts, attributions, heatmaps, bias audit tables) instantly. Once you train the model, it will automatically switch to live neural model inferences.
