# NetGuard AI — Network Intrusion Detection using Machine Learning

An intermediate portfolio project that learns to classify network-flow records as **BENIGN** or **ATTACK** using a Random Forest classifier. Includes a Streamlit dashboard for CSV analysis, downloadable predictions, model metrics, confusion matrix, and feature-importance chart.

> **Important:** This is an offline educational prototype. It does not capture live network traffic or replace an IDS/firewall. Dataset scores may not generalize to real networks.

## Dataset

Use the official **CIC-IDS2017** dataset from the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/ids-2017.html

Download **MachineLearningCSV.zip** from the page. Start with the CSV files, not the much larger raw PCAP files.

### Folder setup

Extract the downloaded CSV files into this folder:

```text
network_intrusion_detection_project/
├── data/
│   └── raw/
│       ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│       ├── ...
├── models/
├── outputs/
├── app.py
├── train.py
└── requirements.txt
```

Exact filenames may vary. It is okay to keep the CSV files in subfolders inside `data/raw/`; the script searches recursively.

## Windows setup (PowerShell)

Open a terminal in the project folder. Python 3.12 is suitable.

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 2. Install libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Put the dataset in place

Extract the CSV files from `MachineLearningCSV.zip` into `data\raw\`.

### 4. Train the model

```powershell
python train.py
```

The training script samples up to 150,000 rows to keep memory use manageable. It cleans column names, converts features to numeric values, handles infinite/missing values, removes identifier fields, and creates a stratified train/test split. To use more rows later, change `MAX_ROWS` near the top of `train.py`.

Files produced:
- `models/intrusion_model.joblib` — trained pipeline
- `models/feature_columns.json` — exact training feature order
- `outputs/metrics.json` — test metrics
- `outputs/confusion_matrix.png` — confusion matrix
- `outputs/feature_importance.csv` — top feature importance values

### 5. Launch the dashboard

```powershell
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`). Open it in your browser and upload a compatible CIC-IDS2017 CSV. You can review predictions and download them as a new CSV.

## What the model predicts

- **BENIGN** — the record is labeled as benign by the model.
- **ATTACK** — the record is predicted to resemble attack traffic.

The attack class groups all non-BENIGN labels together for a first binary-classification version. This project can later be extended to multiclass attack identification.

## Evaluation

Do not report accuracy alone. Review:
- Precision for ATTACK: how many flagged rows were actually attacks in the test labels.
- Recall for ATTACK: how many attacks the model detected.
- F1-score: a balance of precision and recall.
- Confusion matrix: false positives and false negatives.

The test split is a useful first benchmark, but random row splits can produce optimistic results when related flows occur in both train and test sets. A stronger extension is to train on some days and test on a separate day or capture scenario.

## Troubleshooting

**`No CSV files found`**  
The dataset may still be zipped, or files are not inside `data/raw/`.

**`Only one class remains`**  
Make sure your extracted CSV set includes both BENIGN and attack records.

**Training is slow / memory usage is high**  
Close unnecessary programs and reduce `MAX_ROWS` to `50000`. Train from a subset first.

**`streamlit` is not recognized**  
Run `python -m streamlit run app.py` after activating the virtual environment.

**Dashboard says model not trained**  
Run `python train.py` successfully before launching the app.

## Portfolio checklist

1. Add screenshots of the dashboard and confusion matrix to the repository.
2. Record the actual test metrics from your run; never invent results.
3. Explain why recall and false-positive rate matter for intrusion detection.
4. Add a limitations section and describe your data split.
5. Do not claim it protects a live network unless you build and validate a separate live-traffic pipeline.
