# Heart Murmur Detection

Deep learning classifiers that detect heart murmurs from phonocardiogram (PCG) recordings, using the [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/) dataset.

Project for **Machine Learning II (CC3043)**, DCC/FCUP, 2026/2027.

## Task

Given the heart sound recordings of a patient, predict the patient-level `Murmur` label:

| Label | Meaning |
|---|---|
| `Present` | A murmur was heard |
| `Absent` | No murmur was heard |
| `Unknown` | The expert could not decide (noise, poor recording quality) |

This is the murmur task of the [George B. Moody PhysioNet Challenge 2022](https://physionet.org/content/challenge-2022/1.0.0/).

Points that shape the design:

- **The label belongs to the patient, not to a recording.** Each patient has up to four recordings (aortic, pulmonary, tricuspid and mitral valves), and a murmur may be audible at only some of them. The system outputs exactly one prediction per patient.
- **The classes are imbalanced.** Most patients are `Absent`, and `Unknown` is the smallest class.

## Dataset

The public part of CirCor DigiScope has **3163 recordings from 942 patients** aged 0 to 21. Recordings last between 4.8 and 80.4 seconds.

The dataset is **not included** in this repository. To download it into `data/`:

```bash
scripts/download_data.sh
```

The script fetches the files from PhysioNet's open-data S3 mirror (much faster than the main PhysioNet server) and checks them against the official `SHA256SUMS.txt`.

You can also download the ZIP from the [dataset page](https://physionet.org/content/circor-heart-sound/1.0.3/). Place the contents in `data/` so that the layout is:

```
data/
├── training_data.csv        # one row per patient, includes the Murmur label
└── training_data/
    ├── <id>_<loc>.wav       # audio, loc ∈ {AV, PV, TV, MV, Phc}
    ├── <id>_<loc>.hea       # WFDB header
    ├── <id>_<loc>.tsv       # heart cycle segmentation (S1, systole, S2, diastole)
    └── <id>.txt             # patient metadata
```

## Approach

We implement two of the three architectures required by the assignment (MLP, CNN, RNN). Each model goes through the same four stages:

1. **Pre-processing:** resampling, splitting recordings into fixed-length windows, normalization, and feature extraction (e.g. mel-spectrograms, MFCCs).
2. **Model architecture:** the network design for each classifier.
3. **Training strategy:** loss and class-imbalance handling, optimizer, hyperparameter selection, regularization.
4. **Evaluation:** patient-level predictions scored with the weighted accuracy.

The recording-level predictions are aggregated into one prediction per patient.

> Model choices, design decisions and results will be documented here as the project progresses.

## Evaluation

### Protocol

- Data is split **by patient**: all recordings of a patient belong to the same subset.
- Splits are **stratified** by the `Murmur` label.
- The test set is never used for model decisions (architecture, hyperparameters, early stopping, thresholds). Those decisions use the validation set only.

### Metric

The main metric is the weighted accuracy from the PhysioNet Challenge 2022, computed at the patient level. `Present` patients weigh 5, `Unknown` 3 and `Absent` 1:

$$
\mathrm{WA}_{\text{murmur}} = \frac{5\,m_{PP} + 3\,m_{UU} + m_{AA}}{5\,(m_{PP}+m_{UP}+m_{AP}) + 3\,(m_{PU}+m_{UU}+m_{AU}) + (m_{PA}+m_{UA}+m_{AA})}
$$

Here $m_{ij}$ is the number of patients predicted as class $i$ whose true class is $j$. Reference implementation: [physionetchallenges/evaluation-2022](https://github.com/physionetchallenges/evaluation-2022).

For each classifier we report the weighted accuracy and the patient-level confusion matrix.

## Results

| Model | Weighted accuracy | Notes |
|---|---|---|
| — | — | — |

## Setup

```bash
git clone https://github.com/AndreXie05/HeartMurmurDetection.git
cd HeartMurmurDetection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

> Instructions to train the models and reproduce the reported results will be added here.

## Team

- André Xie
- _Member 2_
- _Member 3_
