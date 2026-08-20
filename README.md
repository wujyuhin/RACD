# RACD: Retrieval-Augmented Cognitive Diagnosis

This repository provides the implementation of **RACD**, a retrieval-augmented cognitive
diagnosis framework. The workflow consists of three steps: (1) train a basic NCDM model,
(2) pre-train an EAD model with distinctiveness regularization, and (3) fine-tune the RACD
model with hash retrieval and aggregation. Both transductive and inductive scenarios are
supported.

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

The `EduCDM` library is bundled under `EduCDM/` and is imported directly by the code, so no
separate installation is required.

## Data

The `math1` dataset is provided under `data/math1/`:

| Split | Path | Description |
| ----- | ---- | ----------- |
| Transductive | `data/math1/TransData/` | `train.csv` / `val.csv` / `test.csv` + `Q_mat.npy` |
| Inductive | `data/math1/InducData/` | `train.csv` + `val/` and `test/` splits at different remain rates + `Q_mat.npy` |

Statistics: 4209 students, 20 items, 11 knowledge concepts. All subscripts start from 0.

---

# Transductive Scenario

The workflow consists of three steps:

**Step 1**: Run `RACD/Transductive/s1_main_ncdm_trans.py`
- Trains a simple NCDM model on the transductive split.
- Model and student/question features saved at:
  - `data/math1/TransData/NCDM.pth`
  - `data/math1/TransData/NCDM_theta.csv`
  - `data/math1/TransData/NCDM_psi.csv`

**Step 2**: Run `RACD/Transductive/s2_main_pretrain_trans.py`
- Implements the EAD pre-trained model with distinctiveness regularization (as described in the paper).
- Pre-trained model saved at: `RACD/Transductive/report/EAD_report/`

**Step 3**: Run `RACD/Transductive/s3_main_finetue_trans.py`
- Implements the fine-tuned RACD model with distinctiveness regularization, hash retrieval, and aggregation.
- Results saved at: `RACD/Transductive/report/RACD_report/`

---

# Inductive Scenario

The workflow mirrors the Transductive scenario:

**Step 1**: Run `RACD/Inductive/s1_main_ncdm.py`
- Trains a basic NCDM model using inductive input data (`train.csv`).
- Model and student/question features saved at:
  - `data/math1/InducData/NCDM.pth`
  - `data/math1/InducData/NCDM_theta.csv`
  - `data/math1/InducData/NCDM_psi.csv`

**Step 2**: Run `RACD/Inductive/s2_main_pretrain.py`
- Implements the EAD pre-trained model with distinctiveness regularization.
- Pre-trained model saved at: `RACD/Inductive/report/EAD_report/`

**Step 3**: Run `RACD/Inductive/s3_main_finetue.py`
- Implements the fine-tuned RACD model with distinctiveness regularization, hash retrieval, and aggregation.
- Results saved at: `RACD/Inductive/report/RACD_report/`

---

## Repository Structure

```
.
├── RACD/
│   ├── Inductive/       # inductive scenario (s1/s2/s3 scripts, model/, NCDM/, utils/)
│   └── Transductive/    # transductive scenario (s1/s2/s3 scripts, model/, NCDM/, utils/)
├── data/math1/          # math1 dataset (TransData/ and InducData/)
├── EduCDM/              # bundled EduCDM library (dependency)
├── requirements.txt
└── LICENSE
```

## License

This project is released under the [MIT License](LICENSE).
