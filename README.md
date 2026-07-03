# REEF-NET — MLT-Corrected Electron Flux Forecasting for ISRO

> **Radiation-belt Electron Ensemble Forecaster** — forecasting energetic electron fluxes at geostationary orbit with Indian-longitude correction via ISRO's GRASP payload.

**Status:** ISRO Bharatiya Antariksh Hackathon 2026   
**Team:** REEF-NET (Aryan Bandyopadhyay, Asmit Chandra, Avinash Mishra, Faiz Ali)  
**Institute:** NISER, Bhubaneswar

---

## 🎯 One-Sentence Summary

We forecast GEO electron flux at 30-min, 6-hr, and 12-hr horizons using a **Temporal Fusion Transformer** with **conformal prediction intervals**, benchmarked against a fine-tuned **Chronos** time-series foundation model, with a **magnetic-local-time (MLT) correction** that adapts US-longitude GOES-trained models to ISRO's Indian-longitude GRASP measurements.

## 🏗 Three Pillars

| # | Pillar | Description |
|---|---|---|
| 01 | **TFT** (primary) | Temporal Fusion Transformer with 9 quantile outputs (3 horizons × 3 quantiles) |
| 02 | **Chronos** (benchmark) | Amazon's T5-based time-series foundation model, fine-tuned for comparison |
| 03 | **MLT Correction** (ISRO-specific) | 3-stage transfer learning using GRASP data at Indian longitude |

Wrapped in **conformal prediction intervals** for operational trust.

## 📊 Ablation Study

We conduct an ablation study comparing two versions of the TFT pipeline:
- **Version A (Control):** TFT trained on GOES only — no fine-tuning
- **Version B (Treatment):** TFT with MLT correction via GRASP fine-tuning

Both evaluated on **Accuracy, Precision, Recall, F1-score** with bootstrap 95% confidence intervals.

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/<your-org>/reef-net.git
cd reef-net

# Install
pip install -r requirements.txt
pip install -e .

# Download sample data (one month GOES + Wind)
python scripts/download_data.py --sample

# Run full preprocessing pipeline
python scripts/run_pipeline.py

# Train TFT (4-12 hours on single GPU)
python scripts/run_pipeline.py --train

# Run ablation study
python scripts/run_ablation.py

# Launch dashboard
streamlit run src/reef_net/app/dashboard.py
```

## 📁 Repository Layout

```
reef-net/
├── configs/                    # YAML configs (default, ablation)
├── data/
│   ├── raw/                    # Original CDF files (gitignored)
│   ├── interim/                # Cleaned parquet
│   └── processed/              # Train/val/test splits
├── docs/                       # Methodology, ablation study design
├── notebooks/                  # EDA, exploration
├── scripts/                    # CLI entry points
├── src/reef_net/
│   ├── data/                   # CDF loaders, preprocessing, features
│   ├── models/                 # TFT, Chronos, baselines
│   ├── training/               # Training loops
│   ├── inference/              # Prediction, conformal, MLT correction
│   ├── evaluation/             # Metrics, ablation runner
│   └── app/                    # Streamlit dashboard
├── tests/                      # Unit tests
├── .github/workflows/ci.yml    # CI pipeline
├── Dockerfile                  # Reproducible environment
├── Makefile                    # Common commands
└── requirements.txt            # Pinned dependencies
```

## 🛠 Tech Stack

- **Deep Learning:** PyTorch, pytorch-lightning, pytorch-forecasting, HuggingFace Transformers
- **Data:** cdflib, spacepy, sunpy, pandas, pyarrow
- **ML:** scikit-learn, Optuna
- **Viz:** Streamlit, Plotly, Matplotlib
- **Repro:** Docker, Hydra, pytest

## 📈 Evaluation Metrics

| Metric | Type | Purpose |
|---|---|---|
| Skill Score vs Persistence | Regression | Headline metric |
| CRPS | Probabilistic | Proper scoring rule |
| Brier Skill Score | Probabilistic | Exceedance events |
| Accuracy / Precision / Recall / F1 | Classification | Ablation study |
| Reliability Diagram | Calibration | Conformal validation |

**Baselines:** Persistence → Climatology → Boyd-Chi linear filter → NARX → **REEF-NET (ours)**

## 📚 Key References

- Lim et al. 2021 — *Temporal Fusion Transformers*, Int. J. Forecasting
- Ansari et al. 2024 — *Chronos: Learning the Language of Time Series*, arXiv:2403.07815
- Newell et al. 2007 — Solar wind coupling function, J. Geophys. Res.
- Boynton et al. 2011 — GEO electron flux prediction, J. Atmos. Solar-Terr. Phys.
- Sun & Saenko 2016 — Deep CORAL domain adaptation, ECCV Workshop
- Guo et al. 2017 — On Calibration of Modern Neural Networks, ICML

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- ISRO ISSDC for GRASP data access
- NOAA NCEI and NASA CDAWeb for public data archives
- Kyoto WDC for geomagnetic indices
