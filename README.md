# Human Activity & Behaviour Recognition using Pose Estimation and Machine Learning

A research-grade, real-time vision system that recognizes human activities from webcam video streams using **Google MediaPipe Pose** and classical **Machine Learning**. 

This repository implements an end-to-end pipeline: from raw video capture and 3D skeleton keypoint extraction to translation/scale-invariant feature engineering, rigorous **Leave-One-Subject-Out (LOSO)** evaluation, and live webcam deployment with heads-up display (HUD).

---

## 🚀 Key Features

- **Privacy-Preserving & Lightweight:** Operates exclusively on 33 anatomical pose landmarks, completely discarding facial identity and background pixels. Runs at 30+ FPS on standard commodity CPUs.
- **Scale- & Translation-Invariant Representation:** All 33 coordinates are centered at the mid-hip point and scaled by the rigid torso length, eliminating camera distance and screen placement artifacts.
- **158-Dimensional Feature Vector:** Combines 150 spatial features (coordinates, segment lengths, joint angles, visibilities) with 8 scale-invariant temporal dynamic features (velocities, accelerations, segmental displacements).
- **Rigorous Subject-Independent Benchmark:** Evaluated via 4-fold Leave-One-Subject-Out (LOSO) cross-validation to guarantee zero subject leakage and measure true real-world generalizability.
- **Turnkey Live Webcam Deployment:** Real-time OpenCV application featuring rolling temporal buffer queues, 7-frame majority voting smoothing, confidence scoring, and interactive keyboard controls.

---

## 📁 Project Structure

```
Human_Activity_Rec/
├── data/
│   ├── raw/
│   │   ├── pose_dataset.csv                         # Initial pilot dataset
│   │   └── multisubject_pose_dataset.csv            # 4-subject dataset (21,227 frames)
│   └── processed/
│       ├── upfall_pose_dataset.csv                  # UP-Fall standardized 3D pose
│       ├── upfall_normalized_temporal_features.csv  # UP-Fall normalized features
│       └── multisubject_normalized_temporal_features.csv # 158-feature multi-subject set (21,211 rows)
├── docs/
│   └── RESEARCH_REPORT.md                           # Comprehensive scientific research report
├── models/
│   ├── pose_landmarker_full.task                    # MediaPipe Pose Landmarker model
│   ├── final_deployment_model.pkl                   # Trained Random Forest deployment model
│   └── final_feature_schema.json                    # Exact 158-feature schema definition
├── results/
│   ├── multisubject_dataset_audit.txt               # Data quality & integrity report
│   ├── multisubject_feature_engineering_report.txt  # Feature generation report
│   ├── loso_evaluation_report.txt                   # Complete 4-fold LOSO evaluation report
│   ├── loso_error_analysis_report.txt               # Generalization diagnostics & ANOVA report
│   ├── final_model_training_report.txt              # Deployment model training report
│   ├── live_prediction_report.txt                   # Real-time deployment specification
│   ├── loso_model_comparison.png                    # LOSO model performance chart
│   └── loso_confusion_matrices.png                  # Cross-subject confusion matrices
├── src/
│   ├── pose_detection.py                            # MediaPipe video capture & skeleton drawer
│   ├── collect_multisubject_data.py                 # Multi-subject data collection tool
│   ├── audit_multisubject_dataset.py                # Dataset quality audit script
│   ├── multisubject_normalized_features.py          # 158-feature engineering pipeline
│   ├── evaluate_loso.py                             # Leave-One-Subject-Out cross-validation
│   ├── analyze_loso_errors.py                       # Error analysis & distribution shift diagnostics
│   ├── train_final_model.py                         # Deployment model trainer
│   ├── live_prediction.py                           # Real-time webcam prediction application
│   └── load_upfall.py                               # UP-Fall dataset ingestion adapter
├── README.md                                        # Project overview and quickstart
└── requirements.txt                                 # Core dependencies
```

---

## 📊 Dataset Description

The system was evaluated on a dedicated multi-subject dataset collected across 4 adult participants:

- **Subjects:** `S01`, `S02`, `S03`, `S04`
- **Target Activities:** 
  1. `standing` (Class 1)
  2. `sitting` (Class 2)
  3. `walking` (Class 3)
  4. `running` (Class 4)
- **Volume:** 21,227 raw frames across 16 continuous sequences (~1,326 frames per sequence, ~66 seconds each).
- **Data Quality:** 0 missing/NaN values, 0 infinite values, 0 duplicate rows, and strictly monotonic timestamps.
- **Processed Dataset:** 21,211 frames (after dropping the first frame of each sequence for temporal differential calculation).

---

## 🔬 Feature Engineering (158 Features)

| Feature Group | Count | Description |
|:---|:---:|:---|
| **Normalized Coordinates** | 99 | 33 landmarks × 3D $(x, y, z)$ centered at hip midpoint and divided by torso length. |
| **Normalized Segment Lengths** | 10 | Anatomical segment distances (shoulders, hips, arms, thighs, shins) scaled by torso length. |
| **Joint Angles** | 8 | Elbows, knees, shoulders, and hip flexion angles in degrees. |
| **Visibility Scores** | 33 | MediaPipe landmark tracking confidence scores $[0.0, 1.0]$. |
| **Temporal Dynamics** | 8 | Scale-normalized frame displacement (mean, max, upper body, lower body, left/right side), speed, and acceleration. |

---

## 📈 Results Summary (Subject-Independent LOSO)

Unlike random split evaluations that report artificial >95% accuracy due to temporal frame correlation, **Leave-One-Subject-Out (LOSO)** evaluation measures true generalization to completely unseen people:

| Metric | Logistic Regression | Random Forest (Best) |
|:---|:---:|:---:|
| **Overall Accuracy** | 60.97% ± 11.87% | **66.26% ± 11.34%** |
| **Macro F1 Score** | 60.29% ± 12.42% | **64.84% ± 13.13%** |
| **Weighted F1 Score** | 59.85% ± 13.05% | **64.48% ± 13.39%** |
| **Sitting F1** | 98.52% ± 1.53% | **99.78% ± 0.43%** |
| **Walking F1** | 52.52% ± 14.10% | **58.24% ± 9.45%** |
| **Running F1** | 51.86% ± 19.42% | **59.97% ± 16.62%** |
| **Standing F1** | 38.25% ± 27.53% | **41.36% ± 31.86%** |

### Key Diagnostic Insights
- **Sitting is highly generalizable:** Reached **99.78% F1** across unseen subjects; hip flexion angle provides an unambiguous signature.
- **Standing / Locomotion Confusion:** 64.03% of standing frames were classified as locomotion (walking or running). Diagnostic ANOVA revealed this was driven by camera truncation of lower limbs in Subject S01 (foot visibility dropped to 0.455 vs ~0.978 in other subjects), causing decision trees to misinterpret missing feet as running.
- **Walking / Running Boundary:** Overlap occurs due to natural differences in human cadence (e.g., S02 walks faster than S01 runs).

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Human_Activity_Rec.git
   cd Human_Activity_Rec
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify MediaPipe model:**
   Ensure `models/pose_landmarker_full.task` exists (downloaded automatically or placed manually).

---

## 🚦 How to Run

### 1. Run Data Quality Audit
```powershell
python src/audit_multisubject_dataset.py
```

### 2. Run Feature Engineering Pipeline
```powershell
python src/multisubject_normalized_features.py
```

### 3. Run Subject-Independent LOSO Evaluation
```powershell
python src/evaluate_loso.py
```

### 4. Run Error Diagnostics & ANOVA Shift Analysis
```powershell
python src/analyze_loso_errors.py
```

### 5. Train the Final Deployment Model
```powershell
python src/train_final_model.py
```

### 6. Verify Deployment Pipeline (No Camera Required)
```powershell
python src/live_prediction.py --verify
```

### 7. Launch Live Real-Time Webcam Prediction
```powershell
python src/live_prediction.py
```
- Press **`q`** to safely quit.
- Press **`r`** to reset temporal buffer and smoothing history.

---

## ⚠️ Limitations

1. **Cohort Size ($N=4$):** Evaluated across 4 individuals. While rigorous and leakage-free, expanding to larger, more diverse cohorts will further stabilize gait decision boundaries.
2. **Camera Framing Sensitivity:** The system requires full-body visibility including feet. Truncated lower limbs introduce visibility shifts that degrade standing recognition.
3. **Deployment Disclaimer:** The live webcam demo uses a model trained on all 4 subjects for demo stability. In-sample training accuracy (100%) must **never** be cited as subject-independent generalization performance; the genuine generalization metric is the LOSO result (**66.26% accuracy**).

---

## 🔮 Future Work

- **Visibility-Aware Pruning:** Remove raw landmark tracking visibility features to prevent camera-crop bias.
- **Windowed Temporal Aggregation:** Replace single-frame differential dynamics with 0.5s–1.0s windowed averages to smooth high-frequency sensor noise.
- **Sequence Deep Learning:** Implement Temporal Convolutional Networks (TCN) or lightweight GRU/Transformer models to learn long-range gait phase dynamics.
- **Test-Time Adaptation:** Include a 3-second standing calibration step to adapt velocity thresholds to each user's unique body scale.

---

## 📖 Complete Scientific Report

For comprehensive methodology, biomechanical formulas, confusion matrix breakdowns, and UP-Fall dataset validation notes, see the full research paper:
👉 [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)
