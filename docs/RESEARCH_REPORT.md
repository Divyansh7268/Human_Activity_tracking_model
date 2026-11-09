# Human Activity & Behaviour Recognition using Pose Estimation and Machine Learning

**Authors / Project Team:** Antigravity AI & Autonomous Research Laboratory  
**Date:** November 2026  
**Repository:** `c:\Human_Activity_Rec`  
**Status:** Completed Research Project & Deployment Prototype  

---

## 2. Abstract

Human Activity Recognition (HAR) from monocular RGB video is a foundational task in pervasive computing, ambient assisted living, and human-computer interaction. While deep learning applied directly to raw pixel frames often learns spurious background or scene correlations, pose-based activity recognition offers a robust, privacy-preserving, and computationally efficient alternative by abstracting human movement into spatial landmarks. This research presents an end-to-end vision-based framework for classifying four core human activities: **standing**, **sitting**, **walking**, and **running**. Using Google MediaPipe Pose, 33 three-dimensional anatomical landmarks are extracted per frame and transformed into a 158-dimensional scale- and translation-invariant representation combining 150 spatial features (hip-centered coordinates, torso-scaled segment lengths, biomechanical joint angles, and tracking visibility scores) and 8 temporal motion features (landmark displacement, segmental dynamics, speed, and acceleration).

Crucially, this work confronts the fundamental challenge of **subject-independent generalization**. Evaluating machine learning models using random frame splits often yields artificially inflated test metrics (>98%) due to temporal autocorrelation and subject identity leakage. To establish rigorous empirical performance bounds, we collected a structured 4-subject dataset (21,227 raw frames, 16 full-length sequences) and conducted a complete **Leave-One-Subject-Out (LOSO)** cross-validation. On completely unseen individuals, Random Forest achieved **66.26% ± 11.34% accuracy** (Macro F1: **64.84% ± 13.13%**), outperforming Logistic Regression (**60.97% ± 11.87% accuracy**, Macro F1: **60.29% ± 12.42%**). Diagnostic error analysis revealed that sitting postures generalize near-flawlessly (99.78% F1), whereas standing postures suffer substantial degradation (41.36% F1), with 64.03% of standing frames misclassified as locomotion due to lower-limb camera truncation and visibility distribution shifts. A deployment model trained on all available data was integrated into a real-time webcam inference system with rolling temporal buffering and 7-frame majority smoothing. This document provides complete scientific documentation, diagnostic analyses, limitations, and reproducible guidelines.

---

## 3. Problem Statement

Automated human activity recognition from ubiquitous RGB sensors is challenging due to large intra-class variation, anthropometric heterogeneity (differing heights, limb ratios, and body builds), camera perspective discrepancies, and motion-blur artifacts. Traditional computer vision systems that operate directly on RGB grids suffer from high computational overhead, privacy intrusions, and susceptibility to background memorization.

Furthermore, a pervasive methodological flaw in existing activity recognition literature is the reliance on random train/test splits. Because consecutive video frames exhibit near-identical poses and sensor noise, randomly distributing frames across train and test sets tests the model's ability to interpolate within a known video stream rather than generalize across different human beings. The primary problem addressed in this research is: **Can a normalized, low-dimensional pose representation, coupled with lightweight machine learning, generalize across completely unseen individuals in a strictly subject-independent evaluation protocol?**

---

## 4. Research Motivation

1. **Privacy and Edge Computing:** Extracting skeleton coordinates discards facial identity and ambient environmental context, making pose-based systems privacy-compliant for domestic surveillance, hospital ward monitoring, and rehabilitation clinics.
2. **Computational Efficiency:** Abstracting high-resolution 1080p RGB video into a compact vector of 33 keypoints allows inference at 30+ FPS on standard commodity CPUs without requiring dedicated GPU accelerators.
3. **Scientific Rigor in Generalization:** Many published benchmarks claim >95% accuracy on benchmark datasets, only to collapse when exposed to a new individual with different proportions or gait rhythms. Establishing transparent, uninflated baseline metrics using Leave-One-Subject-Out (LOSO) evaluation provides honest insight into what classical models can and cannot achieve with pose landmarks.

---

## 5. Objectives

The project was executed across structured experimental stages to meet the following research objectives:
1. **Pose Ingestion Pipeline:** Develop an asynchronous, high-throughput pipeline to extract 33 MediaPipe pose landmarks from live webcam streams and pre-recorded video sequences.
2. **Translation- and Scale-Invariant Feature Engineering:** Design a coordinate normalization pipeline centered at the hip midpoint and scaled by rigid trunk length, eliminating camera-to-subject distance and frame-position artifacts.
3. **Temporal Movement Dynamics:** Formulate scale-invariant temporal features capturing velocity, acceleration, and segmental limb displacements across consecutive frames without introducing cross-sequence leakage.
4. **Standardized External Dataset Compatibility:** Build adapters to evaluate external public benchmarks (UP-Fall dataset) and audit their suitability for subject-independent benchmarking.
5. **Multi-Subject Data Acquisition:** Collect a clean, multi-subject dataset covering four activities across multiple participants with full trial balance.
6. **Strict Subject-Independent Evaluation:** Execute a complete Leave-One-Subject-Out (LOSO) cross-validation protocol to evaluate model generalization to entirely unseen subjects.
7. **Empirical Error & Distribution Shift Diagnostics:** Conduct ANOVA effect-size and confusion matrix analysis to identify root causes of inter-subject failure modes.
8. **Real-Time Deployment System:** Implement a fully functional, low-latency webcam prediction application with rolling temporal buffering, smoothing, and diagnostic HUD overlays.

---

## 6. System Overview

The system architecture follows a modular pipeline from raw visual sensing to stabilized activity classification:

```
[ Webcam / Video Input ]
           │
           ▼
[ MediaPipe Pose Landmarker ] ──► 33 Landmarks (x, y, z, visibility)
           │
           ▼
[ Pose Normalization Engine ]
  ├─ Reference: Hip Midpoint (Translation Invariant)
  ├─ Reference Scale: Torso Length (Scale Invariant)
  ├─ 99 Normalized Coordinates
  ├─ 10 Normalized Segment Distances
  ├─ 8 Biomechanical Joint Angles
  └─ 33 MediaPipe Tracking Visibilities (150 Spatial Features)
           │
           ▼
[ Temporal Dynamics Engine ]
  └─ Rolling Frame Buffer (dt = 0.15s) ──► 8 Scale-Normalized Temporal Features
           │
           ▼
[ 158-Dimensional Feature Vector ]
           │
           ▼
[ Random Forest Deployment Classifier ]
           │
           ▼
[ 7-Frame Majority-Vote Smoothing ]
           │
           ▼
[ Live HUD & Visual Feedback Display ]
```

---

## 7. Dataset

### Custom Multi-Subject Webcam Dataset
To ensure complete control over sequence boundaries, metadata integrity, and subject identity, a dedicated dataset was recorded using the project's standardized acquisition pipeline (`src/collect_multisubject_data.py`).

- **Participants:** 4 distinct subjects (`S01`, `S02`, `S03`, `S04`)
- **Target Activities:** 4 classes:
  1. `standing` (Class code 1)
  2. `sitting` (Class code 2)
  3. `walking` (Class code 3)
  4. `running` (Class code 4)
- **Total Raw Frames:** 21,227 frames
- **Total Sequences:** 16 independent trial sequences (4 subjects × 4 activities × 1 trial sequence)
- **Sampling Interval:** Nominal interval $\Delta t \approx 0.15\text{ s}$ (~20 FPS)
- **Storage Format:** `data/raw/multisubject_pose_dataset.csv` (139 columns: 7 metadata + 132 landmark coordinates and visibilities)

### Distribution and Coverage Matrix

| Subject ID | Standing Frames | Sitting Frames | Walking Frames | Running Frames | Total Subject Frames | Dataset Share |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S01** | 1,633 | 1,497 | 1,258 | 1,164 | 5,552 | 26.16% |
| **S02** | 1,225 | 1,427 | 1,080 | 1,170 | 4,902 | 23.09% |
| **S03** | 1,288 | 1,380 | 1,468 | 1,306 | 5,442 | 25.64% |
| **S04** | 1,318 | 1,046 | 1,628 | 1,339 | 5,331 | 25.11% |
| **Total** | **5,464** | **5,350** | **5,434** | **4,979** | **21,227** | **100.0%** |

### Data Quality Findings (Step 12 Quality Audit)
The comprehensive audit (`results/multisubject_dataset_audit.txt`) verified:
- **Missing / Infinite Values:** Exactly 0 NaN and 0 Inf values across all columns.
- **Duplicate Rows:** Exactly 0 duplicate rows.
- **Monotonic Sequence Integrity:** Frame IDs sequentially increase from $0$ to $N-1$ without gaps. Timestamps are strictly monotonically increasing.
- **Class Balance:** Class distribution across the dataset is balanced: Standing (25.74%), Sitting (25.20%), Walking (25.60%), Running (23.46%).
- **Subject Coverage:** 100% of subjects performed all 4 activities.

---

## 8. Pose Estimation

Pose estimation is executed via Google's **MediaPipe Pose Landmarker (Tasks API)** using the full-precision checkpoint (`models/pose_landmarker_full.task`).

### Landmark Topology
MediaPipe detects 33 full-body anatomical landmarks:
- **Landmarks 0–10:** Cranial and facial keypoints (Nose, Inner/Outer Eyes, Ears, Mouth)
- **Landmarks 11–16:** Upper limb segments (Left/Right Shoulders, Elbows, Wrists)
- **Landmarks 17–22:** Hand extremities (Left/Right Pinkies, Indexes, Thumbs)
- **Landmarks 23–28:** Core pelvic and lower limb joints (Left/Right Hips, Knees, Ankles)
- **Landmarks 29–32:** Foot keypoints (Left/Right Heels, Foot Indexes / Toes)

### Extracted Quantities
For each landmark $i \in \{0, \dots, 32\}$, MediaPipe outputs four values:
1. $x_i$: Normalized horizontal position in image space $[0.0, 1.0]$.
2. $y_i$: Normalized vertical position in image space $[0.0, 1.0]$.
3. $z_i$: Estimated depth relative to the mid-hip point.
4. $v_i$: Landmark tracking visibility confidence score $[0.0, 1.0]$.

Operating in `RunningMode.VIDEO` enforces temporal tracking priors across consecutive frames, minimizing joint fluttering.

---

## 9. Feature Engineering

To guarantee that the machine learning classifier cannot learn spurious spatial positions (such as where in the camera frame a participant stood), raw screen coordinates are strictly discarded. The final representation comprises **158 features** structured as follows:

```
Total Machine Learning Features: 158
 ├── Spatial Features (150)
 │    ├── 99 Normalized 3D Coordinates (33 landmarks × 3 [x, y, z])
 │    ├── 10 Normalized Segment Distances (Torso-scaled)
 │    ├──  8 Biomechanical Joint Angles (Degrees)
 │    └── 33 MediaPipe Visibility Scores ([0.0, 1.0])
 └── Temporal Features (8)
      └── Scale-invariant velocity, acceleration, and segmental displacements
```

### 1. Translation Normalization (Hip Centering)
The anatomical origin is defined as the midpoint of the left and right hip joints:
$$\mathbf{p}_{\text{hip\_mid}} = \frac{\mathbf{p}_{\text{left\_hip}} + \mathbf{p}_{\text{right\_hip}}}{2}$$
Every landmark coordinate is centered relative to this origin:
$$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_{\text{hip\_mid}}, \quad \forall i \in \{0, \dots, 32\}$$

### 2. Scale Normalization (Torso Length Scaling)
The characteristic torso length $L_{\text{torso}}$ is computed as the Euclidean distance between the shoulder midpoint and the hip midpoint:
$$\mathbf{p}_{\text{shoulder\_mid}} = \frac{\mathbf{p}_{\text{left\_shoulder}} + \mathbf{p}_{\text{right\_shoulder}}}{2}$$
$$L_{\text{torso}} = \|\mathbf{p}_{\text{shoulder\_mid}} - \mathbf{p}_{\text{hip\_mid}}\|_2$$
To prevent division-by-zero when a subject is occluded, $s = \max(L_{\text{torso}}, 10^{-4})$. The 99 normalized landmark coordinates are:
$$\hat{\mathbf{p}}_i = \frac{\mathbf{p}'_i}{s} = \left[ \frac{x'_i}{s}, \frac{y'_i}{s}, \frac{z'_i}{s} \right]$$

### 3. Normalized Body Segment Lengths (10 Features)
Ten key anatomical distances are computed and divided by $s$:
1. `norm_shoulder_width` = $\|\mathbf{p}_{\text{left\_shoulder}} - \mathbf{p}_{\text{right\_shoulder}}\|_2 / s$
2. `norm_hip_width` = $\|\mathbf{p}_{\text{left\_hip}} - \mathbf{p}_{\text{right\_hip}}\|_2 / s$
3. `norm_left_upper_arm` = $\|\mathbf{p}_{\text{left\_shoulder}} - \mathbf{p}_{\text{left\_elbow}}\|_2 / s$
4. `norm_left_forearm` = $\|\mathbf{p}_{\text{left\_elbow}} - \mathbf{p}_{\text{left\_wrist}}\|_2 / s$
5. `norm_right_upper_arm` = $\|\mathbf{p}_{\text{right\_shoulder}} - \mathbf{p}_{\text{right\_elbow}}\|_2 / s$
6. `norm_right_forearm` = $\|\mathbf{p}_{\text{right\_elbow}} - \mathbf{p}_{\text{right\_wrist}}\|_2 / s$
7. `norm_left_thigh` = $\|\mathbf{p}_{\text{left\_hip}} - \mathbf{p}_{\text{left\_knee}}\|_2 / s$
8. `norm_left_shin` = $\|\mathbf{p}_{\text{left\_knee}} - \mathbf{p}_{\text{left\_ankle}}\|_2 / s$
9. `norm_right_thigh` = $\|\mathbf{p}_{\text{right\_hip}} - \mathbf{p}_{\text{right\_knee}}\|_2 / s$
10. `norm_right_shin` = $\|\mathbf{p}_{\text{right\_knee}} - \mathbf{p}_{\text{right\_ankle}}\|_2 / s$

### 4. Biomechanical Joint Angles (8 Features)
Joint angles are naturally scale- and translation-invariant. For vectors $\mathbf{u} = \mathbf{a} - \mathbf{b}$ and $\mathbf{v} = \mathbf{c} - \mathbf{b}$ forming an angle at vertex $\mathbf{b}$:
$$\theta = \arccos\left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2 + \epsilon} \right) \times \frac{180^\circ}{\pi}$$
The 8 computed angles are:
- Left & Right Elbow Angles ($\mathbf{p}_{\text{shoulder}} \to \mathbf{p}_{\text{elbow}} \to \mathbf{p}_{\text{wrist}}$)
- Left & Right Knee Angles ($\mathbf{p}_{\text{hip}} \to \mathbf{p}_{\text{knee}} \to \mathbf{p}_{\text{ankle}}$)
- Left & Right Shoulder Angles ($\mathbf{p}_{\text{hip}} \to \mathbf{p}_{\text{shoulder}} \to \mathbf{p}_{\text{elbow}}$)
- Left & Right Hip Angles ($\mathbf{p}_{\text{shoulder}} \to \mathbf{p}_{\text{hip}} \to \mathbf{p}_{\text{knee}}$) — crucial for separating seated and upright postures.

### 5. Landmark Tracking Visibility Confidence (33 Features)
The raw MediaPipe visibility scores $v_i \in [0.0, 1.0]$ for all 33 landmarks are retained to provide confidence cues regarding joint occlusion.

---

## 10. Temporal Features

Static spatial features alone struggle to differentiate standing upright from pausing during a walk, or to separate walking from running. To resolve movement dynamics, **8 scale-invariant temporal features** are calculated between consecutive frames ($t$ and $t-1$) strictly within continuous sequence boundaries:

$$\mathbf{p}_{\text{scaled}}(t) = \frac{\mathbf{p}_{\text{raw}}(t)}{s(t)}$$
$$\Delta \mathbf{p}_i(t) = \|\mathbf{p}_{\text{scaled}, i}(t) - \mathbf{p}_{\text{scaled}, i}(t-1)\|_2, \quad i \in \{0, \dots, 32\}$$

The 8 temporal metrics are defined as follows:

1. **`mean_landmark_displacement`**: Average displacement across all 33 landmarks:
   $$\bar{D}(t) = \frac{1}{33} \sum_{i=0}^{32} \Delta \mathbf{p}_i(t)$$
2. **`max_landmark_displacement`**: Peak landmark displacement, capturing sudden limb thrusts:
   $$D_{\max}(t) = \max_{i} \Delta \mathbf{p}_i(t)$$
3. **`upper_body_movement`**: Mean displacement of head, shoulders, arms, and hands (indices $0 \le i \le 22$).
4. **`lower_body_movement`**: Mean displacement of pelvic and lower limb landmarks (indices $23 \le i \le 32$).
5. **`left_side_movement`**: Mean displacement of all left-side anatomical landmarks ($i \in \{1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31\}$).
6. **`right_side_movement`**: Mean displacement of all right-side anatomical landmarks ($i \in \{4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32\}$).
7. **`movement_speed`**: Overall displacement divided by the sampling time interval:
   $$V(t) = \frac{\bar{D}(t)}{\Delta t} \quad (\Delta t = 0.15\text{ s})$$
8. **`movement_acceleration`**: Absolute rate of change in speed:
   $$A(t) = \frac{|V(t) - V(t-1)|}{\Delta t}$$

*Boundary Handling:* The first frame ($t=0$) of each sequence lacks a preceding frame; exactly 16 boundary rows were dropped, leaving 21,211 processed rows (99.92% retention) with zero cross-sequence contamination.

---

## 11. Machine Learning Models

Two complementary model paradigms were investigated:

### 1. Logistic Regression (Linear Baseline)
- **Implementation:** `sklearn.linear_model.LogisticRegression`
- **Configuration:** `max_iter=1000`, `solver='lbfgs'`, multi-class cross-entropy.
- **Preprocessing:** All 158 features standardized via `StandardScaler` fitted strictly on training subjects inside each cross-validation fold.
- **Role:** Establishes the linear separability of the 158-dimensional feature space.

### 2. Random Forest Classifier (Non-linear Ensemble)
- **Implementation:** `sklearn.ensemble.RandomForestClassifier`
- **Configuration:** `n_estimators=200`, `random_state=42`, `n_jobs=-1`.
- **Properties:** Invariant to monotonic feature scalings; captures non-linear interactions (e.g., knee flexion combined with high lower-body speed); robust against localized outlier keypoint jitter.
- **Role:** Primary non-linear classifier for benchmark evaluation and real-time deployment.

---

## 12. Evaluation Methodology

### Why Subject-Independent Evaluation is Mandatory
In human activity recognition, random frame-level train/test splitting causes severe **data leakage**:
1. **Temporal Proximity:** Frames $t$ and $t+1$ from the same video recording share virtually identical clothing, lighting, camera angle, and posture.
2. **Identity Memorization:** Classifiers memorize individual anthropometric idiosyncrasies (e.g., Subject 1's specific shoulder-to-hip ratio) rather than learning generalized activity patterns.
3. **Overoptimistic Claims:** Models tested on random splits often report 98–99% accuracy, but catastrophically fail when deployed on a new user.

### Leave-One-Subject-Out (LOSO) Cross-Validation
To measure true generalizability, we implemented a 4-fold **Leave-One-Subject-Out (LOSO)** evaluation (`src/evaluate_loso.py`). In each fold, data from three subjects formed the training set, while data from the fourth subject was held out entirely as the test set:

- **Fold 1:** Train on `S02 + S03 + S04` (15,663 frames) $\to$ Test on `S01` (5,548 frames)
- **Fold 2:** Train on `S01 + S03 + S04` (16,313 frames) $\to$ Test on `S02` (4,898 frames)
- **Fold 3:** Train on `S01 + S02 + S04` (15,773 frames) $\to$ Test on `S03` (5,438 frames)
- **Fold 4:** Train on `S01 + S02 + S03` (15,884 frames) $\to$ Test on `S04` (5,327 frames)

*Strict Experimental Safeguards:*
- Zero frames of the test subject exist in the training partition.
- `StandardScaler` is fitted strictly on the training partition in each fold.
- No hyperparameter tuning or feature selection was performed on the held-out subject.

---

## 13. Results

### Cross-Subject Benchmark Summary

The actual cross-subject evaluation metrics across all 4 folds are summarized below:

| Metric | Logistic Regression | Random Forest | Advantage (RF vs LR) |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **60.97% ± 11.87%** | **66.26% ± 11.34%** | **+5.29%** |
| **Macro F1 Score** | **60.29% ± 12.42%** | **64.84% ± 13.13%** | **+4.55%** |
| **Weighted F1 Score** | **59.85% ± 13.05%** | **64.48% ± 13.39%** | **+4.63%** |
| **Macro Precision** | 60.74% ± 14.37% | 66.87% ± 14.79% | +6.13% |
| **Macro Recall** | 61.73% ± 11.17% | 67.02% ± 11.09% | +5.29% |
| **Standing F1** | 38.25% ± 27.53% | **41.36% ± 31.86%** | +3.11% |
| **Sitting F1** | 98.52% ± 1.53% | **99.78% ± 0.43%** | +1.26% |
| **Walking F1** | 52.52% ± 14.10% | **58.24% ± 9.45%** | +5.72% |
| **Running F1** | 51.86% ± 19.42% | **59.97% ± 16.62%** | +8.11% |

### Fold-by-Fold Performance Breakdown (Random Forest)

| Held-out Subject | Test Frames | Accuracy | Macro F1 | Weighted F1 | Standing F1 | Sitting F1 | Walking F1 | Running F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S01** | 5,548 | 54.20% | 49.97% | 48.87% | **0.53%** | 99.13% | 57.77% | 42.44% |
| **S02** | 4,898 | 77.40% | 75.92% | 77.18% | 57.83% | 100.00% | 65.00% | 80.86% |
| **S03** | 5,438 | 59.12% | 57.61% | 57.89% | 33.43% | 100.00% | 45.04% | 51.97% |
| **S04** | 5,327 | 74.34% | 75.85% | 73.95% | 73.66% | 100.00% | 65.15% | 64.61% |
| **Mean ± Std** | — | **66.26% ± 11.34%** | **64.84% ± 13.13%** | **64.48% ± 13.39%** | **41.36% ± 31.86%** | **99.78% ± 0.43%** | **58.24% ± 9.45%** | **59.97% ± 16.62%** |

### Class-Level Findings
1. **Sitting Generalizes Near-Flawlessly (99.78% F1):** Sitting was classified almost perfectly across all four subjects, reaching 100.00% F1 in S02, S03, and S04, and 99.13% in S01. Normalized hip flexion angles combined with static temporal features provide an unambiguous, subject-invariant biomechanical signature.
2. **Standing is the Weakest Class (41.36% F1):** Standing exhibited extreme between-subject variance ($\pm 31.86\%$), collapsing to a near-zero F1 score of 0.53% on Subject S01 and 33.43% on S03, while recovering to 73.66% on S04.
3. **Locomotion Confusion (Walking vs. Running):** Walking (58.24% F1) and Running (59.97% F1) exhibited consistent mutual confusion across all subjects due to interpersonal differences in natural cadence.

---

## 14. Error Analysis

To understand why performance split into generalizing subjects (`S02`, `S04`) and failing subjects (`S01`, `S03`), a diagnostic error analysis was performed (`src/analyze_loso_errors.py`).

### Major Confusion Patterns
Examining all 21,211 held-out predictions identified three primary error modes:

```
Confusion Summary (True ──► Predicted):
 ├── Standing ──► Walking   : 1,789 frames (32.77% of all standing samples)
 ├── Standing ──► Running   : 1,692 frames (30.99% of all standing samples)
 ├── Walking  ──► Running   : 1,351 frames (24.88% of all walking samples)
 └── Running  ──► Walking   : 1,234 frames (24.80% of all running samples)
```

**Critical Discovery:** Out of 5,460 true standing frames across the dataset, **3,481 frames (64.03%) were misclassified as active locomotion** (walking or running).

### Investigation of the S01 Standing Collapse
In Subject S01, out of 1,632 standing frames, 1,320 (80.9%) were classified as running, 307 (18.8%) as walking, and only 5 frames (0.3%) were correctly predicted as standing.

To determine why the model predicted intense running when the subject was stationary:
1. **Hypothesis: Camera Vibration / Keypoint Jitter?**  
   *Rejected by evidence:* Cranial and shoulder keypoints exhibited minimal displacement in S01 standing sequences ($\bar{D} \approx 0.012$).
2. **Hypothesis: Lower-Body Occlusion and Visibility Shift?**  
   *Strongly supported by evidence:* One-way ANOVA across subjects revealed that the features with the highest between-subject variance ($\eta^2$) were overwhelmingly **lower-body visibility scores**:

| Rank | Feature | Between-Subject Effect Size ($\eta^2$) | ANOVA $F$-Statistic |
|:---:|:---|:---:|:---:|
| 1 | `landmark_31_visibility` (Left Foot Index) | **0.2351** | $F = 2173.1$ |
| 2 | `landmark_32_visibility` (Right Foot Index) | **0.2123** | $F = 1904.9$ |
| 3 | `landmark_27_visibility` (Left Ankle) | **0.2087** | $F = 1864.2$ |
| 4 | `landmark_29_visibility` (Left Heel) | **0.1962** | $F = 1725.3$ |
| 5 | `landmark_28_visibility` (Right Ankle) | **0.1891** | $F = 1648.4$ |

### Scientific Interpretation
- During S01 data collection, the participant stood closer to the webcam, causing the feet and ankles to be partially truncated by the bottom edge of the image frame.
- Consequently, S01's foot visibility (`landmark_32`) dropped to **$0.455 \pm 0.251$** (median: 0.515), compared to **$0.978 \pm 0.017$** in S02, **$0.964 \pm 0.020$** in S03, and **$0.976 \pm 0.018$** in S04.
- In the training sets (S02, S03, S04), lower-body visibility dropped primarily during active running (due to rapid limb excursion and motion blur). The Random Forest learned that low foot visibility is a strong surrogate indicator of running.
- When presented with S01 standing with truncated feet, the decision trees fired their "running" rules. Camera framing geometry—rather than subject movement—governed the misclassification.

### Locomotion Threshold Misalignment
Inter-individual speed variability further compounded the walking/running boundary:
- S02 Walking Speed: **6.70 torso lengths/sec**
- S01 Running Speed: **3.80 torso lengths/sec**
- S03 Walking Speed: **2.81 torso lengths/sec**

Because Subject S02 walks faster than Subject S01 runs, a single global speed threshold cannot separate walking from running across different people without personalized calibration.

---

## 15. UP-Fall Dataset Validation

Prior to multi-subject webcam collection, the pipeline was tested on the publicly available **UP-Fall Detection Dataset** (Zenodo Record 12773013) to explore whether existing external benchmarks could serve for subject-independent evaluation.

### Adapter Integration & Findings
1. **Schema Adapter (`src/load_upfall.py`):** Successfully ingested UP-Fall 3D skeleton CSVs and mapped 33 joint definitions into the project's internal format (`data/processed/upfall_pose_dataset.csv`), validating compatibility with both spatial and temporal feature extraction.
2. **Initial Pilot Results (`results/upfall_initial_model_test.txt`):** On an initial 800-frame subset (8 sequences, 3 activities: standing, sitting, walking), Logistic Regression achieved 96.30% accuracy and Random Forest achieved 94.61% accuracy.
3. **Why UP-Fall Could NOT Serve as Final LOSO Benchmark:**  
   The audit (`results/subject_independent_evaluation_readiness.txt`) revealed fundamental limitations:
   - **Sample Size ($N=2$):** The published 3D skeleton archive only contained Activities of Daily Living (ADL: A6 walking, A7 standing, A8 sitting) for **two genuine subjects** (Subject 1 and Subject 2).
   - **Severe Class Imbalance:** Subject 2 contained zero walking trials. Holding out Subject 2 left walking unrepresented in testing; holding out Subject 1 left the model with zero walking training examples.
   - **Missing Class:** UP-Fall contains no running protocol.
   - **Verdict:** True subject-independent evaluation was impossible on UP-Fall without fabricating data. Following strict research integrity guidelines, UP-Fall was retained as an adapter validation exercise, and custom multi-subject data collection was commissioned.

---

## 16. Deployment

A real-time webcam inference system was built in `src/live_prediction.py` utilizing the complete 158-feature representation.

### Deployment Model (`models/final_deployment_model.pkl`)
- **Architecture:** Random Forest Classifier (`n_estimators=200`, `random_state=42`)
- **Training Data:** Trained on all 21,211 frames across all 4 subjects to provide maximal stability for live demonstrations.
- **In-Sample Training Accuracy:** 100.00% (memorization diagnostic; not representative of generalizability).
- **Feature Schema:** Enforced via `models/final_feature_schema.json` (exact ordering of all 158 features).

### Real-Time Inference Pipeline
1. **Webcam Capture:** OpenCV captures RGB frames at native webcam frame rates.
2. **MediaPipe Inference:** Runs `PoseLandmarker` in `RunningMode.VIDEO` with monotonic millisecond timestamps.
3. **Rolling Temporal Buffer:** Maintains a deque of recent frames (up to 30 frames). Each entry stores raw landmark coordinates and calculated torso length.
4. **Graceful Warmup:** Predictions are suppressed until at least $\ge 3$ frames are buffered (`MIN_BUFFER_FRAMES=3`), preventing numerical instability during initial startup.
5. **Prediction Smoothing:** Stores the last 7 raw model predictions (`SMOOTH_WINDOW=7`) and outputs the majority-voted class to eliminate momentary flicker.
6. **Live Heads-Up Display (HUD):**
   - Color-coded activity label (Green: Standing, Orange: Sitting, Cyan: Walking, Red: Running)
   - Real-time classification confidence percentage
   - Live frames-per-second (FPS) counter
   - Buffer depth indicator and smoothing status
   - Rendered 33-landmark skeleton overlay
7. **Interactive Keyboard Controls:**
   - `q`: Safely releases camera hardware, closes OpenCV windows, and exits cleanly.
   - `r`: Clears temporal deques and prediction history, resetting the model state.
8. **Failure Handling:** Automatically skips frames with occluded poses, handles zero torso lengths via numerical clamping ($\epsilon = 10^{-4}$), and catches camera acquisition dropouts without crashing.

---

## 17. Limitations

1. **Small Cohort Size ($N=4$):** While four subjects provided a valid, leakage-free LOSO cross-validation, an $N=4$ sample size is small. Each training fold had access to only three gait patterns.
2. **Environmental & Sensor Homogeneity:** Data was recorded using identical webcam hardware and fixed indoor settings. Real-world deployment across varying focal lengths, elevated angles, and low-light environments remains unmeasured.
3. **Locomotion Overlap:** Distinguishing fast walking from slow running using purely kinematic displacement is sensitive to individual stride lengths.
4. **Visibility Vulnerability:** MediaPipe tracking visibility scores introduced significant bias when lower limbs were truncated by the field of view.
5. **Deployment Model Interpretation:** The deployment model was trained on all available data for live demonstration purposes. Its training performance must **never** be cited as evidence of subject-independent generalization; true generalization remains the LOSO metric (66.26% accuracy).

---

## 18. Future Work

1. **Scale Subject Diversity:** Expand the dataset to 20+ participants representing broader age groups, body mass indices, and footwear styles.
2. **Visibility Feature Pruning:** Eliminate raw visibility confidence scores from the ML feature set or replace them with binary occlusion indicators to prevent framing-dependent classification artifacts.
3. **Windowed Temporal Aggregation:** Compute velocity and acceleration over rolling multi-frame windows (e.g., 0.5 s or 1.0 s) rather than instantaneous $t$ vs $t-1$ differences to filter high-frequency sensor noise.
4. **Sequential Deep Learning Models:** Evaluate recurrent architectures (LSTM, GRU) and Temporal Convolutional Networks (TCN) or Transformers (e.g., PoseConv3D) that model gait phase transitions over multi-second temporal receptive fields.
5. **Subject-Adaptive Normalization:** Explore adaptive test-time calibration where a brief 3-second standing baseline dynamically scales velocity and stride thresholds for the active user.
6. **Activity Expansion:** Incorporate transition classes (sit-to-stand, stand-to-sit) and hazardous events such as tripping, slipping, and falling.

---

## 19. Conclusion

This project designed, implemented, and empirically audited an end-to-end vision-based human activity recognition pipeline using MediaPipe Pose and machine learning. By constructing a 158-dimensional scale- and translation-invariant feature space, the system decouples activity recognition from camera distance and screen position.

Critically, this investigation highlights the stark divergence between random frame-level evaluation (>95% accuracy) and rigorous, leakage-free **Leave-One-Subject-Out cross-validation (66.26% ± 11.34% accuracy)**. While static postures such as sitting transfer across individuals with near-perfect fidelity (99.78% F1), dynamic locomotion and standing postures face significant generalization hurdles caused by camera framing truncation, visibility feature shifts, and natural variations in human cadence. Rather than claiming artificially inflated performance metrics, this work establishes an honest, reproducible empirical benchmark and delivers a robust, real-time webcam prototype for continuous activity monitoring.

---

## 20. Reproducibility

### Environment & Dependencies
- **Operating System:** Windows 10 / 11
- **Python Version:** 3.11.x (managed via virtual environment at `.venv\`)
- **Core Libraries:**
  - `mediapipe >= 0.10.14`
  - `opencv-python >= 4.9.0`
  - `scikit-learn >= 1.4.0`
  - `numpy >= 1.26.0`
  - `pandas >= 2.2.0`
  - `joblib >= 1.3.0`
  - `matplotlib >= 3.8.0`

### Main Scripts Directory

| Script | Purpose |
|:---|:---|
| `src/pose_detection.py` | MediaPipe pose detection and visual verification |
| `src/collect_multisubject_data.py` | Interactive multi-subject data collection tool |
| `src/audit_multisubject_dataset.py` | Dataset quality, integrity, and balance audit |
| `src/multisubject_normalized_features.py` | 158-feature spatial and temporal engineering pipeline |
| `src/evaluate_loso.py` | Complete Leave-One-Subject-Out (LOSO) cross-validation |
| `src/analyze_loso_errors.py` | Diagnostic confusion analysis and ANOVA effect size |
| `src/train_final_model.py` | Trains final deployment Random Forest model |
| `src/live_prediction.py` | Real-time webcam application with HUD overlay |
| `src/load_upfall.py` | UP-Fall external benchmark ingestion adapter |

### Primary Data & Model Artifacts
- **Raw Pose Dataset:** `data/raw/multisubject_pose_dataset.csv` (21,227 rows)
- **Processed Feature Dataset:** `data/processed/multisubject_normalized_temporal_features.csv` (21,211 rows, 165 columns)
- **Deployment Model:** `models/final_deployment_model.pkl`
- **Feature Schema:** `models/final_feature_schema.json` (158 feature names in exact order)
- **MediaPipe Pose Model:** `models/pose_landmarker_full.task`

### Step-by-Step Reproduction Commands

```powershell
# 1. Activate project virtual environment
cd c:\Human_Activity_Rec
.\.venv\Scripts\Activate.ps1

# 2. Run dataset quality audit
python src/audit_multisubject_dataset.py

# 3. Generate 158 normalized spatial + temporal features
python src/multisubject_normalized_features.py

# 4. Execute Leave-One-Subject-Out (LOSO) cross-validation
python src/evaluate_loso.py

# 5. Run generalization error analysis and ANOVA shift diagnostics
python src/analyze_loso_errors.py

# 6. Train final deployment model on complete dataset
python src/train_final_model.py

# 7. Verify live prediction pipeline startup (dry-run, no camera required)
python src/live_prediction.py --verify

# 8. Launch real-time live webcam activity recognition
python src/live_prediction.py
```

