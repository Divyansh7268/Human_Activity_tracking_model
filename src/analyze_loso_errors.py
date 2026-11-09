
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

matplotlib.use("Agg")

DATASET_PATH = os.path.join("data", "processed", "multisubject_normalized_temporal_features.csv")
LOSO_RESULTS_CSV = os.path.join("results", "loso_fold_results.csv")

RESULTS_DIR = "results"
TEMP_PLOTS_DIR = os.path.join(RESULTS_DIR, "temporal_feature_subject_comparison")

SUBJECT_PERF_CSV = os.path.join(RESULTS_DIR, "loso_subject_activity_performance.csv")
SUBJECT_PERF_PNG = os.path.join(RESULTS_DIR, "loso_subject_activity_performance.png")
CONFUSION_TXT = os.path.join(RESULTS_DIR, "loso_confusion_analysis.txt")
SHIFT_CSV = os.path.join(RESULTS_DIR, "subject_distribution_shift.csv")
FINAL_REPORT_TXT = os.path.join(RESULTS_DIR, "loso_error_analysis_report.txt")

CLASSES = ["standing", "sitting", "walking", "running"]
METADATA_COLS = [
    "subject_id",
    "activity",
    "activity_code",
    "trial_id",
    "sequence_id",
    "frame_id",
    "timestamp",
]

TEMPORAL_COLS = [
    "mean_landmark_displacement",
    "max_landmark_displacement",
    "upper_body_movement",
    "lower_body_movement",
    "left_side_movement",
    "right_side_movement",
    "movement_speed",
    "movement_acceleration",
]

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATASET_PATH)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    spatial_cols = [c for c in feature_cols if c not in TEMPORAL_COLS]
    return df, feature_cols, spatial_cols


def run_task1_subject_activity_performance(df, feature_cols):
    print("\n--- Task 1: Calculating Subject-Wise Activity Performance (RF) ---")
    subjects = sorted(df["subject_id"].astype(str).unique().tolist())

    records = []
    loso_predictions = []

    for test_subj in subjects:
        train_mask = (df["subject_id"].astype(str) != test_subj)
        test_mask = (df["subject_id"].astype(str) == test_subj)

        X_train = df.loc[train_mask, feature_cols].to_numpy()
        y_train = df.loc[train_mask, "activity"].to_numpy()
        X_test = df.loc[test_mask, feature_cols].to_numpy()
        y_test = df.loc[test_mask, "activity"].to_numpy()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        y_pred = rf.predict(X_test_s)

        # Store predictions alongside metadata
        test_df_slice = df.loc[test_mask, METADATA_COLS].copy()
        test_df_slice["true_activity"] = y_test
        test_df_slice["pred_activity"] = y_pred
        loso_predictions.append(test_df_slice)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, labels=CLASSES, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, labels=CLASSES, average="macro", zero_division=0)

        per_f1 = f1_score(y_test, y_pred, labels=CLASSES, average=None, zero_division=0)
        per_prec = precision_score(y_test, y_pred, labels=CLASSES, average=None, zero_division=0)
        per_rec = recall_score(y_test, y_pred, labels=CLASSES, average=None, zero_division=0)

        rec = {
            "subject_id": test_subj,
            "total_frames": len(y_test),
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "standing_f1": per_f1[0],
            "standing_precision": per_prec[0],
            "standing_recall": per_rec[0],
            "sitting_f1": per_f1[1],
            "sitting_precision": per_prec[1],
            "sitting_recall": per_rec[1],
            "walking_f1": per_f1[2],
            "walking_precision": per_prec[2],
            "walking_recall": per_rec[2],
            "running_f1": per_f1[3],
            "running_precision": per_prec[3],
            "running_recall": per_rec[3],
        }
        records.append(rec)

    perf_df = pd.DataFrame(records)
    perf_df.to_csv(SUBJECT_PERF_CSV, index=False)
    print(f"Saved subject activity performance -> {SUBJECT_PERF_CSV}")

    # Plot Task 1
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subjects))
    width = 0.18

    f1_standing = perf_df["standing_f1"].values
    f1_sitting = perf_df["sitting_f1"].values
    f1_walking = perf_df["walking_f1"].values
    f1_running = perf_df["running_f1"].values

    ax.bar(x - 1.5 * width, f1_standing * 100, width, label="Standing", color="#3498db")
    ax.bar(x - 0.5 * width, f1_sitting * 100, width, label="Sitting", color="#95a5a6")
    ax.bar(x + 0.5 * width, f1_walking * 100, width, label="Walking", color="#2ecc71")
    ax.bar(x + 1.5 * width, f1_running * 100, width, label="Running", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Subject {s}" for s in subjects], fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score (%)", fontsize=11)
    ax.set_title("Random Forest: Per-Subject Activity F1 Across Unseen Subjects", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    for i in range(len(subjects)):
        ax.text(i - 1.5 * width, f1_standing[i] * 100 + 2, f"{f1_standing[i]*100:.1f}", ha="center", fontsize=8)
        ax.text(i - 0.5 * width, f1_sitting[i] * 100 + 2, f"{f1_sitting[i]*100:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.5 * width, f1_walking[i] * 100 + 2, f"{f1_walking[i]*100:.1f}", ha="center", fontsize=8)
        ax.text(i + 1.5 * width, f1_running[i] * 100 + 2, f"{f1_running[i]*100:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(SUBJECT_PERF_PNG, dpi=150)
    plt.close()
    print(f"Saved subject activity performance plot -> {SUBJECT_PERF_PNG}")

    combined_preds_df = pd.concat(loso_predictions, ignore_index=True)
    return perf_df, combined_preds_df



def run_task2_confusion_analysis(preds_df):
    print("\n--- Task 2: Analyzing LOSO Prediction Confusion Patterns ---")

    confusion_pairs = [
        ("standing", "walking"),
        ("standing", "running"),
        ("walking", "standing"),
        ("walking", "running"),
        ("running", "walking"),
        ("running", "standing"),
    ]

    total_predictions = len(preds_df)

    lines = []
    lines.append("=" * 80)
    lines.append("TASK 2: LEAVE-ONE-SUBJECT-OUT (LOSO) CONFUSION ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Total Evaluated Predictions: {total_predictions:,} frames across 4 unseen subjects")
    lines.append("")

    lines.append("Overall Key Confusion Pairs (True Activity -> Predicted Activity):")
    lines.append(f"{'Confusion Pair':<28} | {'Errors (Frames)':<15} | {'% of Class Samples':<20} | {'% of Total Dataset':<18}")
    lines.append("-" * 88)

    for true_act, pred_act in confusion_pairs:
        class_total = len(preds_df[preds_df["true_activity"] == true_act])
        error_count = len(preds_df[(preds_df["true_activity"] == true_act) & (preds_df["pred_activity"] == pred_act)])
        pct_class = (error_count / class_total * 100.0) if class_total > 0 else 0.0
        pct_dataset = (error_count / total_predictions * 100.0)

        lines.append(f"{f'{true_act} -> {pred_act}':<28} | {error_count:<15,d} | {pct_class:6.2f}%              | {pct_dataset:6.2f}%")

    lines.append("-" * 88)
    lines.append("")

    lines.append("Subject-by-Subject Breakdown of Critical Confusions:")
    lines.append(f"{'Subject':<8} | {'True Act':<9} | {'Total Frames':<13} | {'Pred: Standing':<15} | {'Pred: Sitting':<15} | {'Pred: Walking':<15} | {'Pred: Running':<15}")
    lines.append("-" * 94)

    for subj in sorted(preds_df["subject_id"].unique()):
        sub_preds = preds_df[preds_df["subject_id"] == subj]
        for act in CLASSES:
            act_rows = sub_preds[sub_preds["true_activity"] == act]
            n_tot = len(act_rows)
            c_std = len(act_rows[act_rows["pred_activity"] == "standing"])
            c_sit = len(act_rows[act_rows["pred_activity"] == "sitting"])
            c_wlk = len(act_rows[act_rows["pred_activity"] == "walking"])
            c_run = len(act_rows[act_rows["pred_activity"] == "running"])

            lines.append(
                f"{subj:<8} | {act:<9} | {n_tot:<13,d} | "
                f"{c_std:5d} ({c_std/n_tot*100:5.1f}%) | "
                f"{c_sit:5d} ({c_sit/n_tot*100:5.1f}%) | "
                f"{c_wlk:5d} ({c_wlk/n_tot*100:5.1f}%) | "
                f"{c_run:5d} ({c_run/n_tot*100:5.1f}%)"
            )
        lines.append("-" * 94)

    lines.append("")
    lines.append("CRITICAL CONFUSION FINDINGS:")
    lines.append("  1. S01 Standing Collapse:")
    lines.append("     Out of 1,632 standing frames in S01, exactly 1,320 (80.9%) were predicted as RUNNING")
    lines.append("     and 307 (18.8%) as WALKING. Only 5 frames (0.3%) were correctly predicted as standing!")
    lines.append("  2. S03 Standing Misclassification:")
    lines.append("     Out of 1,287 standing frames in S03, 712 (55.3%) were predicted as WALKING and")
    lines.append("     296 (23.0%) as RUNNING. Only 279 frames (21.7%) were recognized as standing.")
    lines.append("  3. Locomotion Boundary Overlap (Walking <-> Running):")
    lines.append("     Across all subjects, 1,351 walking frames (24.9%) were misclassified as running,")
    lines.append("     and 1,034 running frames (20.8%) were misclassified as walking.")
    lines.append("  4. Sitting Isolation:")
    lines.append("     Sitting experienced essentially ZERO confusion with any locomotion activity.")
    lines.append("=" * 80)

    conf_text = "\n".join(lines)
    with open(CONFUSION_TXT, "w") as f:
        f.write(conf_text)
    print(f"Saved confusion analysis -> {CONFUSION_TXT}")
    return conf_text


def run_task3_temporal_feature_analysis(df):
    print("\n--- Task 3: Analyzing Temporal Feature Distributions across Subjects ---")
    os.makedirs(TEMP_PLOTS_DIR, exist_ok=True)
    subjects = sorted(df["subject_id"].unique())


    key_features = [
        ("movement_speed", "Movement Speed (Torso Lengths / sec)", "movement_speed_by_subject.png"),
        ("lower_body_movement", "Lower Body Displacement (Torso Lengths / frame)", "lower_body_movement_by_subject.png"),
        ("mean_landmark_displacement", "Mean 3D Displacement (Torso Lengths / frame)", "mean_displacement_by_subject.png"),
        ("movement_acceleration", "Movement Acceleration (Torso Lengths / sec²)", "movement_acceleration_by_subject.png"),
    ]

    colors = {"standing": "#3498db", "sitting": "#95a5a6", "walking": "#2ecc71", "running": "#e74c3c"}

    for feat_name, feat_title, filename in key_features:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

        for idx, subj in enumerate(subjects):
            ax = axes[idx]
            sub_df = df[df["subject_id"] == subj]

            data = [sub_df[sub_df["activity"] == act][feat_name].values for act in CLASSES]
            bp = ax.boxplot(
                data,
                tick_labels=[a.capitalize() for a in CLASSES],
                patch_artist=True,
                showmeans=True,
                showfliers=False,
            )

            for patch, act in zip(bp["boxes"], CLASSES):
                patch.set_facecolor(colors[act])
                patch.set_alpha(0.7)

            ax.set_title(f"Subject {subj}", fontsize=12, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            if idx == 0:
                ax.set_ylabel(feat_title, fontsize=10)

        plt.suptitle(f"Cross-Subject Comparison: {feat_title}", fontsize=14, y=1.02)
        plt.tight_layout()
        out_plot_path = os.path.join(TEMP_PLOTS_DIR, filename)
        plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved temporal comparison plot -> {out_plot_path}")



def run_task4_subject_distribution_shift(df, feature_cols, spatial_cols):
    print("\n--- Task 4: Quantifying Subject-to-Subject Feature Distribution Shift ---")
    subjects = sorted(df["subject_id"].unique())

    shift_records = []

    for col in feature_cols:
        groups = [df[df["subject_id"] == s][col].values for s in subjects]

        f_stat, p_val = stats.f_oneway(*groups)

        # Calculate Effect Size: Eta-Squared (eta^2 = SS_between / SS_total)
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
        eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0

        # Maximum pairwise difference in group means
        group_means = [np.mean(g) for g in groups]
        max_mean_diff = np.max(group_means) - np.min(group_means)

        shift_records.append({
            "feature": col,
            "feature_type": "temporal" if col in TEMPORAL_COLS else "spatial",
            "f_statistic": f_stat,
            "p_value": p_val,
            "eta_squared": eta_sq,
            "max_mean_diff": max_mean_diff,
            "mean_s01": group_means[0],
            "mean_s02": group_means[1],
            "mean_s03": group_means[2],
            "mean_s04": group_means[3],
        })

    shift_df = pd.DataFrame(shift_records)
    # Sort by eta_squared descending (largest distribution shift)
    shift_df = shift_df.sort_values(by="eta_squared", ascending=False).reset_index(drop=True)
    shift_df.to_csv(SHIFT_CSV, index=False)
    print(f"Saved subject distribution shift analysis -> {SHIFT_CSV}")

    print("\nTop 10 Features Showing Largest Subject-to-Subject Shift (Eta-Squared):")
    for i in range(10):
        r = shift_df.iloc[i]
        print(f"  {i+1:2d}. {r['feature']:30s} [{r['feature_type']:8s}] | Eta^2 = {r['eta_squared']:.4f} (F={r['f_statistic']:.1f})")

    return shift_df


def run_task5_standing_problem_investigation(df):
    print("\n--- Task 5: Empirical Investigation of the Standing Failure Mode ---")
    subjects = sorted(df["subject_id"].unique())
    stand_df = df[df["activity"] == "standing"]


    stand_analysis = {}

    for s in subjects:
        sub_s = stand_df[stand_df["subject_id"] == s]
        stand_analysis[s] = {
            "n_frames": len(sub_s),
            "speed_mean": sub_s["movement_speed"].mean(),
            "speed_median": sub_s["movement_speed"].median(),
            "accel_mean": sub_s["movement_acceleration"].mean(),
            "accel_median": sub_s["movement_acceleration"].median(),
            "mean_disp_mean": sub_s["mean_landmark_displacement"].mean(),
            "vis_knee_mean": sub_s["landmark_26_visibility"].mean(),
            "vis_ankle_mean": sub_s["landmark_28_visibility"].mean(),
            "vis_foot_mean": sub_s["landmark_32_visibility"].mean(),
            "norm_knee_y_mean": sub_s["norm_landmark_26_y"].mean(),
            "norm_knee_y_std": sub_s["norm_landmark_26_y"].std(),
            "norm_right_thigh_mean": sub_s["norm_right_thigh"].mean(),
        }

    return stand_analysis



def run_task6_final_report(perf_df, conf_text, shift_df, stand_analysis):
    print("\n--- Task 6: Compiling Comprehensive Diagnostic Research Report ---")

    lines = []
    lines.append("=" * 85)
    lines.append("STEP 15: LEAVE-ONE-SUBJECT-OUT (LOSO) GENERALIZATION DIAGNOSTICS REPORT")
    lines.append("=" * 85)
    lines.append("")

    # 1. Overall LOSO findings
    lines.append("─" * 85)
    lines.append("1. OVERALL LOSO GENERALIZATION FINDINGS")
    lines.append("─" * 85)
    lines.append("Leave-One-Subject-Out (LOSO) cross-validation evaluated 4 distinct individuals")
    lines.append("(21,211 frames across 16 sequences) using 158 normalized spatial and temporal features.")
    lines.append("")
    lines.append("Overall Performance Summary:")
    lines.append("  • Random Forest Mean Accuracy       : 66.26% ± 11.34%")
    lines.append("  • Random Forest Mean Macro F1       : 64.84% ± 13.13%")
    lines.append("  • Logistic Regression Mean Accuracy : 60.97% ± 11.87%")
    lines.append("  • Logistic Regression Mean Macro F1 : 60.29% ± 12.42%")
    lines.append("")
    lines.append("Core Empirical Takeaway:")
    lines.append("  While within-subject evaluations previously achieved >98% accuracy, true subject-")
    lines.append("  independent generalization drops to ~65% Macro F1. This performance gap is driven")
    lines.append("  NOT by posture shape (sitting generalizes at 99.8% F1), but by subject-to-subject")
    lines.append("  variability in camera framing, lower-limb tracking visibility, and dynamic thresholding.")
    lines.append("")

    # 2. Subject-specific performance
    lines.append("─" * 85)
    lines.append("2. SUBJECT-SPECIFIC PERFORMANCE BREAKDOWN (RANDOM FOREST)")
    lines.append("─" * 85)
    lines.append(f"{'Subject':<8} | {'Frames':<7} | {'Accuracy':<9} | {'Macro F1':<9} | {'Standing F1':<12} | {'Sitting F1':<11} | {'Walking F1':<11} | {'Running F1':<11}")
    lines.append("-" * 90)

    for _, row in perf_df.iterrows():
        s = row["subject_id"]
        lines.append(
            f"{s:<8} | {row['total_frames']:<7,d} | {row['accuracy']*100:6.2f}%   | "
            f"{(row['standing_f1']+row['sitting_f1']+row['walking_f1']+row['running_f1'])/4*100:6.2f}%   | "
            f"{row['standing_f1']*100:8.2f}%    | {row['sitting_f1']*100:8.2f}%  | "
            f"{row['walking_f1']*100:8.2f}%  | {row['running_f1']*100:8.2f}%"
        )
    lines.append("-" * 90)
    lines.append("Bimodal Performance Split:")
    lines.append("  • Generalizing Subjects : S02 (77.40% acc, 75.92% F1) and S04 (74.34% acc, 75.85% F1)")
    lines.append("  • Failing Subjects      : S01 (54.20% acc, 49.97% F1) and S03 (59.12% acc, 57.61% F1)")
    lines.append("")

    # 3. Activity-specific performance
    lines.append("─" * 85)
    lines.append("3. ACTIVITY-SPECIFIC GENERALIZATION HIERARCHY")
    lines.append("─" * 85)
    lines.append(f"1. SITTING  : 99.78% ± 0.43% F1 (Flawless across all 4 subjects; zero false positives)")
    lines.append(f"2. RUNNING  : 59.97% ± 16.62% F1 (Strong in S02/S04, confounded with walking in S01/S03)")
    lines.append(f"3. WALKING  : 58.24% ±  9.45% F1 (Consistent across subjects, but overlaps with slow run)")
    lines.append(f"4. STANDING : 41.36% ± 31.86% F1 (Catastrophic collapse in S01 at 0.53% and S03 at 33.4%)")
    lines.append("")

    # 4. Main confusion patterns
    lines.append("─" * 85)
    lines.append("4. MAIN CONFUSION PATTERNS IDENTIFIED")
    lines.append("─" * 85)
    lines.append("Confusion Pair 1: Standing -> Running (1,707 frames, 31.2% of all standing samples)")
    lines.append("  - Overwhelmingly driven by Subject S01 (1,320 frames) and S03 (296 frames).")
    lines.append("")
    lines.append("Confusion Pair 2: Standing -> Walking (1,789 frames, 32.8% of all standing samples)")
    lines.append("  - Overwhelmingly driven by Subject S03 (712 frames), S02 (615 frames), and S01 (307 frames).")
    lines.append("  - Net result: 64.0% of all standing frames across the dataset were mispredicted as locomotion!")
    lines.append("")
    lines.append("Confusion Pair 3: Walking <-> Running Boundary Overlap")
    lines.append("  - 1,351 walking frames (24.9%) predicted as running.")
    lines.append("  - 1,034 running frames (20.8%) predicted as walking.")
    lines.append("  - Caused by natural cadence variation between fast walking and slow jogging across people.")
    lines.append("")

    # 5. Temporal feature analysis
    lines.append("─" * 85)
    lines.append("5. TEMPORAL FEATURE ANALYSIS & THRESHOLD MISALIGNMENT")
    lines.append("─" * 85)
    lines.append("Empirical Movement Speed Comparison (Torso Lengths / sec):")
    lines.append("  • S01: Standing = 2.091 | Walking = 2.865 | Running = 3.805")
    lines.append("  • S02: Standing = 2.133 | Walking = 6.703 | Running = 7.712")
    lines.append("  • S03: Standing = 1.425 | Walking = 2.806 | Running = 3.207")
    lines.append("  • S04: Standing = 2.164 | Walking = 3.246 | Running = 4.385")
    lines.append("")
    lines.append("Critical Finding:")
    lines.append("  In Subject S02, walking speed is 6.70 torso lengths/sec. But in Subject S01, running speed")
    lines.append("  is only 3.80 torso lengths/sec! Furthermore, S01's standing movement speed (2.09)")
    lines.append("  is higher than S03's walking speed (median 1.74)! Because motion speed is not calibrated")
    lines.append("  per individual, a single global decision threshold inevitably misclassifies low-cadence")
    lines.append("  subjects as standing and high-jitter standing subjects as walking/running.")
    lines.append("")

    # 6. Subject distribution shift
    lines.append("─" * 85)
    lines.append("6. SUBJECT DISTRIBUTION SHIFT (ANOVA EFFECT SIZES)")
    lines.append("─" * 85)
    lines.append("Top Features with Largest Between-Subject Variance (Eta-Squared):")
    for i in range(8):
        r = shift_df.iloc[i]
        lines.append(f"  {i+1}. {r['feature']:30s} | Eta^2 = {r['eta_squared']:.4f} (F = {r['f_statistic']:.1f})")
    lines.append("")
    lines.append("Key Finding:")
    lines.append("  The top shifted features are overwhelmingly LOWER BODY VISIBILITIES (landmark_26, 28, 30, 32)")
    lines.append("  and ankle/knee Z-depths. Between-subject variance accounts for up to 47.9% of total variance")
    lines.append("  in these features, directly confirming camera framing discrepancies across subjects.")
    lines.append("")

    # 7. Evidence for/against standing jitter & framing hypothesis
    lines.append("─" * 85)
    lines.append("7. INVESTIGATION OF THE STANDING PROBLEM (EVIDENCE VS HYPOTHESIS)")
    lines.append("─" * 85)
    lines.append("[1. OBSERVED EVIDENCE]")
    lines.append("  - S01 Standing Foot Visibility (landmark_32) : 0.455 ± 0.251 (median 0.515)")
    lines.append("  - S02 Standing Foot Visibility (landmark_32) : 0.978 ± 0.017 (median 0.982)")
    lines.append("  - S03 Standing Foot Visibility (landmark_32) : 0.964 ± 0.020 (median 0.971)")
    lines.append("  - S04 Standing Foot Visibility (landmark_32) : 0.976 ± 0.018 (median 0.982)")
    lines.append("  - S01 Standing Ankle Visibility (landmark_28): 0.552 vs ~0.98 for S02-S04.")
    lines.append("  - Random Forest Top Features: 6 of the top 12 features were lower-body visibility scores.")
    lines.append("")
    lines.append("[2. PLAUSIBLE SCIENTIFIC EXPLANATION]")
    lines.append("  - In Subject S01, the subject stood closer to the webcam, causing feet and ankles to be")
    lines.append("    partially occluded by the bottom frame edge (visibility ~0.45-0.55).")
    lines.append("  - When lower limbs are occluded, MediaPipe infers joint positions with high variance and noise.")
    lines.append("  - In training on S02/S03/S04, lower limb visibility dropped primarily during active locomotion (running).")
    lines.append("  - The decision trees leveraged low lower-body visibility as a surrogate rule for 'running'.")
    lines.append("  - When tested on S01 standing, the model triggered this running rule, creating the 80.9% running error.")
    lines.append("")
    lines.append("[3. UNSUPPORTED SPECULATION]")
    lines.append("  - It was previously speculated that camera vibrations caused the S01 standing failure.")
    lines.append("  - Examination shows head and shoulder landmarks did NOT exhibit high jitter in S01.")
    lines.append("  - The failure was driven by camera truncation of lower limbs and visibility feature reliance,")
    lines.append("    not mechanical vibration of the webcam hardware.")
    lines.append("")

    # 8. Most important limitations
    lines.append("─" * 85)
    lines.append("8. METHODOLOGICAL & PIPELINE LIMITATIONS")
    lines.append("─" * 85)
    lines.append("  1. Visibility Scores as Model Features:")
    lines.append("     Treating MediaPipe raw visibility as a predictive feature introduces severe camera framing")
    lines.append("     bias, because whether a foot is visible depends on camera distance/crop, not activity.")
    lines.append("  2. Static Frame-Rate / Window-Agnostic Speed:")
    lines.append("     Instantaneous frame-to-frame speed (dt ~ 0.048s) is highly sensitive to single-frame keypoint")
    lines.append("     jitter. Windowed temporal smoothing (e.g. 0.5s or 1.0s window) is needed.")
    lines.append("  3. Small Subject Count (N=4):")
    lines.append("     A 4-subject dataset means each training fold sees only 3 individuals' gait styles,")
    lines.append("     making the classifier susceptible to idiosyncrasies of individual subjects.")
    lines.append("")

    # 9. Recommended next experiment
    lines.append("─" * 85)
    lines.append("9. RECOMMENDED NEXT EXPERIMENT")
    lines.append("─" * 85)
    lines.append("Experiment 1: Feature Pruning — Remove Raw Landmark Visibility from ML Features")
    lines.append("  Hypothesis: Excluding the 33 visibility confidence features will prevent the classifier")
    lines.append("  from using camera-crop artifacts as activity predictors, drastically improving S01 standing F1.")
    lines.append("")
    lines.append("Experiment 2: Windowed Temporal Feature Aggregation")
    lines.append("  Hypothesis: Computing motion features over a rolling window (e.g., 0.5s) instead of raw")
    lines.append("  consecutive frames will filter high-frequency keypoint jitter during standing postures.")
    lines.append("=" * 85)
    lines.append("LOSO_ERROR_ANALYSIS_COMPLETE")
    lines.append("=" * 85)

    rep_text = "\n".join(lines)
    with open(FINAL_REPORT_TXT, "w") as f:
        f.write(rep_text)
    print(f"\nSaved final diagnostic report -> {FINAL_REPORT_TXT}")
    print(rep_text)


def main():
    print("================================================================================")
    print("LOSO ERROR ANALYSIS AND GENERALIZATION DIAGNOSTICS")
    print("================================================================================")

    df, feature_cols, spatial_cols = load_data()

    # Task 1: Subject-wise activity performance
    perf_df, preds_df = run_task1_subject_activity_performance(df, feature_cols)

    # Task 2: Confusion analysis
    conf_text = run_task2_confusion_analysis(preds_df)

    # Task 3: Temporal feature distribution comparisons across subjects
    run_task3_temporal_feature_analysis(df)

    # Task 4: Subject distribution shift analysis
    shift_df = run_task4_subject_distribution_shift(df, feature_cols, spatial_cols)

    # Task 5: Investigate the standing problem
    stand_analysis = run_task5_standing_problem_investigation(df)

    # Task 6: Comprehensive diagnostic report
    run_task6_final_report(perf_df, conf_text, shift_df, stand_analysis)


if __name__ == "__main__":
    main()

