
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

CSV_PATH = os.path.join("data", "raw", "multisubject_pose_dataset.csv")
REPORT_PATH = os.path.join("results", "multisubject_dataset_report.txt")
CHART_PATH = os.path.join("results", "multisubject_activity_distribution.png")

TARGET_ACTIVITIES = ["standing", "sitting", "walking", "running"]
NUM_LANDMARKS = 33
EXPECTED_COLUMNS = 7 + (NUM_LANDMARKS * 4)  # 139


def generate_empty_report():
    """Write an informative report when the dataset is not yet collected."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    text = (
        "================================================================================\n"
        "MULTI-SUBJECT DATASET AUDIT REPORT\n"
        "================================================================================\n"
        f"Target Dataset: {CSV_PATH}\n"
        "Status: FILE NOT FOUND OR EMPTY.\n\n"
        "To collect data, run:\n"
        "  python src/collect_multisubject_data.py\n\n"
        "Protocol Requirement:\n"
        "  5+ subjects x 4 activities (standing, sitting, walking, running) x 3+ trials.\n"
        "================================================================================\n"
    )
    with open(REPORT_PATH, "w") as f:
        f.write(text)
    print(text)


def inspect_dataset() -> bool:
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        print(f"[NOTICE] Dataset file not found at: {CSV_PATH}")
        generate_empty_report()
        return False

    df = pd.read_csv(CSV_PATH)
    if len(df) == 0:
        print(f"[NOTICE] Dataset at {CSV_PATH} has 0 rows.")
        generate_empty_report()
        return False

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("RESEARCH-GRADE MULTI-SUBJECT POSE DATASET INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Source file       : {CSV_PATH}")
    lines.append(f"Total Rows (Frames): {len(df)}")
    lines.append(f"Total Columns     : {len(df.columns)} (Expected: {EXPECTED_COLUMNS})")
    lines.append("")

    # 1. Subject & Sequence Metrics
    subjects = sorted(df["subject_id"].astype(str).unique().tolist())
    sequences = sorted(df["sequence_id"].unique().tolist())
    total_subjects = len(subjects)
    total_sequences = len(sequences)

    lines.append("─" * 80)
    lines.append("1. SUBJECT & SEQUENCE OVERVIEW")
    lines.append("─" * 80)
    lines.append(f"Total Unique Subjects   : {total_subjects}")
    lines.append(f"Subject IDs             : {', '.join(subjects)}")
    lines.append(f"Total Unique Sequences  : {total_sequences}")
    lines.append("")

    # Sequences and Frames per subject
    seq_per_subj = df.groupby("subject_id")["sequence_id"].nunique()
    frames_per_subj = df["subject_id"].value_counts().sort_index()

    lines.append("Breakdown by Subject:")
    lines.append(f"{'Subject ID':<12} | {'Sequences':<10} | {'Total Frames':<14} | {'Activities Present'}")
    lines.append("-" * 75)
    for s in subjects:
        sub_df = df[df["subject_id"].astype(str) == s]
        n_seq = sub_df["sequence_id"].nunique()
        n_frames = len(sub_df)
        acts = sorted(sub_df["activity"].unique().tolist())
        lines.append(f"{s:<12} | {n_seq:<10} | {n_frames:<14} | {', '.join(acts)}")
    lines.append("")

    # 2. Activity Metrics
    lines.append("─" * 80)
    lines.append("2. ACTIVITY DISTRIBUTION")
    lines.append("─" * 80)
    seq_per_act = df.groupby("activity")["sequence_id"].nunique()
    frames_per_act = df["activity"].value_counts()

    lines.append(f"{'Activity':<12} | {'Sequences':<10} | {'Total Frames':<14} | {'% of Total Frames'}")
    lines.append("-" * 65)
    for act in TARGET_ACTIVITIES:
        n_seq = seq_per_act.get(act, 0)
        n_frames = frames_per_act.get(act, 0)
        pct = (n_frames / len(df) * 100.0) if len(df) > 0 else 0.0
        lines.append(f"{act:<12} | {n_seq:<10} | {n_frames:<14} | {pct:6.2f}%")
    lines.append("")

    # 3. Sequence Length Statistics (Frames per sequence)
    lines.append("─" * 80)
    lines.append("3. SEQUENCE DURATION & FRAME STATISTICS")
    lines.append("─" * 80)
    frames_per_seq = df.groupby("sequence_id")["frame_id"].count()
    min_frames = int(frames_per_seq.min())
    max_frames = int(frames_per_seq.max())
    mean_frames = float(frames_per_seq.mean())
    std_frames = float(frames_per_seq.std()) if len(frames_per_seq) > 1 else 0.0

    lines.append(f"Min Frames per Sequence  : {min_frames}")
    lines.append(f"Max Frames per Sequence  : {max_frames}")
    lines.append(f"Mean Frames per Sequence : {mean_frames:.2f} (±{std_frames:.2f})")
    lines.append("")

    # 4. Data Quality & Integrity Checks
    lines.append("─" * 80)
    lines.append("4. DATA QUALITY & INTEGRITY AUDIT")
    lines.append("─" * 80)

    # Missing values
    missing_total = int(df.isnull().sum().sum())
    lines.append(f"Total Missing (NaN) Values : {missing_total}")
    if missing_total > 0:
        missing_cols = df.isnull().sum()[lambda x: x > 0]
        lines.append(f"  Columns with missing values: {missing_cols.to_dict()}")

    # Duplicate rows
    dup_rows = int(df.duplicated().sum())
    lines.append(f"Exact Duplicate Rows       : {dup_rows}")

    # Duplicate sequence IDs across different subjects/activities
    seq_subj_check = df.groupby("sequence_id")["subject_id"].nunique()
    multi_subject_seqs = seq_subj_check[seq_subj_check > 1].index.tolist()
    if multi_subject_seqs:
        lines.append(f"WARNING: Sequences spanning multiple subjects: {multi_subject_seqs}")
    else:
        lines.append("Sequence ID Uniqueness    : PASSED (Every sequence belongs to exactly one subject)")

    # Frame continuity check: per sequence, frame_id should be 0, 1, 2, ..., N-1
    continuity_issues = []
    for seq_id, seq_group in df.groupby("sequence_id"):
        frame_ids = seq_group["frame_id"].values
        expected_ids = np.arange(len(frame_ids))
        if not np.array_equal(frame_ids, expected_ids):
            continuity_issues.append(seq_id)

    if continuity_issues:
        lines.append(f"Frame Continuity Check     : FAILED in {len(continuity_issues)} sequences ({continuity_issues[:3]}...)")
    else:
        lines.append("Frame Continuity Check     : PASSED (Monotonically sequential starting at 0 for all sequences)")

    # 5. Activity Completeness Check
    lines.append("")
    lines.append("─" * 80)
    lines.append("5. SUBJECT-INDEPENDENT PROTOCOL READINESS CHECK")
    lines.append("─" * 80)
    missing_activity_report = {}
    for s in subjects:
        sub_acts = set(df[df["subject_id"].astype(str) == s]["activity"].unique())
        missing_acts = set(TARGET_ACTIVITIES) - sub_acts
        if missing_acts:
            missing_activity_report[s] = sorted(list(missing_acts))

    lines.append(f"Subject Count Target (>=5) : {'MET' if total_subjects >= 5 else f'NOT MET ({total_subjects}/5 collected)'}")
    if missing_activity_report:
        lines.append(f"Subjects Missing Target Activities ({len(missing_activity_report)}):")
        for s, m_acts in missing_activity_report.items():
            lines.append(f"  • {s} is missing: {', '.join(m_acts)}")
    else:
        lines.append("All subjects have all 4 target activities: YES (100% complete coverage)")

    lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nSaved report -> {REPORT_PATH}")

    # Generate Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frames per Activity
    act_counts = [frames_per_act.get(a, 0) for a in TARGET_ACTIVITIES]
    colors = ["#3498db", "#95a5a6", "#2ecc71", "#e74c3c"]
    axes[0].bar([a.capitalize() for a in TARGET_ACTIVITIES], act_counts, color=colors, edgecolor="black")
    axes[0].set_title("Total Frames per Activity", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Frame Count", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, c in enumerate(act_counts):
        axes[0].text(i, c + max(act_counts) * 0.01 + 1, str(c), ha="center", fontweight="bold")

    # Plot 2: Sequences per Subject
    subj_seq_counts = [seq_per_subj.get(s, 0) for s in subjects]
    axes[1].bar(subjects, subj_seq_counts, color="#9b59b6", edgecolor="black")
    axes[1].set_title("Sequences per Subject", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Sequence Count", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, c in enumerate(subj_seq_counts):
        axes[1].text(i, c + 0.1, str(c), ha="center", fontweight="bold")

    plt.suptitle("Multi-Subject Pose Dataset: Sample and Sequence Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization -> {CHART_PATH}")
    return True


if __name__ == "__main__":
    inspect_dataset()

