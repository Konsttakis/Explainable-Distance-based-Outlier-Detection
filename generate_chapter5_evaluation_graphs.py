import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from sklearn.decomposition import PCA
import seaborn as sns
from collections import defaultdict, Counter
from itertools import groupby

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Techniques.dynamic import dynamic_kr as dynamic_baseline
from Techniques.dynamic_expl import dynamic_kr_expl
from Techniques.distanceKR import static_kr
from TSB_UAD_code.feature import Window

# Output directory (the thesis includes the figures from here)
OUTPUT_DIRS = [os.path.join(BASE_DIR, 'data', 'graphs')]
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 14

# Unified Color Palette for Semantic Explanations
LABEL_COLORS = {
    'Isolated': '#E53E3E',            # Strong Red
    'Near cluster': '#3182CE',        # Classic Blue
    'Sparse microcluster': '#DD6B20', # Amber Orange
    'Sparse cluster': '#319795',      # Teal
    'Dense microcluster': '#805AD5',  # Purple
    'Dense cluster': '#38A169',       # Forest Green
    'Unclassified Outlier': '#718096',# Slate Gray
    'Inlier': '#CBD5E0'               # Light Muted Gray
}

LABEL_MARKERS = {
    'Isolated': 'o',
    'Near cluster': '^',
    'Sparse microcluster': 's',
    'Sparse cluster': 'D',
    'Dense microcluster': 'P',
    'Dense cluster': '*',
    'Unclassified Outlier': 'X',
    'Inlier': '.'
}

# The six semantic classes, ordered from the least to the most strict (priority scale of Chapter 3)
SEMANTIC_CLASSES = ['Isolated', 'Near cluster', 'Sparse microcluster', 'Dense microcluster', 'Sparse cluster', 'Dense cluster']

SEQ_LEN = 10          # subsequence length l
WINDOW, SLIDE = 200, 100

YAHOO_DIR = os.path.join(BASE_DIR, 'data', 'YAHOO')
NAB_DIR = os.path.join(BASE_DIR, 'data', 'NAB_formatted')
NAB_ARCHIVE = os.path.join(BASE_DIR, 'data', 'NAB_Archive')


def yahoo(family, i):
    names = {'A1': f'Yahoo_A1real_{i}_data.out', 'A2': f'Yahoo_A2synthetic_{i}_data.out',
             'A3': f'YahooA3Benchmark-TS{i}_data.out', 'A4': f'YahooA4Benchmark-TS{i}_data.out'}
    return os.path.join(YAHOO_DIR, names[family])


def nab(name):
    return os.path.join(NAB_DIR, name + '.csv')


MACHINE_TEMP = nab('realKnownCause_machine_temperature_system_failure')
CPU_ASG = nab('realKnownCause_cpu_utilization_asg_misconfiguration')

# Composite suite shared by Figures 5.5, 5.9 and 5.10
COMPOSITE_SUITE = [yahoo('A1', 10), yahoo('A2', 14), yahoo('A3', 10), MACHINE_TEMP, CPU_ASG]


def save_fig(fig, filename):
    for d in OUTPUT_DIRS:
        png_p = os.path.join(d, f"{filename}.png")
        fig.savefig(png_p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}.png")

def add_compact_badge(fig, dataset_str, param_str="w=200, s=100, policy='or', l=10", y_pos=0.015):
    text = f"Dataset: {dataset_str}   |   Params: {param_str}"
    fig.text(0.5, y_pos, text, ha='center', fontsize=8.8, fontstyle='italic', color='#2D3748',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDF2F7', edgecolor='#CBD5E0', linewidth=0.7))

def load_dataset(filepath, sequence_len=SEQ_LEN):
    """Returns the raw series, its subsequence embedding and the raw point labels.
    X_data[j] holds raw points j..j+l-1, i.e. its newest point has raw index j+l-1."""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        data = df['value'].astype(float).values
        labels = df['label'].astype(int).values if 'label' in df.columns else None
    else:
        df = pd.read_csv(filepath, header=None).dropna()
        data = df.iloc[:, 0].astype(float).values
        labels = df.iloc[:, 1].astype(int).values if df.shape[1] > 1 else None

    X_data = Window(window=sequence_len).convert(data).to_numpy()
    return data, X_data, labels

def nab_windows(filepath):
    """Official NAB anomaly windows (combined_windows.json) as raw (start, end) indices, or None for non-NAB files."""
    fname = os.path.basename(filepath)
    if not fname.startswith('real') and not fname.startswith('artificial'):
        return None
    category, rest = fname.split('_', 1)
    key = f"{category}/{rest}"
    with open(os.path.join(NAB_ARCHIVE, 'labels', 'combined_windows.json')) as f:
        windows = json.load(f)[key]
    ts = pd.to_datetime(pd.read_csv(os.path.join(NAB_ARCHIVE, 'data', category, rest))['timestamp'])
    return [(int(np.searchsorted(ts, pd.Timestamp(a))), int(np.searchsorted(ts, pd.Timestamp(b), side='right')) - 1)
            for a, b in windows]

def ground_truth_points(filepath, n_raw, labels):
    """Raw-level ground truth: official NAB windows for NAB streams, the provided point labels for Yahoo."""
    wins = nab_windows(filepath)
    if wins is None:
        return labels.copy()
    gt = np.zeros(n_raw, dtype=int)
    for a, b in wins:
        gt[a:b + 1] = 1
    return gt

def expand_labels(gt, sequence_len=SEQ_LEN):
    """Label expansion of the original ADBOD protocol [1] (ExperimentTSB.py): +-(l-1) around every labelled point."""
    out = gt.copy()
    for i in np.where(gt == 1)[0]:
        out[max(0, i - sequence_len + 1):min(len(gt), i + sequence_len)] = 1
    return out

def ad1_f1(gt_expanded, pred_sub, sequence_len=SEQ_LEN):
    """Range-based AD1 F1 exactly as in the original ADBOD evaluation [1]:
    subsequence predictions are padded back to raw resolution (centred), recall uses the existence reward (alpha=1)
    and precision the range-based overlap (alpha=0), both with flat bias and cardinality 'one' (Tatbul et al.)."""
    from prts import ts_recall, ts_precision
    pred = list(pred_sub)
    pred = np.array([pred[0]] * int(np.ceil((sequence_len - 1) / 2)) + pred + [pred[-1]] * ((sequence_len - 1) // 2))
    y_true = gt_expanded[:len(pred)]
    if y_true.sum() == 0 or pred.sum() == 0:
        return 0.0
    rec = ts_recall(y_true, pred, alpha=1, cardinality='one', bias='flat')
    prec = ts_precision(y_true, pred, alpha=0, cardinality='one', bias='flat')
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def unique_point_records(records):
    """Telemetry is logged once per window; keep one record per stream point.
    The first record is the one whose label survives the 'or' policy."""
    seen, out = set(), []
    for r in records:
        if r['idx'] not in seen:
            seen.add(r['idx'])
            out.append(r)
    return out

def to_relative(raw_idx, start_idx, sequence_len=SEQ_LEN):
    """Raw stream index -> position on the case-study time axis (newest point of subsequence start_idx is t=0)."""
    return raw_idx - (sequence_len - 1) - start_idx

def shade_nab_windows(axes, filepath, start_idx, n, alpha=0.55, label=True):
    spans = []
    for a, b in nab_windows(filepath):
        ra, rb = to_relative(a, start_idx), to_relative(b, start_idx)
        if rb < 0 or ra > n - 1:
            continue
        ra, rb = max(0, ra), min(n - 1, rb)
        spans.append((ra, rb))
        for ax in axes:
            ax.axvspan(ra, rb, facecolor='#FEFCBF', alpha=alpha, edgecolor='#D69E2E', linestyle='--', linewidth=1.5,
                       label='Official NAB Anomaly Window' if (label and len(spans) == 1) else None)
    return spans

def label_counts_by_phase(labels_e, spans):
    """Counts of each semantic class before / inside / after the (first) NAB window on the relative axis."""
    t = np.arange(len(labels_e))
    a, b = spans[0]
    res = {}
    for lab in SEMANTIC_CLASSES:
        m = labels_e == lab
        if m.sum():
            res[lab] = (int((m & (t < a)).sum()), int((m & (t >= a) & (t <= b)).sum()), int((m & (t > b)).sum()))
    return res

RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'Evaluation_Results')

def evaluate_three_way(path):
    """AD1 F1 of static DBOD, baseline DYN and explainable DYN on one complete stream (None if no ground truth)."""
    data, X_data, labels = load_dataset(path)
    gt_points = ground_truth_points(path, len(data), labels)
    if gt_points is None or gt_points.sum() == 0:
        return None
    gt = expand_labels(gt_points)
    # Static DBOD (k=15, R=0.5); R=0.5 is only meaningful on min-max normalised windows
    y_stat = static_kr(k=15, R=0.5, slide=SLIDE, window=WINDOW, window_norm=True).fit(X_data)
    y_dyn = dynamic_baseline(slide=SLIDE, window=WINDOW).fit(X_data)
    y_expl = dynamic_kr_expl(slide=SLIDE, window=WINDOW).fit(X_data)
    return {
        "N": len(X_data),
        "Static DBOD": ad1_f1(gt, y_stat),
        "Baseline DYN (Ref [1])": ad1_f1(gt, y_dyn),
        "Explainable DYN (Ours)": ad1_f1(gt, y_expl),
        "Identical predictions": bool(np.array_equal(y_dyn, y_expl)),
        "Static outlier %": 100 * y_stat.mean(),
        "DYN outlier %": 100 * y_dyn.mean()
    }

def family_of(fname):
    for key, fam in [('A1real', 'Yahoo A1'), ('A2synthetic', 'Yahoo A2'), ('A3Benchmark', 'Yahoo A3'), ('A4Benchmark', 'Yahoo A4')]:
        if key in fname:
            return fam
    return 'NAB'

def evaluate_full_suite():
    """Three-way AD1 evaluation on every Yahoo S5 (367) and NAB (58) stream; streams without ground truth are skipped."""
    paths = [os.path.join(YAHOO_DIR, f) for f in sorted(os.listdir(YAHOO_DIR)) if f.endswith('_data.out')] + \
            [os.path.join(NAB_DIR, f) for f in sorted(os.listdir(NAB_DIR)) if f.endswith('.csv')]
    rows, skipped = [], []
    for p in paths:
        res = evaluate_three_way(p)
        if res is None:
            skipped.append(os.path.basename(p))
            continue
        rows.append({"file": os.path.basename(p), "family": family_of(os.path.basename(p)), **res})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'full_367_yahoo_nab_evaluation.csv'), index=False)
    print(f"Full suite: {len(paths)} streams, {len(df)} evaluated, {len(skipped)} skipped (no ground-truth anomalies): {skipped}")
    return df

def generate_figure_5_1_detection_fidelity():
    print("\n--- Generating Figure 5.1: Detection Fidelity (AD1 Comparison) ---")
    methods = ["Static DBOD", "Baseline DYN (Ref [1])", "Explainable DYN (Ours)"]

    datasets = [
        ("Yahoo A1Real-10", yahoo('A1', 10)),
        ("Yahoo A2Synth-14", yahoo('A2', 14)),
        ("Yahoo A3Synth-10", yahoo('A3', 10)),
        ("Yahoo A4Synth-10", yahoo('A4', 10)),
        ("NAB IoT Temp", MACHINE_TEMP),
        ("NAB AdExchange", nab('realAdExchange_exchange-3_cpc_results')),
        ("NAB Traffic", nab('realTraffic_speed_7578')),
        ("NAB CloudWatch", nab('realAWSCloudwatch_ec2_cpu_utilization_825cc2'))
    ]
    df = pd.DataFrame([{"Dataset": name, **evaluate_three_way(path)} for name, path in datasets])
    print("F1 Results Table (representative streams):")
    print(df.to_string(index=False))
    print("Representative mean F1: " + ", ".join(f"{m}={df[m].mean():.3f}" for m in methods))

    df_all = evaluate_full_suite()
    fam_order = ['Yahoo A1', 'Yahoo A2', 'Yahoo A3', 'Yahoo A4', 'NAB']
    fam_mean = df_all.groupby('family')[methods].mean().reindex(fam_order)
    fam_mean.loc['All'] = df_all[methods].mean()
    fam_n = df_all.groupby('family').size().reindex(fam_order).tolist() + [len(df_all)]
    print("Full-suite mean F1 per family:")
    print(fam_mean.round(3).to_string())
    print("Full-suite median F1: " + ", ".join(f"{m}={df_all[m].median():.3f}" for m in methods))
    yahoo_rows = df_all[df_all['family'] != 'NAB']
    print("Yahoo-only mean F1: " + ", ".join(f"{m}={yahoo_rows[m].mean():.3f}" for m in methods))
    print(f"DYN > static on {(df_all[methods[1]] > df_all[methods[0]]).sum()} / {len(df_all)} streams, "
          f"ties {(df_all[methods[1]] == df_all[methods[0]]).sum()}; "
          f"identical baseline/explainable predictions on all streams = {df_all['Identical predictions'].all()}; "
          f"max |delta F1| = {(df_all[methods[1]] - df_all[methods[2]]).abs().max():.4f}")

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(18, 6.6), gridspec_kw={'width_ratios': [1.45, 1.0]})
    colors = ["#A0AEC0", "#3182CE", "#38A169"]
    legend_names = ["Static DBOD ($k=15, R=0.5$, normalised windows)", "Baseline DYN (dynamic.py) [1]", "Explainable DYN (dynamic_expl.py) [Ours]"]
    width = 0.26

    x = np.arange(len(df))
    for j, (m, c, lab) in enumerate(zip(methods, colors, legend_names)):
        ax.bar(x + (j - 1) * width, df[m], width, label=lab, color=c, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Range-based $AD1$ $F_1$-score", fontweight='bold')
    ax.set_title("(a) Representative Streams (one per benchmark family)", fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"], rotation=20, ha='right', fontweight='bold')
    ax.set_ylim(0, 1.32)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')
    ax.text(0.02, 0.97, f"Representative streams (n={len(df)}):\n- Static Mean F1: {df[methods[0]].mean():.2f}\n"
                        f"- Baseline DYN: {df[methods[1]].mean():.2f}\n- Explainable DYN: {df[methods[2]].mean():.2f}\n"
                        f"- Identical binary decisions on all streams",
            transform=ax.transAxes, verticalalignment='top', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#F0FFF4', edgecolor='#9AE6B4', alpha=0.95))

    xb = np.arange(len(fam_mean))
    for j, (m, c) in enumerate(zip(methods, colors)):
        bars = axb.bar(xb + (j - 1) * width, fam_mean[m], width, color=c, edgecolor="black", linewidth=0.7)
        if j < 2:   # the explainable bars are identical to the baseline ones
            for b in bars:
                axb.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=8)
    axb.set_xticks(xb)
    axb.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(fam_mean.index, fam_n)], fontweight='bold')
    axb.set_ylabel("Mean Range-based $AD1$ $F_1$-score", fontweight='bold')
    axb.set_title(f"(b) Complete Benchmark Suite ({len(df_all)} streams with labelled anomalies)", fontweight='bold', pad=10)
    axb.set_ylim(0, 1.0)
    axb.grid(True, axis='y', linestyle='--', alpha=0.35)

    fig.suptitle("Detection Accuracy Invariance: Zero-Penalty Verification\n"
                 "Baseline DYN [1] vs. Explainable DYN [Ours] across Time Series Benchmarks", fontweight='bold', fontsize=13, y=0.99)
    add_compact_badge(fig, "Yahoo S5 (A1-A4) & NAB, complete streams | Ground truth: Yahoo labels, official NAB anomaly windows", y_pos=0.015)
    plt.subplots_adjust(bottom=0.22, top=0.84, wspace=0.18)
    save_fig(fig, "Figure_5_1_Detection_Fidelity_Comparison")

def generate_figure_5_2_computational_overhead(n_repeats=5):
    print("\n--- Generating Figure 5.2: Computational Efficiency & Overhead (3-Way Comparison) ---")
    from Techniques.dynamic_expl3 import dynamic_kr as dynamic_naive
    datasets = [
        ("Yahoo A1Real", yahoo('A1', 10)),
        ("Yahoo A2Synth", yahoo('A2', 14)),
        ("Yahoo A3Synth", yahoo('A3', 10)),
        ("Yahoo A4Synth", yahoo('A4', 10)),
        ("NAB IoT Temp", MACHINE_TEMP),
        ("NAB CPU ASG", CPU_ASG)
    ]

    def median_runtime(cls, X):
        times = []
        for _ in range(n_repeats):
            clf = cls(slide=SLIDE, window=WINDOW)
            t0 = time.perf_counter()
            clf.fit(X)
            times.append(time.perf_counter() - t0)
        return float(np.median(times)), clf

    timing_res = []
    for name, path in datasets:
        _, X_data, _ = load_dataset(path)
        X_sub = X_data[:2000]

        t_base, _ = median_runtime(dynamic_baseline, X_sub)
        t_naive, clf_n = median_runtime(dynamic_naive, X_sub)
        t_expl, clf_e = median_runtime(dynamic_kr_expl, X_sub)

        timing_res.append({
            "Dataset": name,
            "N": len(X_sub),
            "Baseline (dynamic.py) [1]": t_base,
            "Naive Expl (dynamic_expl3.py) [4]": t_naive,
            "Integrated Expl (dynamic_expl.py) [Ours]": t_expl,
            "Speedup": t_naive / t_expl,
            "Integrated vs Baseline (%)": 100 * (t_expl / t_base - 1),
            "Identical labels": clf_n.outlier_labels == clf_e.outlier_labels
        })

    df_t = pd.DataFrame(timing_res)
    print(f"3-Way Timing Results (median of {n_repeats} runs):")
    print(df_t.to_string(index=False))
    print(f"Mean speedup={df_t['Speedup'].mean():.2f}x (min {df_t['Speedup'].min():.2f}, max {df_t['Speedup'].max():.2f}); "
          f"mean integrated-vs-baseline runtime change={df_t['Integrated vs Baseline (%)'].mean():.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0), gridspec_kw={'width_ratios': [1.35, 1.0]})

    x = np.arange(len(df_t))
    width = 0.26
    ax1.bar(x - width, df_t["Baseline (dynamic.py) [1]"], width, label="Baseline DYN (dynamic.py) [1]", color="#718096", edgecolor="black", linewidth=0.7)
    ax1.bar(x, df_t["Naive Expl (dynamic_expl3.py) [4]"], width, label="Naive Expl (dynamic_expl3.py) [4]", color="#E53E3E", edgecolor="black", linewidth=0.7)
    ax1.bar(x + width, df_t["Integrated Expl (dynamic_expl.py) [Ours]"], width, label="Integrated Expl (dynamic_expl.py) [Ours]", color="#38A169", edgecolor="black", linewidth=0.7)
    ax1.set_ylabel("Execution Time (Seconds, median of %d runs)" % n_repeats, fontweight='bold')
    ax1.set_title("(a) Computational Runtime Comparison", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_t["Dataset"], rotation=20, ha='right', fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax1.set_ylim(0, df_t[["Baseline (dynamic.py) [1]", "Naive Expl (dynamic_expl3.py) [4]"]].values.max() * 1.35)
    ax1.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    # Speedup Bar Plot
    bars = ax2.bar(df_t["Dataset"], df_t["Speedup"], color="#3182CE", edgecolor="black", linewidth=0.7, width=0.48)
    ax2.set_ylabel("Speedup Factor $S = T_{naive} / T_{integrated}$", fontweight='bold')
    ax2.set_title("(b) Computational Speedup from Integration", fontweight='bold')
    ax2.set_xticks(range(len(df_t)))
    ax2.set_xticklabels(df_t["Dataset"], rotation=20, ha='right', fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax2.set_ylim(0, max(df_t["Speedup"]) * 1.35)

    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.05, f"{h:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=9.5)

    mean_sp = df_t["Speedup"].mean()
    ax2.axhline(mean_sp, color="#C53030", linestyle="--", linewidth=1.5, label=f"Mean Speedup: {mean_sp:.2f}x")
    ax2.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    fig.suptitle("Computational Optimization: Eliminating Redundant Distance Calculations ($O(N^2)$)\n"
                 "Baseline DYN [1] vs. Naive Integration [4] vs. Integrated Architecture [Ours]",
                 fontweight='bold', fontsize=13, y=0.98)
    add_compact_badge(fig, "First min(2,000, N) subsequences per stream (Yahoo N = 1,411-1,670; NAB N = 2,000)", y_pos=0.015)
    plt.subplots_adjust(bottom=0.22, top=0.83, wspace=0.26)
    save_fig(fig, "Figure_5_2_Computational_Overhead")

def draw_sankey_band(ax, x0, x1, y0_bottom, y0_top, y1_bottom, y1_top, color, alpha=0.55):
    verts = [
        (x0, y0_bottom),
        (x0 + (x1 - x0) * 0.5, y0_bottom),
        (x0 + (x1 - x0) * 0.5, y1_bottom),
        (x1, y1_bottom),
        (x1, y1_top),
        (x0 + (x1 - x0) * 0.5, y1_top),
        (x0 + (x1 - x0) * 0.5, y0_top),
        (x0, y0_top),
        (x0, y0_bottom)
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
    ax.add_patch(patch)

SANKEY_SUITE = (
    [yahoo('A1', i) for i in range(1, 5)] + [yahoo('A2', i) for i in range(1, 5)] +
    [yahoo('A3', i) for i in range(1, 5)] + [yahoo('A4', i) for i in range(1, 5)] +
    [MACHINE_TEMP, nab('realKnownCause_ambient_temperature_system_failure'), CPU_ASG] +
    [nab('realTraffic_occupancy_6005'), nab('realTraffic_speed_6005'), nab('realTraffic_TravelTime_387')]
)

def generate_figure_5_3_sankey_and_entropy():
    print("\n--- Generating Figure 5.3: Black-Box Deconstruction & Shannon Entropy ---")
    total_pts = 0
    total_inliers = 0
    total_outliers = 0
    class_counts = defaultdict(int)

    for f in SANKEY_SUITE:
        _, X_data, _ = load_dataset(f)
        clf = dynamic_kr_expl(slide=SLIDE, window=WINDOW)
        clf.fit(X_data)
        for lab in clf.outlier_labels:
            total_pts += 1
            if lab == "Inlier":
                total_inliers += 1
            else:
                total_outliers += 1
                class_counts[lab] += 1

    sorted_labels = [l for l in SEMANTIC_CLASSES if class_counts[l] > 0]

    # Shannon Entropy on the outlier set. Baseline: one state ("1") -> H = 0
    h_base = 0.0
    total_active = sum(class_counts[l] for l in sorted_labels)
    probs = [class_counts[l] / total_active for l in sorted_labels]
    h_expl = -sum(p * np.log2(p) for p in probs if p > 0)

    print(f"Streams={len(SANKEY_SUITE)}, total N={total_pts:,}, inliers={100 * total_inliers / total_pts:.1f}%, "
          f"outliers={100 * total_outliers / total_pts:.1f}% ({total_outliers:,} pts), unclassified={class_counts['Unclassified Outlier']}")
    for l in SEMANTIC_CLASSES:
        print(f"   {l:22s} {class_counts[l]:6d}  {100 * class_counts[l] / total_active:5.1f}%")
    print(f"Shannon Entropy on Outlier Set: Baseline = {h_base:.2f} bits, Explainable = {h_expl:.2f} bits (Information Gain: +{h_expl:.2f} bits)")

    fig = plt.figure(figsize=(16, 8.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.45, 2.65)
    ax.set_ylim(-8, 142)
    ax.axis('off')

    w1, w2 = 0.24, 0.16
    inlier_pct = (total_inliers / total_pts) * 100
    outlier_pct = (total_outliers / total_pts) * 100

    # Stage 1: Baseline Decision (bar heights are schematic; the percentages are the measured values)
    x1 = 0.4
    inlier_h = 42.0
    ax.bar(x1, inlier_h, width=w1, bottom=5.0, color='#A0AEC0', edgecolor='black', linewidth=1.1)
    ax.text(x1, 5.0 + inlier_h/2, "y = 0", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    ax.text(x1 - w1/2 - 0.04, 5.0 + inlier_h/2, f"Normal Stream:\n{inlier_pct:.1f}%\n({total_inliers:,} pts)",
            ha='right', va='center', fontweight='bold', fontsize=9.5, color='#4A5568')

    outlier_h = 36.0
    outlier_y0 = 58.0
    ax.bar(x1, outlier_h, width=w1, bottom=outlier_y0, color='#E53E3E', edgecolor='black', linewidth=1.1)
    ax.text(x1, outlier_y0 + outlier_h/2, "y = 1", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    ax.text(x1 - w1/2 - 0.04, outlier_y0 + outlier_h/2, f"Baseline Outliers:\n{outlier_pct:.1f}%\n({total_outliers:,} pts)",
            ha='right', va='center', fontweight='bold', fontsize=9.5, color='#C53030')
    ax.text(x1, 0.5, "(bar heights not to scale)", ha='center', fontsize=8, fontstyle='italic', color='#718096')

    # Stage 2: Semantic Labels
    x2 = 1.95
    curr_y1 = outlier_y0
    curr_y2 = 8.0
    total_disp_h = 75.0
    last_text_y = -999.0

    for lab in sorted_labels:
        cnt = class_counts[lab]
        pct = (cnt / total_active) * 100
        h_s1 = (cnt / total_active) * outlier_h
        h_s2 = max(2.2, (cnt / total_active) * total_disp_h)
        col = LABEL_COLORS.get(lab, '#4A5568')

        draw_sankey_band(ax, x1 + w1/2, x2 - w2/2, curr_y1, curr_y1 + h_s1, curr_y2, curr_y2 + h_s2, col, alpha=0.55)
        ax.bar(x2, h_s2, width=w2, bottom=curr_y2, color=col, edgecolor='black', linewidth=0.9)

        target_y = curr_y2 + h_s2 / 2
        text_y = max(target_y, last_text_y + 6.8)
        last_text_y = text_y

        if abs(text_y - target_y) > 1.0:
            ax.plot([x2 + w2/2 + 0.01, x2 + w2/2 + 0.035], [target_y, text_y], color='#718096', linewidth=0.8, linestyle=':')

        ax.text(x2 + w2/2 + 0.04, text_y, f"{lab}\n{pct:.1f}% ({cnt:,} pts)",
                ha='left', va='center', fontweight='bold', fontsize=9, color='#1A202C')

        curr_y1 += h_s1
        curr_y2 += h_s2 + 2.5

    ax.text(x1, 104, "STAGE 1: BASELINE DYN [1]\n(Monolithic Opaque Outlier)", ha='center', fontweight='bold', fontsize=11, color='#2D3748')
    ax.text(x2 + 0.08, 104, "STAGE 2: EXPLAINABLE DYN [Ours]\n(Decomposed Structural Semantics)", ha='center', fontweight='bold', fontsize=11, color='#2D3748')

    entropy_box = (
        "QUANTITATIVE INFORMATION GAIN (SHANNON ENTROPY):\n"
        f"• Baseline Outlier Set Entropy: H_base = -1.0 * log2(1.0) = {h_base:.2f} bits (single state, flat binary mask)\n"
        f"• Explainable Semantic Entropy:  H_expl = -Σ p_c * log2(p_c) = {h_expl:.2f} bits ({len(sorted_labels)} observed classes)\n"
        f"• Net Information Gain Provided: ΔH = +{h_expl:.2f} bits per detected anomaly (Theoretical Max = log2(6) = 2.58 bits)"
    )
    ax.text(1.15, 134, entropy_box, ha='center', va='top', fontsize=9.2,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEFCBF', edgecolor='#D69E2E', alpha=0.95))

    fig.suptitle("Deconstructing the Black-Box: Semantic Flow and Information Entropy Gain\n"
                 "From Homogeneous Binary Flags (dynamic.py) to Fine-Grained Topological Structures (dynamic_expl.py)",
                 fontweight='bold', fontsize=13, y=0.98)

    add_compact_badge(fig, f"{len(SANKEY_SUITE)} complete streams (Yahoo A1-A4: 4 each; NAB: 3 known-cause, 3 traffic) | Total N = {total_pts:,} pts")
    plt.subplots_adjust(bottom=0.08, top=0.88)
    save_fig(fig, "Figure_5_3_BlackBox_Sankey_Decomposition")

DOMAIN_GROUPS = {
    'Yahoo A1 (Real Traffic)': [yahoo('A1', i) for i in range(1, 7)],
    'Yahoo A2 (Synthetic Spikes)': [yahoo('A2', i) for i in range(1, 7)],
    'Yahoo A3 (Synthetic Outliers)': [yahoo('A3', i) for i in range(1, 7)],
    'Yahoo A4 (Outliers + Change-points)': [yahoo('A4', i) for i in range(1, 7)],
    'NAB Temperature (Machine / Ambient)': [MACHINE_TEMP, nab('realKnownCause_ambient_temperature_system_failure')],
    'NAB Cloud (CloudWatch / EC2)': [CPU_ASG, nab('realKnownCause_ec2_request_latency_system_failure'),
                                     nab('realAWSCloudwatch_ec2_cpu_utilization_825cc2')],
    'NAB Road Traffic / Ads': [nab('realTraffic_occupancy_6005'), nab('realTraffic_speed_6005'), nab('realTraffic_TravelTime_387'),
                                nab('realAdExchange_exchange-2_cpc_results'), nab('realAdExchange_exchange-3_cpc_results'),
                                nab('realAdExchange_exchange-4_cpc_results')]
}

def generate_figure_5_4_benchmark_domain_profiles():
    print("\n--- Generating Figure 5.4: Benchmark Domain Semantic Profiles ---")
    domain_data = defaultdict(lambda: defaultdict(int))
    for group_name, files in DOMAIN_GROUPS.items():
        for f in files:
            _, X_data, _ = load_dataset(f)
            clf = dynamic_kr_expl(slide=SLIDE, window=WINDOW)
            clf.fit(X_data)
            for lab in clf.outlier_labels:
                if lab != "Inlier" and lab != "Unclassified Outlier":
                    domain_data[group_name][lab] += 1

    df_dom = pd.DataFrame(domain_data).reindex(SEMANTIC_CLASSES).fillna(0)
    df_pct = df_dom.div(df_dom.sum(axis=0), axis=1) * 100
    n_streams = sum(len(v) for v in DOMAIN_GROUPS.values())
    print(f"Streams={n_streams}; outliers per group:", df_dom.sum(axis=0).astype(int).to_dict())
    print("Semantic share (%) per group:")
    print(df_pct.round(1).to_string())

    fig, ax = plt.subplots(figsize=(13, 7.4))
    y_pos = np.arange(len(df_pct.columns))
    left_accum = np.zeros(len(df_pct.columns))
    bar_h = 0.52

    for lab in SEMANTIC_CLASSES:
        vals = df_pct.loc[lab].values
        if vals.sum() == 0:
            continue
        col = LABEL_COLORS.get(lab, '#718096')
        ax.barh(y_pos, vals, left=left_accum, height=bar_h, color=col, edgecolor='white', label=lab)
        for i, (v, l) in enumerate(zip(vals, left_accum)):
            if v > 7.5:
                ax.text(l + v/2, y_pos[i], f"{v:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        left_accum += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{g}\n({len(DOMAIN_GROUPS[g])} streams, n={int(df_dom[g].sum()):,})" for g in df_pct.columns],
                       fontweight='bold', fontsize=9.5)
    ax.set_xlabel("Semantic Share of Detected Outliers (%)", fontweight='bold', labelpad=10)
    ax.set_title("Benchmark Semantic Fingerprints per Application Domain\n"
                 "Share of the six structural classes among the detected outliers", fontweight='bold', pad=12)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=6, framealpha=0.95, facecolor='white', edgecolor='#CBD5E0', title="Semantic Class")

    add_compact_badge(fig, f"Yahoo S5 (A1-A4, series 1-6) & NAB (Temperature, Cloud, Traffic/Ads) | {n_streams} complete streams", y_pos=0.018)
    plt.subplots_adjust(bottom=0.26, top=0.89, left=0.22)
    save_fig(fig, "Figure_5_4_Benchmark_Domain_Profiles")

def run_composite_suite():
    """Runs the explainable detector on the composite suite once and returns per-point telemetry and label series."""
    records, label_series = [], []
    for f in COMPOSITE_SUITE:
        _, X_data, _ = load_dataset(f)
        clf = dynamic_kr_expl(slide=SLIDE, window=WINDOW)
        clf.fit(X_data)
        profiles = {p['idx']: p for p in unique_point_records(clf.density_profiles)}
        for t in unique_point_records(clf.label_telemetry):
            # Sanity check: the kept record carries the label that survived the 'or' policy
            assert t['final_label'] == clf.outlier_labels[t['idx']]
            records.append({**t, 'counts': profiles[t['idx']]['counts'], 'stream': os.path.basename(f)})
        label_series.append(clf.outlier_labels)
    return pd.DataFrame(records), label_series

def generate_figure_5_5_density_fingerprints(df_tel):
    print("\n--- Generating Figure 5.5: Multi-Scale Density Profiles vs Single-Radius Cutoff ---")
    radii_ticks = ["R/4", "R/2", "R (Cutoff)", "2R", "4R"]
    sorted_labels = [l for l in SEMANTIC_CLASSES if (df_tel['final_label'] == l).any()]
    mean_counts = {l: np.mean(np.stack(df_tel.loc[df_tel['final_label'] == l, 'counts'].values), axis=0) for l in sorted_labels}
    print(f"Unique outlier points: {len(df_tel)}")
    for l in sorted_labels:
        print(f"   {l:22s} n={(df_tel['final_label'] == l).sum():5d} mean counts [R/4,R/2,R,2R,4R] = {np.round(mean_counts[l], 1)}"
              f" | mean k = {df_tel.loc[df_tel['final_label'] == l, 'k'].mean():.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    for lab in sorted_labels:
        n = (df_tel['final_label'] == lab).sum()
        ax1.plot(radii_ticks, mean_counts[lab], marker=LABEL_MARKERS.get(lab, 'o'), linewidth=2.3, markersize=7.5,
                 label=f"{lab} (n={n})", color=LABEL_COLORS.get(lab, '#000000'))

    ax1.axvline(x=2, color='#E53E3E', linestyle='--', linewidth=2.2, label='Baseline Cutoff (dynamic.py single R)')
    ax1.set_title("(a) Radial Neighbor Growth Curves", fontweight='bold')
    ax1.set_xlabel("Probed Neighborhood Radius Ring", fontweight='bold')
    ax1.set_ylabel("Mean Neighbor Cardinality", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    vals_2r = [mean_counts[l][3] for l in sorted_labels]
    vals_4r = [mean_counts[l][4] for l in sorted_labels]
    y = np.arange(len(sorted_labels))
    h = 0.35
    ax2.barh(y - h/2, vals_2r, h, label="Cardinality at 2R", color="#4299E1", edgecolor="black", linewidth=0.7)
    ax2.barh(y + h/2, vals_4r, h, label="Cardinality at 4R", color="#805AD5", edgecolor="black", linewidth=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sorted_labels, fontweight='bold')
    ax2.set_xlabel("Average Neighbor Count", fontweight='bold')
    ax2.set_title("(b) Neighborhood Expansion Profile (Macro-Topology)", fontweight='bold')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax2.legend(loc='lower right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    fig.suptitle("Multi-Scale Probing vs. Single-Radius Cutoff: Revealing Radial Density Gradients\n"
                 "The Shape of the Neighbor Growth Curve Defines the Geometric Fingerprint of the Outlier",
                 fontweight='bold', fontsize=13, y=0.98)
    add_compact_badge(fig, "Composite Suite: Yahoo A1-10/A2-14/A3-10 & NAB machine temp./CPU ASG (complete streams, one entry per outlier)", y_pos=0.015)
    plt.subplots_adjust(bottom=0.18, top=0.83, wspace=0.28)
    save_fig(fig, "Figure_5_5_Density_Gradients_Profile")

def plot_case_study(target, start_idx, end_idx, unit_label, ylim, filename, top_title, bot_title, badge):
    data, X_data, labels = load_dataset(target)
    end_idx = min(end_idx, len(X_data))
    X_sub = X_data[start_idx:end_idx]
    data_sub = data[SEQ_LEN - 1 + start_idx: SEQ_LEN - 1 + end_idx]
    labels_sub = labels[SEQ_LEN - 1 + start_idx: SEQ_LEN - 1 + end_idx] if labels is not None else None
    time_axis = np.arange(len(data_sub))

    y_b = dynamic_baseline(slide=SLIDE, window=WINDOW).fit(X_sub)
    clf_e = dynamic_kr_expl(slide=SLIDE, window=WINDOW)
    y_e = clf_e.fit(X_sub)
    labels_e = np.array(clf_e.outlier_labels)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(15, 8.8), sharex=True, gridspec_kw={'hspace': 0.32})
    spans = shade_nab_windows([ax_top, ax_bot], target, start_idx, len(data_sub))

    print(f"Raw-stream range [{SEQ_LEN - 1 + start_idx}, {SEQ_LEN - 2 + end_idx}] | NAB windows on relative axis: {spans} "
          f"| baseline outliers={int(y_b.sum())}, explainable outliers={int(y_e.sum())}, identical={np.array_equal(y_b, y_e)}")
    t = np.arange(len(y_b))
    a, b = spans[0]
    print(f"   Baseline outliers before/inside/after window: {int(y_b[t < a].sum())}/{int(y_b[(t >= a) & (t <= b)].sum())}/{int(y_b[t > b].sum())}")
    for lab, (bef, ins, aft) in label_counts_by_phase(labels_e, spans).items():
        print(f"   {lab:22s} before/inside/after window: {bef}/{ins}/{aft}")

    if labels_sub is not None and np.sum(labels_sub) > 0:
        true_pts = np.where(labels_sub == 1)[0]
        for ax in (ax_top, ax_bot):
            ax.scatter(time_axis[true_pts], data_sub[true_pts], marker='*', s=180, color='#D97706', edgecolors='black', linewidth=0.9,
                       label=f'NAB Labelled Anomaly Point [n={len(true_pts)}]', zorder=7)

    # Top: Baseline
    ax_top.plot(time_axis, data_sub, color='#4A5568', alpha=0.55, linewidth=1.1, label=unit_label)
    out_b = np.where(y_b == 1)[0]
    ax_top.scatter(time_axis[out_b], data_sub[out_b], color='#E53E3E', s=55, marker='o', edgecolors='black', linewidth=0.7,
                   label=f"Baseline Outliers (dynamic.py y=1) [n={len(out_b)}]", zorder=5)
    ax_top.set_title(top_title, fontweight='bold', fontsize=11, pad=8)
    ax_top.set_ylabel(unit_label, fontweight='bold')
    ax_top.set_ylim(*ylim)
    ax_top.grid(True, linestyle='--', alpha=0.35)
    ax_top.legend(loc='upper left', ncol=2, framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    # Bottom: Explainable
    ax_bot.plot(time_axis, data_sub, color='#718096', alpha=0.45, linewidth=1.1, label=unit_label)
    for lab in [l for l in LABEL_COLORS if l != 'Inlier' and (labels_e == l).sum() > 0]:
        mask = (labels_e == lab)
        ax_bot.scatter(time_axis[mask], data_sub[mask], color=LABEL_COLORS[lab], marker=LABEL_MARKERS.get(lab, 'o'),
                       s=65, edgecolors='black', linewidth=0.7, label=f"Explainable: {lab} [n={mask.sum()}]", zorder=5)
    ax_bot.set_title(bot_title, fontweight='bold', fontsize=11, pad=8)
    ax_bot.set_xlabel("Stream Timestep (Relative t in evaluation window)", fontweight='bold')
    ax_bot.set_ylabel(unit_label, fontweight='bold')
    ax_bot.set_ylim(*ylim)
    ax_bot.grid(True, linestyle='--', alpha=0.35)
    ax_bot.legend(loc='upper left', ncol=3, framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    add_compact_badge(fig, badge, y_pos=0.012)
    plt.subplots_adjust(bottom=0.15, top=0.90)
    save_fig(fig, filename)

def generate_figure_5_6_case_study_nab_temp():
    print("\n--- Generating Figure 5.6: Case Study NAB Machine Temperature Failure ---")
    plot_case_study(
        MACHINE_TEMP, 1000, 3200, "Machine Temperature (°C)", (35, 148), "Figure_5_6_Case_Study_NAB_Machine_Temperature",
        "Initial Algorithm: Opaque Non-Explainable Baseline (dynamic.py)\n"
        "Monochromatic Decision: every alarm looks the same, whether an isolated jump or the failure itself",
        "Our Proposed Algorithm: Semantic Explainability (dynamic_expl.py)\n"
        "Same alarms, each annotated with the structure of its neighbourhood",
        "NAB: machine_temperature_system_failure.csv (Industrial IoT | Subsequences t=[1000..3200])")

def generate_figure_5_7_case_study_aws_cpu():
    print("\n--- Generating Figure 5.7: Case Study AWS ASG Misconfiguration ---")
    plot_case_study(
        CPU_ASG, 15800, 18000, "CPU Utilization (%)", (-5, 148), "Figure_5_7_Case_Study_AWS_CPU_ASG",
        "Initial Algorithm: Opaque Non-Explainable Baseline (dynamic.py)\n"
        "Monochromatic Decision: Flat binary label cannot diagnose the structural cause of the surge",
        "Our Proposed Algorithm: Semantic Explainability (dynamic_expl.py)\n"
        "Same alarms, each annotated with the structure of its neighbourhood",
        "NAB: cpu_utilization_asg_misconfiguration.csv (AWS CloudWatch | Subsequences t=[15800..18000])")

def generate_figure_5_8_pca_topological_projection():
    print("\n--- Generating Figure 5.8: PCA 2D Topological Projection ---")
    _, X_data, _ = load_dataset(MACHINE_TEMP)
    X_sub = X_data[1000:3200]

    y_b = dynamic_baseline(slide=SLIDE, window=WINDOW).fit(X_sub)
    clf_e = dynamic_kr_expl(slide=SLIDE, window=WINDOW)
    clf_e.fit(X_sub)
    labels_e = np.array(clf_e.outlier_labels)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sub)
    var = pca.explained_variance_ratio_ * 100
    print(f"PCA explained variance: PC1={var[0]:.1f}%, PC2={var[1]:.1f}% (total {var[:2].sum():.1f}%)")
    inl = X_pca[labels_e == 'Inlier']
    for lab in SEMANTIC_CLASSES:
        m = labels_e == lab
        if m.sum():
            dist = np.min(np.linalg.norm(X_pca[m][:, None, :] - inl[None, :, :], axis=2), axis=1)
            print(f"   {lab:22s} n={m.sum():3d} centroid=({X_pca[m, 0].mean():7.1f},{X_pca[m, 1].mean():6.1f}) "
                  f"median distance to nearest inlier in PCA plane={np.median(dist):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True, sharey=True)

    ax1.scatter(X_pca[y_b == 0, 0], X_pca[y_b == 0, 1], color='#CBD5E0', s=20, alpha=0.4, label='Inliers (y=0)')
    ax1.scatter(X_pca[y_b == 1, 0], X_pca[y_b == 1, 1], color='#E53E3E', s=60, marker='o', edgecolors='black', linewidth=0.7,
                label=f"Baseline Outlier (y=1) [n={(y_b==1).sum()}]", zorder=5)
    ax1.set_title("Initial Algorithm: Opaque Baseline (dynamic.py)\n"
                  "Binary collapse blind to spatial manifold geometry", fontweight='bold', fontsize=11)
    ax1.set_xlabel(f"PC1 ({var[0]:.1f}% Variance)", fontweight='bold')
    ax1.set_ylabel(f"PC2 ({var[1]:.1f}% Variance)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    ax2.scatter(X_pca[labels_e == 'Inlier', 0], X_pca[labels_e == 'Inlier', 1], color='#E2E8F0', s=20, alpha=0.4, label='Inliers')
    for lab in [l for l in LABEL_COLORS if l != 'Inlier' and (labels_e == l).sum() > 0]:
        mask = (labels_e == lab)
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], color=LABEL_COLORS[lab], marker=LABEL_MARKERS.get(lab, 'o'),
                    s=70, edgecolors='black', linewidth=0.8, label=f"{lab} ({mask.sum()})", zorder=5)

    ax2.set_title("Our Proposed Algorithm: Semantic Explainability (dynamic_expl.py)\n"
                  "The same outliers, separated by neighbourhood structure", fontweight='bold', fontsize=11)
    ax2.set_xlabel(f"PC1 ({var[0]:.1f}% Variance)", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.35)
    ax2.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0', title="Semantic Class")

    fig.suptitle("Latent Space Topological Projection: Black-Box Binary View vs. Decomposed Geometry",
                 fontweight='bold', fontsize=13, y=0.98)
    add_compact_badge(fig, "NAB machine_temperature_system_failure (Subsequences t=[1000..3200] | Dim l=10 -> PCA 2D)")
    plt.subplots_adjust(bottom=0.16, top=0.88, wspace=0.18)
    save_fig(fig, "Figure_5_8_PCA_Topology_Projection")

def generate_figure_5_9_conflict_resolution_and_severity(df_tel):
    print("\n--- Generating Figure 5.9: Conflict Resolution & Severity Spectrum ---")
    order = SEMANTIC_CLASSES + (['Unclassified Outlier'] if (df_tel[['nn_label', 'status_label']] == 'Unclassified Outlier').any().any() else [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6))

    # Left: cross-tabulation on a shared, priority-ordered axis so that the diagonal means agreement
    crosstab = pd.crosstab(df_tel['nn_label'], df_tel['status_label']).reindex(index=order, columns=order, fill_value=0)
    agree = (df_tel['nn_label'] == df_tel['status_label']).mean() * 100
    print(f"Unique outlier points={len(df_tel)}, NN/Status agreement={agree:.1f}%")
    print(crosstab.to_string())
    final_from = np.where(df_tel['nn_label'] == df_tel['status_label'], 'agree',
                          np.where(df_tel['final_label'] == df_tel['status_label'], 'status', 'nn'))
    print("Final label taken from:", Counter(final_from))

    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Outlier Points'}, ax=ax1)
    ax1.set_title(f"(a) Cross-Tabulation of the Two Labeling Methods\nNN-based (Table 3) vs Status-based (Table 4) [4] | agreement {agree:.1f}%", fontweight='bold')
    ax1.set_xlabel("Status-based Label (Algorithm 2)", fontweight='bold')
    ax1.set_ylabel("NN-based Label (Algorithm 1)", fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha='right', fontweight='bold')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, ha='right', fontweight='bold')

    # Right: severity as relative exceedance d_k / R, which is comparable across streams of different scale
    df_v = df_tel[df_tel['final_label'] != "Unclassified Outlier"].copy()
    df_v['ratio'] = df_v['score'] / df_v['r']
    present = [l for l in SEMANTIC_CLASSES if (df_v['final_label'] == l).any()]
    print("Severity d_k/R per final class (median, IQR):")
    for l in present:
        r = df_v.loc[df_v['final_label'] == l, 'ratio']
        print(f"   {l:22s} n={len(r):5d} median={r.median():.3f} q25={r.quantile(.25):.3f} q75={r.quantile(.75):.3f}")
    sns.violinplot(data=df_v, x='final_label', y='ratio', order=present, hue='final_label', hue_order=present,
                   palette=LABEL_COLORS, legend=False, ax=ax2, cut=0)
    ax2.set_ylim(bottom=0.9)
    ax2.axhline(1.0, color='#E53E3E', linestyle='--', linewidth=1.3, label='Cutoff $d_k = R$')
    ax2.set_title("(b) Relative Severity ($d_k / R$) per Semantic Class", fontweight='bold')
    ax2.set_xlabel("Assigned Semantic Class", fontweight='bold')
    ax2.set_ylabel("Relative Exceedance $d_k / R$", fontweight='bold')
    ax2.set_xticks(range(len(present)))
    ax2.set_xticklabels(present, rotation=25, ha='right', fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax2.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    fig.suptitle("Internal Reasoning Dynamics: Algorithmic Consensus and Hierarchical Severity Spectrum",
                 fontweight='bold', fontsize=13, y=0.98)
    add_compact_badge(fig, "Composite Suite: Yahoo A1-10/A2-14/A3-10 & NAB machine temp./CPU ASG (complete streams, one entry per outlier)", y_pos=0.012)
    plt.subplots_adjust(bottom=0.26, top=0.86, wspace=0.28)
    save_fig(fig, "Figure_5_9_Conflict_Resolution_and_Severity")

def generate_figure_5_10_temporal_label_evolution(label_series):
    print("\n--- Generating Figure 5.10: Anomaly Persistence Lifespan & Empirical Transition Lifecycle ---")
    persistence_by_label = defaultdict(list)
    transitions = defaultdict(lambda: defaultdict(int))

    for labels_series in label_series:
        # 1. Persistence runs (consecutive timesteps a label survives)
        for lab, group in groupby(labels_series):
            if lab != "Inlier" and lab != "Unclassified Outlier":
                persistence_by_label[lab].append(len(list(group)))
        # 2. Transition counts (Label_t -> Label_{t+1}) within each stream
        for t in range(len(labels_series) - 1):
            transitions[labels_series[t]][labels_series[t + 1]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2))

    sorted_labels = [l for l in SEMANTIC_CLASSES if len(persistence_by_label[l]) > 0]
    mean_durations = [np.mean(persistence_by_label[l]) for l in sorted_labels]
    max_durations = [np.max(persistence_by_label[l]) for l in sorted_labels]
    colors = [LABEL_COLORS.get(l, '#718096') for l in sorted_labels]
    for l, m, mx in zip(sorted_labels, mean_durations, max_durations):
        print(f"   {l:22s} runs={len(persistence_by_label[l]):4d} mean={m:.1f} max={mx}")

    x = np.arange(len(sorted_labels))
    w = 0.35
    ax1.bar(x - w/2, mean_durations, w, label='Mean Consecutive Duration (Steps)', color=colors, edgecolor='black', linewidth=0.8)
    ax1.bar(x + w/2, max_durations, w, label='Max Consecutive Duration (Steps)', color=colors, alpha=0.45, hatch='//', edgecolor='black', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n(runs={len(persistence_by_label[l])})" for l in sorted_labels], fontweight='bold', rotation=18, ha='right', fontsize=9.5)
    ax1.set_ylabel("Consecutive Timesteps (Duration in Stream)", fontweight='bold')
    ax1.set_title("(a) Anomaly Persistence Lifespan per Class", fontweight='bold', fontsize=11, pad=10)
    if len(max_durations) > 0:
        ax1.set_ylim(0, max(max_durations) * 1.35)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    for i, (m_val, mx_val) in enumerate(zip(mean_durations, max_durations)):
        ax1.text(i - w/2, m_val + 0.2, f"{m_val:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax1.text(i + w/2, mx_val + 0.2, f"{mx_val:.0f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Right: Markov transition matrix (rows normalised over all observed next states)
    active_states = ['Inlier'] + sorted_labels
    df_trans = pd.DataFrame(0.0, index=active_states, columns=active_states)
    for s1 in active_states:
        for s2 in active_states:
            df_trans.loc[s1, s2] = transitions[s1][s2]
    row_totals = pd.Series({s: sum(transitions[s].values()) for s in active_states}).replace(0, 1)
    df_trans_prob = df_trans.div(row_totals, axis=0) * 100
    print("Transition probabilities (%):")
    print(df_trans_prob.round(1).to_string())

    sns.heatmap(df_trans_prob, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Transition Probability (%)'}, ax=ax2)
    ax2.set_title("(b) Empirical Outlier Lifecycle Transition Matrix ($S_t \\to S_{t+1}$)", fontweight='bold', fontsize=11, pad=10)
    ax2.set_xlabel("Next State ($S_{t+1}$)", fontweight='bold')
    ax2.set_ylabel("Current State ($S_t$)", fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha='right', fontweight='bold')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, ha='right', fontweight='bold')

    fig.suptitle("Outlier Lifecycle Dynamics: Temporal Persistence and State Mutation along the Stream",
                 fontweight='bold', fontsize=13, y=0.98)
    add_compact_badge(fig, "Composite Suite: Yahoo A1-10/A2-14/A3-10 & NAB machine temp./CPU ASG (complete streams)", y_pos=0.012)
    plt.subplots_adjust(bottom=0.24, top=0.88, wspace=0.38)
    save_fig(fig, "Figure_5_10_Temporal_Label_Evolution")

def generate_figure_5_11_concept_drift_adaptation():
    print("\n--- Generating Figure 5.11: Concept Drift Parameter Adaptation ---")
    raw_data, X_data, labels = load_dataset(MACHINE_TEMP)

    start_idx, end_idx = 1000, 3200
    X_sub = X_data[start_idx:end_idx]
    data_sub = raw_data[SEQ_LEN - 1 + start_idx: SEQ_LEN - 1 + end_idx]
    labels_sub = labels[SEQ_LEN - 1 + start_idx: SEQ_LEN - 1 + end_idx] if labels is not None else None
    time_axis = np.arange(len(data_sub))

    clf = dynamic_kr_expl(slide=SLIDE, window=WINDOW)
    clf.fit(X_sub)
    labels_e = np.array(clf.outlier_labels)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.8), sharex=True, gridspec_kw={'hspace': 0.32})

    spans = shade_nab_windows([ax1], MACHINE_TEMP, start_idx, len(data_sub))
    shade_nab_windows([ax2], MACHINE_TEMP, start_idx, len(data_sub), alpha=0.35, label=False)
    if labels_sub is not None and np.sum(labels_sub) > 0:
        true_pts = np.where(labels_sub == 1)[0]
        ax1.scatter(time_axis[true_pts], data_sub[true_pts], marker='*', s=180, color='#D97706', edgecolors='black', linewidth=0.9,
                    label=f'NAB Labelled Anomaly Point [n={len(true_pts)}]', zorder=7)

    ax1.plot(time_axis, data_sub, color='#718096', alpha=0.55, linewidth=1.1, label='Sensor Stream (Temp °C)')
    for lab in [l for l in LABEL_COLORS if l != 'Inlier' and (labels_e == l).sum() > 0]:
        mask = (labels_e == lab)
        ax1.scatter(time_axis[mask], data_sub[mask], color=LABEL_COLORS[lab],
                    marker=LABEL_MARKERS.get(lab, 'o'), s=65, edgecolors='black', linewidth=0.7,
                    label=f"Explainable: {lab} [n={mask.sum()}]", zorder=5)

    ax1.set_title("(a) Stream Subsequence with Regime Drift, NAB Anomaly Window & Semantic Explanations", fontweight='bold', fontsize=11, pad=8)
    ax1.set_ylabel("Sensor Value (°C)", fontweight='bold')
    ax1.set_ylim(35, 148)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', ncol=3, framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    # Each window covers subsequences [pos-w, pos); its parameters are drawn at the window centre
    n = len(X_sub)
    ends = list(range(WINDOW, n, SLIDE))
    if not ends or ends[-1] < n:
        ends.append(n)
    window_centres = np.array([e - WINDOW / 2 for e in ends])
    k_hist, r_hist = np.array(clf.k_history, dtype=float), np.array(clf.r_history, dtype=float)
    valid = k_hist > 0
    l1 = ax2.plot(window_centres[valid], k_hist[valid], color='#3182CE', marker='o', markersize=4.5, label='Self-Tuned k (Neighbors)', linewidth=2.0)
    ax2.set_xlabel("Stream Timestep (Relative t in evaluation window)", fontweight='bold')
    ax2.set_ylabel("Self-Tuned k", color='#3182CE', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#3182CE')

    ax2_twin = ax2.twinx()
    l2 = ax2_twin.plot(window_centres[valid], r_hist[valid], color='#DD6B20', marker='s', markersize=4.5, label='Self-Tuned R (Radius)', linewidth=2.0, linestyle='--')
    ax2_twin.set_ylabel("Self-Tuned R", color='#DD6B20', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='#DD6B20')
    ax2.set_title("(b) Dynamic Parameter Adaptation ($k_t, R_t$) per Sliding Window (plotted at window centre)", fontweight='bold', fontsize=11, pad=8)
    ax2.grid(True, linestyle='--', alpha=0.35)

    lines = l1 + l2
    ax2.legend(lines, [l.get_label() for l in lines], loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    a, b = spans[0]
    phases = [("before window", window_centres < a), ("inside window", (window_centres >= a) & (window_centres <= b)), ("after window", window_centres > b)]
    lines_txt = []
    for name, m in phases:
        m = m & valid
        if m.sum():
            lines_txt.append(f"• {name}: R ∈ [{r_hist[m].min():.1f}, {r_hist[m].max():.1f}], k ∈ [{int(k_hist[m].min())}, {int(k_hist[m].max())}]")
            print(f"   {name:14s} windows={m.sum():2d} R in [{r_hist[m].min():.1f}, {r_hist[m].max():.1f}] median R={np.median(r_hist[m]):.1f} "
                  f"k in [{int(k_hist[m].min())}, {int(k_hist[m].max())}]")
    print(f"   k_history={clf.k_history}\n   r_history={np.round(r_hist, 1).tolist()}")
    ax2.text(0.015, 0.95, "MEASURED PARAMETER RANGES:\n" + "\n".join(lines_txt),
             transform=ax2.transAxes, fontsize=8.8, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0FFF4', edgecolor='#9AE6B4', alpha=0.95))

    # Reference: a static detector frozen at the typical pre-failure radius
    r_static = float(np.median(r_hist[phases[0][1] & valid]))
    y_static = static_kr(k=int(np.median(k_hist[phases[0][1] & valid])), R=r_static, slide=SLIDE, window=WINDOW).fit(X_sub)
    t = np.arange(n)
    print(f"   Static detector frozen at pre-failure median (k={int(np.median(k_hist[phases[0][1] & valid]))}, R={r_static:.1f}): "
          f"outliers overall={100 * y_static.mean():.1f}%, inside window={100 * y_static[(t >= a) & (t <= b)].mean():.1f}% "
          f"vs DYN overall={100 * (labels_e != 'Inlier').mean():.1f}%, inside window={100 * (labels_e[(t >= a) & (t <= b)] != 'Inlier').mean():.1f}%")

    fig.suptitle("Concept Drift Adaptability: Dynamic Co-Tuning of Parameters and Semantic Labels",
                 fontweight='bold', fontsize=13, y=0.98)
    add_compact_badge(fig, "NAB: machine_temperature_system_failure.csv (Subsequences t=[1000..3200])", y_pos=0.012)
    plt.subplots_adjust(bottom=0.15, top=0.90)
    save_fig(fig, "Figure_5_11_Concept_Drift_Parameter_Adaptation")

def generate_figure_5_12_scalability_and_stability(n_repeats=3):
    print("\n--- Generating Figure 5.12: Window Structure Sensitivity & Explanation Stability ---")
    _, X_data, _ = load_dataset(MACHINE_TEMP)
    X_sub = X_data[:6000]

    window_sizes = [50, 100, 200, 400, 800]
    slide = 50
    mean_r_by_w, outlier_pct_by_w, label_distributions = [], [], []
    ms_per_window = {'Baseline DYN [1]': [], 'Integrated Expl [Ours]': []}

    for w in window_sizes:
        clf_e = dynamic_kr_expl(slide=slide, window=w)
        scores = clf_e.fit(X_sub)
        r_hist = np.array(clf_e.r_history)
        mean_r_by_w.append(r_hist[r_hist > 0].mean())   # -1 marks windows where DYN found no (k, R)
        outlier_pct_by_w.append((np.sum(scores == 1) / len(scores)) * 100)
        counts = pd.Series([l for l in clf_e.outlier_labels if l not in ('Inlier', 'Unclassified Outlier')]).value_counts(normalize=True) * 100
        label_distributions.append(counts)

        n_windows = len(clf_e.r_history)
        for name, cls in [('Baseline DYN [1]', dynamic_baseline), ('Integrated Expl [Ours]', dynamic_kr_expl)]:
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                cls(slide=slide, window=w).fit(X_sub)
                times.append(time.perf_counter() - t0)
            ms_per_window[name].append(1000 * np.median(times) / n_windows)
        print(f"w={w}: outliers={outlier_pct_by_w[-1]:.2f}% mean R={mean_r_by_w[-1]:.2f} (#windows without (k,R)={int((r_hist <= 0).sum())}) "
              f"time/window: baseline={ms_per_window['Baseline DYN [1]'][-1]:.2f} ms, ours={ms_per_window['Integrated Expl [Ours]'][-1]:.2f} ms "
              f"| classes={counts.round(1).to_dict()}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.8), gridspec_kw={'width_ratios': [1, 1, 1]})

    x = np.arange(len(window_sizes))
    ax1.plot(x, outlier_pct_by_w, marker='o', linewidth=2.4, color='#E53E3E', label='Detected Outlier Ratio (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"w={w}" for w in window_sizes], fontweight='bold')
    ax1.set_xlabel("Sliding Window Size ($w$)", fontweight='bold', labelpad=10)
    ax1.set_ylabel("Detected Outlier Percentage (%)", color='#E53E3E', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#E53E3E')
    ax1.set_ylim(0, max(outlier_pct_by_w) * 1.3)
    ax1.grid(True, linestyle='--', alpha=0.35)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, mean_r_by_w, marker='s', linewidth=2.2, linestyle='--', color='#3182CE', label='Mean Dynamic Radius (R)')
    ax1_twin.set_ylabel("Average Dynamic Radius ($R$)", color='#3182CE', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='#3182CE')
    ax1.set_title("(a) Window Field of View: Outlier Ratio and Mean Radius", fontweight='bold')

    df_w = pd.DataFrame(label_distributions, index=[f"w={w}" for w in window_sizes]).fillna(0)
    sorted_labels = [l for l in SEMANTIC_CLASSES if l in df_w.columns]
    y_pos = np.arange(len(df_w))
    left_accum = np.zeros(len(df_w))
    for lab in sorted_labels:
        vals = df_w[lab].values
        ax2.barh(y_pos, vals, left=left_accum, height=0.48, color=LABEL_COLORS.get(lab, '#718096'), edgecolor='white', label=lab)
        for i, (v, l) in enumerate(zip(vals, left_accum)):
            if v > 8:
                ax2.text(l + v/2, y_pos[i], f"{v:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        left_accum += vals
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_w.index, fontweight='bold')
    ax2.set_xlabel("Semantic Share of Outliers (%)", fontweight='bold', labelpad=10)
    ax2.set_title("(b) Semantic Class Shares across Window Scales", fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, framealpha=0.95, facecolor='white', edgecolor='#CBD5E0', title="Semantic Class")

    for name, col, mk in [('Baseline DYN [1]', '#718096', 'o'), ('Integrated Expl [Ours]', '#38A169', 's')]:
        ax3.plot(x, ms_per_window[name], marker=mk, linewidth=2.2, color=col, label=name)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"w={w}" for w in window_sizes], fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_xlabel("Sliding Window Size ($w$)", fontweight='bold', labelpad=10)
    ax3.set_ylabel("Processing Time per Window (ms, log scale)", fontweight='bold')
    ax3.set_title("(c) Scalability: Latency per Window", fontweight='bold')
    ax3.grid(True, which='both', linestyle='--', alpha=0.35)
    ax3.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#CBD5E0')

    fig.suptitle("Impact of Sliding Window Structure: Spatial Scale Sensitivity, Explanation Stability and Scalability",
                 fontweight='bold', fontsize=13, y=0.99)
    add_compact_badge(fig, "NAB machine_temperature_system_failure (first N = 6,000 subsequences)",
                      param_str=f"Fixed Slide s=50, w in [50..800], policy='or', l=10, timing = median of {n_repeats} runs", y_pos=0.018)
    plt.subplots_adjust(bottom=0.26, top=0.87, wspace=0.34)
    save_fig(fig, "Figure_5_12_Scalability_and_Window_Stability")

if __name__ == '__main__':
    print("="*75)
    print("STARTING FULL CHAPTER 5 EVALUATION GRAPHS SUITE GENERATION")
    print("Comparing Baseline DYN (dynamic.py) vs. Explainable DYN (dynamic_expl.py)")
    print("="*75)

    t_start = time.time()
    generate_figure_5_1_detection_fidelity()
    generate_figure_5_2_computational_overhead()
    generate_figure_5_3_sankey_and_entropy()
    generate_figure_5_4_benchmark_domain_profiles()
    composite_tel, composite_labels = run_composite_suite()
    generate_figure_5_5_density_fingerprints(composite_tel)
    generate_figure_5_6_case_study_nab_temp()
    generate_figure_5_7_case_study_aws_cpu()
    generate_figure_5_8_pca_topological_projection()
    generate_figure_5_9_conflict_resolution_and_severity(composite_tel)
    generate_figure_5_10_temporal_label_evolution(composite_labels)
    generate_figure_5_11_concept_drift_adaptation()
    generate_figure_5_12_scalability_and_stability()

    total_elapsed = time.time() - t_start
    print("="*75)
    print(f"ALL 12 CHAPTER 5 EVALUATION FIGURES GENERATED IN {total_elapsed:.1f} SECONDS")
    print(f"Saved to: {OUTPUT_DIRS[0]}")
    print("="*75)
