"""Figure 2.1: illustrative examples of the six semantic outlier classes.

Standalone script, independent of the Chapter 5 evaluation suite. Each panel is a small synthetic 2D
neighbourhood around a query point; the ring counts and the labels are computed with the real labeling
rules of dynamic_kr_expl (NN-based, Status-based and their combination), so the illustration can never
disagree with the implementation.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from Techniques.dynamic_expl import dynamic_kr_expl

OUTPUT_DIRS = [os.path.join(BASE_DIR, 'data', 'graphs')]
FILENAME = "Figure_2_1_Semantic_Classes_Illustration"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10

# Same class colours and markers as the Chapter 5 figures
LABEL_COLORS = {
    'Isolated': '#E53E3E', 'Near cluster': '#3182CE', 'Sparse microcluster': '#DD6B20',
    'Sparse cluster': '#319795', 'Dense microcluster': '#805AD5', 'Dense cluster': '#38A169'
}
LABEL_MARKERS = {
    'Isolated': 'o', 'Near cluster': '^', 'Sparse microcluster': 's',
    'Sparse cluster': 'D', 'Dense microcluster': 'P', 'Dense cluster': '*'
}

R, K = 1.0, 5                                   # reference radius and neighbour threshold (as chosen by DYN)
RADII = [R / 4, R / 2, R, 2 * R, 4 * R]
RING_NAMES = ["R/4", "R/2", "R", "2R", "4R"]
rng = np.random.default_rng(7)


def ring_points(n, r_min, r_max, angle_centre=0.0, angle_spread=np.pi):
    """n points at distance in (r_min, r_max) from the query, within +-angle_spread of angle_centre."""
    r = rng.uniform(r_min, r_max, n)
    a = angle_centre + rng.uniform(-angle_spread, angle_spread, n)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def blob(n, centre, radius):
    """n points uniformly inside a disc."""
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)]


# One synthetic neighbourhood per class (query point at the origin)
SCENARIOS = [
    ("Isolated", np.vstack([blob(45, (6.0, 0.6), 0.9)])),
    ("Near cluster", np.vstack([blob(45, (3.1, 0.4), 0.75)])),
    ("Sparse microcluster", np.vstack([ring_points(1, 0.6, 0.9, 2.2, 0.2),
                                       ring_points(4, 1.15, 1.8, 0.5, 0.35),
                                       blob(45, (6.0, 0.6), 0.9)])),
    ("Sparse cluster", np.vstack([ring_points(1, 0.6, 0.9, 0.0, 0.2),
                                  blob(55, (2.8, 0.0), 1.6)])),
    ("Dense microcluster", np.vstack([ring_points(1, 0.08, 0.22, 0.3, 0.3),
                                      ring_points(2, 0.3, 0.45, 0.6, 0.4),
                                      ring_points(1, 0.6, 0.9, 0.4, 0.3),
                                      ring_points(4, 1.1, 1.7, 0.6, 0.4),
                                      blob(45, (6.0, 0.6), 0.9)])),
    ("Dense cluster", np.vstack([ring_points(2, 0.08, 0.22, 0.2, 0.4),
                                 ring_points(2, 0.3, 0.45, 0.2, 0.4),
                                 blob(90, (2.45, 0.0), 1.15)])),
]


def label_neighbourhood(points):
    """Ring counts and labels exactly as collect_scores() of dynamic_kr_expl computes them."""
    d = np.linalg.norm(points, axis=1)
    counts = [int(np.sum(d <= rad)) for rad in RADII]      # the query itself is not in `points`
    nn_input = [c / K for c in counts]
    k_vals = [K / 4, K / 2, K, 2 * K, 4 * K]
    status = [[c < kv for kv in k_vals] for c in counts]
    clf = dynamic_kr_expl()
    class_nn, class_status, final = clf._combination_based_labeling(nn_input, status)
    is_outlier = np.sort(d)[K - 1] > R if len(d) >= K else True   # k-th neighbour beyond R
    return counts, class_nn, class_status, final, is_outlier


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    for ax, (intended, pts) in zip(axes.ravel(), SCENARIOS):
        counts, class_nn, class_status, final, is_outlier = label_neighbourhood(pts)
        assert is_outlier, f"{intended}: query point is not an outlier at (k={K}, R={R})"
        assert final == intended, f"{intended}: rules returned {final} (NN={class_nn}, Status={class_status}, counts={counts})"
        print(f"{intended:20s} counts={counts}  NN={class_nn:20s} Status={class_status:20s} -> {final}")

        col = LABEL_COLORS[final]
        for rad, name in zip(RADII, RING_NAMES):
            ax.add_patch(Circle((0, 0), rad, fill=False, linestyle='-' if rad == R else '--',
                                linewidth=1.8 if rad == R else 0.9, edgecolor='#2D3748' if rad == R else '#A0AEC0'))
            if rad >= R / 2:
                ax.text(rad * np.cos(np.radians(125)), rad * np.sin(np.radians(125)), name, fontsize=8,
                        color='#4A5568', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.85))
        ax.scatter([0], [0], s=150 if final == 'Dense cluster' else 75, marker=LABEL_MARKERS[final], color=col,
                   edgecolors='black', linewidths=0.8, zorder=5)
        # neighbours drawn above the query marker so that very close neighbours stay visible
        ax.scatter(pts[:, 0], pts[:, 1], s=11, color='#4A5568', alpha=0.8, linewidths=0, zorder=6)

        ax.set_title(final, fontweight='bold', fontsize=12.5, color='#1A202C', pad=6)
        ax.set_xlabel("Neighbours in R/4, R/2, R, 2R, 4R:  " + ",  ".join(str(c) for c in counts) +
                      f"\nNN-based: {class_nn}   |   Status-based: {class_status}",
                      fontsize=9, color='#2D3748', labelpad=6)
        ax.set_xlim(-4.6, 7.4)
        ax.set_ylim(-4.6, 4.6)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#CBD5E0')

    fig.suptitle(f"Semantic Outlier Classes from Multi-Query Neighbour Counts (illustrative 2D examples, k = {K})",
                 fontweight='bold', fontsize=13.5, y=0.985)
    # The marker/circle legend is given in the LaTeX caption of the figure.
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, wspace=0.05, hspace=0.28)

    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{FILENAME}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FILENAME}.png")


if __name__ == '__main__':
    main()
