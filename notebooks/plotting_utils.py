import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import numpy as np
import seaborn as sns

from musint.benchmarks.muscle_sets import MIA_MUSCLES


def visualize_pose(vertices, frame, ax, elev=20, azim=210, roll=0, vertical_axis="y", title=""):
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1, c="r")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # Add text to the plot
    ax.text(0.5, 0.9, 1, f"Fr {frame:02d}", transform=ax.transAxes)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis)


def plot_emg_data(vals, preds, names, fig=None, gridspec=None, ylim=(0, 1.3), show_pred=True):
    sns.set_style(style="whitegrid")

    T, D = vals.shape

    # Bottom GridSpec (nested within the bottom part of gs_main)
    gs_bottom = GridSpecFromSubplotSpec(int(D / 2), 1, subplot_spec=gridspec[1])

    axes = [fig.add_subplot(gs_bottom[i, 0]) for i in range(int(D / 2))]

    lines = []

    for li in range(0, int(D / 2)):
        ri = li + int(D / 2)

        muscle_name_l = names[li]

        gt_l = vals[:, li]
        gt_r = vals[:, ri]

        pred_l, pred_r = preds[:, li], preds[:, ri]

        axes[li].plot(gt_l, label=f"Left", linewidth=2, color="royalblue", alpha=0.6)
        axes[li].plot(gt_r, label=f"Right", linewidth=2, color="orange", alpha=0.6)

        if show_pred:
            axes[li].plot(pred_l, label=f"Pred Left", linestyle="--", linewidth=2, color="royalblue")
            axes[li].plot(pred_r, label=f"Pred Right", linestyle="--", linewidth=2, color="orange")

            axes[li].fill_between(range(len(gt_l)), gt_l, pred_l, color="skyblue", alpha=0.4)
            axes[li].fill_between(range(len(gt_r)), gt_r, pred_r, color="orange", alpha=0.4)

        axes[li].set_yticks([0, 0.5, 1.0])
        axes[li].set_title(f"{muscle_name_l[:-4]}", rotation=-90, loc="right", x=1.05, y=0.3)

        # Hide x-axis for all but the last subplot
        if li != int(D / 2) - 1:
            axes[li].set_xticklabels([])
            axes[li].xaxis.grid(True)  # Ensure x-axis grid is visible
        else:
            axes[li].set_xlabel("Frames (20 fps)")

        lines.append(axes[li].axvline(x=0, color="r"))

        if ylim is not None:
            axes[li].set_ylim(ylim)
        else:
            axes[li].set_ylim((min(gt_l.min(), gt_r.min()), max(gt_l.max(), gt_r.max())))
        axes[li].grid(True)

    axes[0].legend(loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1, 1.5), handletextpad=2)

    return lines
