"""Draw the training / evaluation protocol without requiring data or fitted weights."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def main():
    fig, ax = plt.subplots(figsize=(10.8, 3.8), layout="constrained")
    ax.set(xlim=(0, 11), ylim=(0, 4))
    ax.axis("off")

    def box(x, y, text, color, width=2.5):
        ax.add_patch(FancyBboxPatch((x, y), width, .8, boxstyle="round,pad=.08",
                                   facecolor=color, edgecolor="#6b7280", lw=.7))
        ax.text(x + width/2, y + .4, text, ha="center", va="center", fontsize=9)

    def arrow(a, b, label=None):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color="#465362", lw=1.4))
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.14, label, ha="center", fontsize=8)

    box(.15, 2.65, "Pretraining regimes\nTarget regime excluded", "#eaf2ff")
    box(3.55, 2.65, "Pretrain: 20,000 updates\nAdam, cosine lr 10⁻³ → 10⁻⁵", "#eaf2ff", 3)
    box(7.45, 2.65, "Fine-tune: 2,000 updates\nTarget trajectories 0–1, lr 10⁻⁴", "#fff0e5", 3.1)
    arrow((2.73, 3.05), (3.45, 3.05))
    arrow((6.63, 3.05), (7.35, 3.05), "weights + normalisation")
    box(.15, .95, "Target trajectories 0–1\nIndependent initial weights", "#e9f8ef")
    box(3.55, .95, "Scratch: 22,000 updates\nAdam, cosine lr 10⁻³ → 10⁻⁵", "#e9f8ef", 3)
    box(7.45, .95, "Held-out trajectories 2–5\nOne-step error + 20-step rollout", "#f2effa", 3.1)
    arrow((2.73, 1.35), (3.45, 1.35))
    arrow((6.63, 1.35), (7.35, 1.35))
    arrow((9, 2.55), (9, 1.85))
    ax.text(5.5, .3, "Every update: sample regime → trajectory and time → predict next frame → loss → gradient → Adam\n"
            "Three model seeds; no evaluation gradient; curves consume no training randomness", ha="center", fontsize=9)
    ax.set_title("E1 protocol: pretraining + fine-tuning versus scratch at matched optimiser steps", fontsize=11)
    fig.savefig(Path(__file__).with_name("e1_protocol.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
