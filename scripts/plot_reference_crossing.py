"""Plot every primary family from qualified portable scores, without native arrays."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fractal_vlm_state_probe.stat_matched_source_storage import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0098_reference_crossing_results"


def main():
    summary = read(PUB / "summary.json")
    if summary["status"] != "VERIFIED_POSTHOC_REFERENCE_CROSSINGS":
        raise ValueError("qualified results required")
    output = PUB / "family_reference_conditions.png"
    if output.exists():
        raise FileExistsError(output)
    titles = {
        "encoder": "Encoder",
        "projector": "Projector",
        "full_raw_rgb": "Full raw RGB",
        "full_processor_pixel_values": "Full processor input",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True)
    colors = ("#007f78", "#b33247")
    for ax, p in zip(axes.flat, summary["primary_profiles"], strict=True):
        missing = 0
        for row in p["family_results"]:
            values = [row["fixed_change"], row["matched_change"]]
            if any(v is None for v in values):
                missing += 1
            ax.plot(
                [0, 1],
                [float("nan") if v is None else v for v in values],
                color="#b5b5b5",
                linewidth=1,
                alpha=0.85,
                zorder=1,
            )
            for x, value in enumerate(values):
                if value is not None:
                    ax.scatter(x, value, color=colors[x], s=29, zorder=2)
        values = [p["overall"]["fixed_change"], p["overall"]["matched_change"]]
        ax.plot(
            [0, 1],
            [float("nan") if v is None else v for v in values],
            color="#202020",
            marker="D",
            markersize=6,
            linewidth=2,
            zorder=3,
        )
        ax.axhline(0, color="#707070", linestyle="--", linewidth=0.8)
        ax.set_title(titles[p["view"]], fontsize=13)
        ax.set_xticks([0, 1], ["Fixed s0 reference", "Matched arrangement"])
        ax.set_xlim(-0.22, 1.22)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        if missing:
            ax.text(
                0.03,
                0.03,
                f"{missing}/8 families have unavailable endpoints",
                transform=ax.transAxes,
                fontsize=8,
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("Paired margin change from s0")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#b5b5b5", marker="o", label="Same family"),
            Line2D([0], [0], color="#202020", marker="D", label="Eight-family mean"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Reference condition changes the correspondence comparison", y=0.96, fontsize=15
    )
    fig.text(
        0.5,
        0.02,
        "16 test pairs / 8 known families. Post-hoc descriptive comparison; no confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    write(
        PUB / "figure_receipt.json",
        {
            "schema_version": 1,
            "analysis_kind": "reference_crossing_figure_receipt",
            "source_summary_sha256": sha(PUB / "summary.json"),
            "figure_sha256": sha(output),
            "script_sha256": sha(Path(__file__)),
            "all_families_retained": 8,
            "source_generation": 0,
            "model_contexts": 0,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
