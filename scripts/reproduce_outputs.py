from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
OUTPUTS = PACKAGE / "outputs"
FIGURES = OUTPUTS / "figures"


ENGINE_COLORS = {
    "ARM Compute Library": "#356C9B",
    "Caffe-SSD": "#9A6A45",
    "MNN": "#4F8F5B",
    "NCNN": "#B64E58",
    "OpenCV": "#7A7A7A",
    "Paddle Lite": "#3E8C8A",
    "Tengine": "#C47A32",
    "TensorFlow Lite": "#8A6BBE",
    "TVM": "#5D6C89",
}

BOXPLOT_COLORS = [
    "#356C9B",
    "#4F8F5B",
    "#C47A32",
    "#8A6BBE",
    "#B64E58",
    "#3E8C8A",
    "#7A7A7A",
    "#9A6A45",
    "#5D6C89",
]

IMAGENET_MODELS = {
    "alexnet": ("AlexNet", "o"),
    "googlenet": ("GoogLeNet", "s"),
    "mobilenet_v2": ("MobileNetV2", "^"),
    "resnet50": ("ResNet50", "D"),
    "squeezenet_v1_1": ("SqueezeNet v1.1", "P"),
}

RETINAFACE_MARKERS = {
    "320x320": "o",
    "640x640": "s",
    "1024x1024": "D",
}


def model_label(name: str) -> str:
    labels = {
        "wide_resnet50_2": "Wide ResNet50-2",
        "squeezenet1_0": "SqueezeNet 1.0",
        "squeezenet1_1": "SqueezeNet 1.1",
        "densenet161": "DenseNet161",
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "googlenet": "GoogLeNet",
        "efficientnet_b4": "EfficientNet-B4",
        "efficientnet_b5": "EfficientNet-B5",
        "efficientnet_b6": "EfficientNet-B6",
        "efficientnet_b7": "EfficientNet-B7",
        "vgg16": "VGG16",
    }
    return labels.get(name, name.replace("_", " "))


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str, png: bool = False) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if png:
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_multi_engine_latency() -> None:
    latency = pd.read_csv(DATA / "base_latency_measurements.csv")
    ref = latency[
        latency["workload_group"].eq("reference")
        & latency["threads"].astype(str).eq("4")
    ].copy()
    grouped = (
        ref.groupby(["device", "operating_system", "model", "engine_family"], as_index=False)[
            "latency_ms"
        ]
        .median()
    )
    counts = (
        grouped.groupby(["device", "operating_system", "model"])["engine_family"]
        .nunique()
        .reset_index(name="n_engines")
    )
    matched = grouped.merge(counts, on=["device", "operating_system", "model"])
    matched = matched[matched["n_engines"].ge(3)].copy()
    matched["best_latency_ms"] = matched.groupby(["device", "operating_system", "model"])[
        "latency_ms"
    ].transform("min")
    matched["relative_latency"] = matched["latency_ms"] / matched["best_latency_ms"]
    matched["is_best"] = matched["latency_ms"].eq(matched["best_latency_ms"])

    summary = (
        matched.groupby("engine_family")
        .agg(
            comparisons=("relative_latency", "size"),
            best_cases=("is_best", "sum"),
            median_ratio=("relative_latency", "median"),
        )
        .reset_index()
        .sort_values(["median_ratio", "best_cases"], ascending=[True, False])
    )
    order = summary["engine_family"].tolist()
    data = [
        matched.loc[matched["engine_family"].eq(engine), "relative_latency"].to_numpy()
        for engine in order
    ]
    labels = [
        f"{row.engine_family}\n(n={int(row.comparisons)}, f={int(row.best_cases)})"
        for row in summary.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        widths=0.58,
        medianprops={"color": "#222222", "linewidth": 1.2},
        boxprops={"linewidth": 0.8, "color": "#333333"},
        whiskerprops={"linewidth": 0.8, "color": "#333333"},
        capprops={"linewidth": 0.8, "color": "#333333"},
        flierprops={
            "marker": "o",
            "markersize": 2.4,
            "markerfacecolor": "#555555",
            "markeredgewidth": 0,
            "alpha": 0.65,
        },
    )
    for patch, color in zip(bp["boxes"], BOXPLOT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    ax.axhline(1.0, color="#222222", linewidth=0.7, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("Relative latency to fastest matched engine")
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.85, max(matched["relative_latency"].max() * 1.25, 3))
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.45, which="both")
    ax.text(
        0.01,
        0.97,
        "Four-thread reference records; f = fastest cases",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
    )
    save(fig, "multi_engine_relative_latency", png=True)


def plot_imagenet_accuracy_latency() -> None:
    accuracy = pd.read_csv(DATA / "base_imagenet_accuracy.csv")
    subset = accuracy.dropna(subset=["top1_pct", "top5_pct", "latency_ms"]).copy()
    subset = subset[
        subset["hardware"].eq("Raspberry Pi 4 Model B Rev 1.1")
        & subset["operating_system"].eq("Debian-aarch64")
        & subset["model"].isin(IMAGENET_MODELS)
    ].copy()

    fig, (ax_high, ax_low) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.1, 3.65),
        gridspec_kw={"height_ratios": [1.25, 0.95], "hspace": 0.06},
    )
    for ax in (ax_high, ax_low):
        for _, row in subset.iterrows():
            label, marker = IMAGENET_MODELS[row["model"]]
            ax.scatter(
                row["latency_ms"],
                row["top1_pct"],
                s=54,
                marker=marker,
                color=ENGINE_COLORS.get(row["engine_family"], "#555555"),
                edgecolor="white",
                linewidth=0.65,
                alpha=0.93,
                label=label,
            )
        ax.set_xscale("log")
        ax.grid(axis="both", color="#D6D6D6", linewidth=0.45, which="both")
        ax.set_axisbelow(True)

    ax_high.set_ylim(66.7, 71.1)
    ax_low.set_ylim(54.7, 56.5)
    ax_high.spines.bottom.set_visible(False)
    ax_low.spines.top.set_visible(False)
    ax_high.tick_params(labelbottom=False, bottom=False)
    ax_low.set_xlabel("One-thread latency (ms, log scale)")
    fig.text(0.012, 0.5, "Top-1 accuracy (%)", va="center", rotation="vertical")

    break_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=7,
        linestyle="none",
        color="#333333",
        mec="#333333",
        mew=0.85,
        clip_on=False,
    )
    ax_high.plot([0, 1], [0, 0], transform=ax_high.transAxes, **break_kwargs)
    ax_low.plot([0, 1], [1, 1], transform=ax_low.transAxes, **break_kwargs)

    engine_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=engine,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6.2,
        )
        for engine, color in ENGINE_COLORS.items()
        if engine in set(subset["engine_family"])
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            label=label,
            markerfacecolor="#F7F7F7",
            markeredgecolor="#333333",
            markersize=6.2,
        )
        for _, (label, marker) in IMAGENET_MODELS.items()
    ]
    engine_legend = ax_high.legend(
        handles=engine_handles,
        title="Engine",
        ncol=3,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.42),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    ax_high.add_artist(engine_legend)
    ax_high.legend(
        handles=model_handles,
        title="Model",
        ncol=3,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.42),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    save(fig, "imagenet_accuracy_latency_rpi4_debian")


def plot_retinaface() -> None:
    retina = pd.read_csv(DATA / "retinaface_application_measurements.csv")
    data = retina.copy()
    data["engine_family"] = data["engine_family"].replace({"TENGINE": "Tengine"})
    data["sample_id"] = data.groupby(["input_size", "engine_family", "metric"]).cumcount()
    latency = data[data["metric"].eq("latency")][
        ["input_size", "engine_family", "sample_id", "latency_ms"]
    ].copy()
    memory = data[data["metric"].eq("memory")][
        ["input_size", "engine_family", "sample_id", "memory_mb"]
    ].copy()
    paired = latency.merge(memory, on=["input_size", "engine_family", "sample_id"])

    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    for _, row in paired.iterrows():
        ax.scatter(
            row["latency_ms"],
            row["memory_mb"],
            s=48,
            marker=RETINAFACE_MARKERS[row["input_size"]],
            color=ENGINE_COLORS.get(row["engine_family"], "#555555"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("Memory usage (MB, log scale)")
    ax.grid(axis="both", color="#D6D6D6", linewidth=0.45, which="both")
    ax.set_axisbelow(True)

    engine_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=engine,
            markerfacecolor=ENGINE_COLORS[engine],
            markeredgecolor="white",
            markersize=6.2,
        )
        for engine in ["NCNN", "Tengine"]
    ]
    input_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            label=input_size,
            markerfacecolor="#F7F7F7",
            markeredgecolor="#333333",
            markersize=6.2,
        )
        for input_size, marker in RETINAFACE_MARKERS.items()
    ]
    engine_legend = ax.legend(
        handles=engine_handles,
        title="Engine",
        ncol=2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.18),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    ax.add_artist(engine_legend)
    ax.legend(
        handles=input_handles,
        title="Input size",
        ncol=3,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    save(fig, "retinaface_latency_memory")


def plot_tvm() -> None:
    latency = pd.read_csv(DATA / "tvm_extension_latency.csv")
    pivot = latency.pivot_table(
        index=["model", "size_mb", "params_m", "gmacs", "cpu_cores"],
        columns="variant",
        values="latency_ms",
        aggfunc="first",
    ).reset_index()
    paired = pivot.dropna(subset=["optimized", "unoptimized"]).copy()
    paired["optimization_speedup"] = paired["unoptimized"] / paired["optimized"]

    optimized = latency[latency["variant"].eq("optimized")].copy()
    scaling = optimized.pivot_table(
        index=["model", "size_mb", "params_m", "gmacs"],
        columns="cpu_cores",
        values="latency_ms",
        aggfunc="first",
    ).reset_index()
    scaling = scaling.dropna(subset=[1, 4]).copy()
    scaling["cpu_scaling_speedup"] = scaling[1] / scaling[4]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].boxplot(
        [
            paired.loc[paired["cpu_cores"].eq(core), "optimization_speedup"].to_numpy()
            for core in [1, 2, 3, 4]
        ],
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#222222", "linewidth": 1.15},
        boxprops={"facecolor": "#5E8CB8", "alpha": 0.82, "linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 2.4,
            "markerfacecolor": "#555555",
            "markeredgewidth": 0,
            "alpha": 0.65,
        },
    )
    axes[0].axhline(1.0, color="#333333", linewidth=0.7, linestyle="--")
    axes[0].set_xticklabels(["1", "2", "3", "4"])
    axes[0].set_xlabel("CPU cores")
    axes[0].set_ylabel("Unoptimized/optimized latency")
    axes[0].set_title("(a) Optimization speed-up")
    axes[0].grid(axis="y", color="#D6D6D6", linewidth=0.45)
    axes[0].set_axisbelow(True)

    top = scaling.sort_values("cpu_scaling_speedup", ascending=True).tail(12)
    labels = [model_label(name) for name in top["model"]]
    axes[1].barh(labels, top["cpu_scaling_speedup"], color="#4F8F5B", height=0.62)
    axes[1].axvline(1.0, color="#333333", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("1-core/4-core latency")
    axes[1].set_title("(b) Largest optimized CPU-scaling gains")
    axes[1].grid(axis="x", color="#D6D6D6", linewidth=0.45)
    axes[1].set_axisbelow(True)
    axes[1].set_xlim(0, max(3.4, top["cpu_scaling_speedup"].max() * 1.12))
    for y, value in enumerate(top["cpu_scaling_speedup"]):
        axes[1].text(value + 0.04, y, f"{value:.2f}", va="center", ha="left", fontsize=7.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "tvm_optimization_scaling")


def write_run_summary() -> None:
    data_files = []
    for path in sorted(DATA.glob("*.csv")):
        frame = pd.read_csv(path)
        data_files.append(
            {
                "file": f"data/{path.name}",
                "rows": len(frame),
                "columns": len(frame.columns),
            }
        )
    summary = {
        "package": "Deployment-stack effects in multi-engine DNN inference on ARM edge platforms",
        "data_files": data_files,
        "generated_outputs": {
            "figures": sorted(path.name for path in FIGURES.glob("*")),
        },
    }
    (OUTPUTS / "reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_plot_style()
    plot_multi_engine_latency()
    plot_imagenet_accuracy_latency()
    plot_retinaface()
    plot_tvm()
    write_run_summary()
    print(f"Reproduced supplementary outputs under {OUTPUTS}")


if __name__ == "__main__":
    main()
