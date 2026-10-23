"""Figures 9 and 10 of the manuscript, plus a calibrated (MAE) version of panel 9c.

  figures/figure9_cm5_vs_umlip.png/.svg      (a) random 10-fold CV, (b) leave-one-chemotype-out,
                                              (c) sigma_p-sign hold-out, all as Spearman rho
  figures/figure9c_sigma_p_sign_mae.png/.svg  panel (c) as MAE with mean signed error annotated
  figures/figure10_outlier_sourcing.png/.svg  CM5 out-of-fold predictions vs experimental sigma,
                                              consensus-flagged pairs highlighted
Style follows notebooks/model_results_and_figure_generation.ipynb (falls back to DejaVu Sans if
Avenir is not installed).
Run from the repository root after 03 and 04:  python umlip_comparison/05_figures.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

paton_hot_pink = "#d00d44"
rosewood_gold = "#D4A574"
rosewood_slate = "#5B7FA6"
rosewood_charcoal = "#4A4A4A"
tol_grey = "#BBBBBB"
MODELS = ["cm5", "mace", "orb"]
COLORS = {"cm5": rosewood_charcoal, "mace": rosewood_slate, "orb": rosewood_gold}
LABELS = {"cm5": "CM5 (this work)", "mace": "MACE-OMOL-0", "orb": "ORB-v3-conservative-omol"}
SPLITS = [("sigma_p<0", "train $\\sigma_p$ < 0,\ntest $\\sigma_p$ ≥ 0"), ("sigma_p>=0", "train $\\sigma_p$ ≥ 0,\ntest $\\sigma_p$ < 0")]


def style():
    sns.set(context="talk", style="ticks", color_codes=True, rc={"legend.frameon": True})
    matplotlib.rcParams["figure.dpi"] = 300
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Avenir" if "Avenir" in names else "DejaVu Sans"
    plt.rcParams.update({"font.size": 18, "svg.fonttype": "none"})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=400)
    plt.close(fig)


def grouped_bars(ax, values, xticklabels, width=0.26):
    x = np.arange(len(xticklabels))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * width, values[m], width=width, color=COLORS[m], label=LABELS[m], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=15)


def sign_split_bars(ax, sp, column):
    w, k = 0.13, 0
    for target, hatch in [("sigma_p", None), ("sigma_m", "//")]:
        for m in MODELS:
            vals = [sp[(sp.model == m) & (sp.target == target) & (sp.train == tr)][column].values[0] for tr, _ in SPLITS]
            ax.bar(np.arange(2) + (k - 2.5) * w, vals, width=w, color=COLORS[m], hatch=hatch, edgecolor="white", linewidth=0.6)
            k += 1
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([lab for _, lab in SPLITS], fontsize=14)


def sign_legend(ax):
    handles = [Line2D([0], [0], marker="s", color="w", label=LABELS[m], markerfacecolor=COLORS[m], markersize=14) for m in MODELS]
    handles += [Line2D([0], [0], marker="s", color="w", label=lab, markerfacecolor="white", markeredgecolor="black", markersize=14)
                for lab in ("$\\sigma_p$ (solid)", "$\\sigma_m$ (hatched)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=10,
              frameon=False, columnspacing=1.0, handletextpad=0.4)


def figure9():
    rcv = pd.read_csv(RES / "random_cv_summary.csv")
    ch = pd.read_csv(RES / "chemotype_holdout.csv")
    ch = ch[ch.protocol == "leave_one_chemotype_out"]
    sp = pd.read_csv(RES / "sigma_p_sign_holdout.csv")
    sp = sp[sp.train != "none (raw CM5 charge)"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    targets = ["sigma_m", "sigma_p", "pooled"]
    grouped_bars(axes[0], {m: [rcv[(rcv.model == m) & (rcv.target == t)].spearman_rho.values[0] for t in targets] for m in MODELS},
                 ["$\\sigma_m$", "$\\sigma_p$", "pooled"])
    axes[0].set(ylim=(0.80, 0.95), ylabel="Spearman $\\rho$", title="Random 10-fold CV")
    axes[0].legend(loc="lower right", fontsize=11, frameon=True, fancybox=True, borderpad=0.5)

    grouped_bars(axes[1], {m: [ch[(ch.model == m) & (ch.target == t)].spearman_rho.values[0] for t in ["sigma_m", "sigma_p"]] for m in MODELS},
                 ["$\\sigma_m$", "$\\sigma_p$"])
    axes[1].set(ylim=(0.80, 1.0), ylabel=f"Spearman $\\rho$ (N={int(ch.n.iloc[0])} pooled)",
                title="Leave-one-chemotype-out\n(11 largest substituent families)")

    sign_split_bars(axes[2], sp, "spearman_rho")
    axes[2].set(ylim=(0, 1.0), ylabel="Spearman $\\rho$", title="$\\sigma_p$-sign hold-out")
    sign_legend(axes[2])
    for ax in axes:
        sns.despine(ax=ax)
        ax.margins(0.03)
    fig.tight_layout()
    save(fig, "figure9_cm5_vs_umlip")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sign_split_bars(ax, sp, "mae")
    ax.set(ylabel="MAE ($\\sigma$ units)", title="$\\sigma_p$-sign hold-out, calibrated error")
    sign_legend(ax)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "figure9c_sigma_p_sign_mae")


def figure10():
    oof = pd.read_csv(RES / "oof_predictions.csv")
    out = pd.read_csv(RES / "consensus_outliers.csv")
    key = list(zip(oof.Substituent, oof.position))
    flagged = set(zip(out.Substituent, out.position))
    secondary = set(zip(out[out.flag_secondary_source].Substituent, out[out.flag_secondary_source].position))
    is_sec = np.array([k in secondary for k in key])
    is_other = np.array([k in flagged for k in key]) & ~is_sec
    is_none = ~(is_sec | is_other)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(oof.sigma_exp[is_none], oof.oof_cm5[is_none], s=28, alpha=0.35, color=rosewood_slate, edgecolor="none")
    ax.scatter(oof.sigma_exp[is_other], oof.oof_cm5[is_other], s=55, alpha=0.9, color=rosewood_gold,
               edgecolor=rosewood_charcoal, linewidth=0.5)
    ax.scatter(oof.sigma_exp[is_sec], oof.oof_cm5[is_sec], s=90, alpha=0.95, color=paton_hot_pink,
               edgecolor=rosewood_charcoal, linewidth=0.7, marker="D")
    ax.plot([-1, 1.5], [-1, 1.5], color=tol_grey, lw=1, ls="--", zorder=0)
    ax.set(xlim=(-1, 1.5), ylim=(-1, 1.5), xlabel="reported experimental $\\sigma$ (Hansch, Leo, Taft, 1991)",
           ylabel="CM5 out-of-fold predicted $\\sigma$")
    handles = [
        Line2D([0], [0], marker="o", color="w", label=f"not flagged ({is_none.sum()} pairs)", markerfacecolor=rosewood_slate, markersize=11, alpha=0.6),
        Line2D([0], [0], marker="o", color="w", label=f"consensus-flagged, primary source ({is_other.sum()} pairs)",
               markerfacecolor=rosewood_gold, markersize=13, alpha=0.9, markeredgecolor=rosewood_charcoal),
        Line2D([0], [0], marker="D", color="w", label=f"consensus-flagged, non-experimental source ({is_sec.sum()} pairs)",
               markerfacecolor=paton_hot_pink, markersize=12, alpha=0.95, markeredgecolor=rosewood_charcoal),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=True, fancybox=True, borderpad=0.6)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "figure10_outlier_sourcing")


if __name__ == "__main__":
    style()
    figure9()
    figure10()
    print("figures written to", FIG)
