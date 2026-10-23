"""Consensus screen for experimental sigma values that all three models disagree with.

A substituent/position pair is flagged when the out-of-fold residuals (predicted - experimental) of
CM5, MACE-OMOL-0 and ORB-v3-conservative-omol (results/oof_predictions.csv) share the same sign and
the smallest |residual| exceeds THRESHOLD (0.15 sigma units). Each flagged pair is linked to its
entry in Table 1 of Hansch, Leo and Taft (Chem. Rev. 1991, 91, 165) via
csv_files/Chem_Rev_smiles_and_values.csv, and to the literature cited for that entry
(data/hansch1991_sources_for_flagged_entries.csv, checked against the Table 1 reference column).

This is a screening heuristic for prioritising entries to check, not a test for experimental error.
All three models are least-squares/ridge fits to the same targets and so share shrinkage towards
the mean, which makes concordant disagreement more likely at the extremes of sigma; the
threshold-sensitivity and shrinkage-side summaries are written to help judge this.

Outputs in results/: consensus_outliers.csv, consensus_threshold_sensitivity.csv
Run from the repository root after 03_cross_validation.py:  python umlip_comparison/04_consensus_outliers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
THRESHOLD = 0.15
MODELS = ["cm5", "mace", "orb"]


def residuals():
    oof = pd.read_csv(HERE / "results" / "oof_predictions.csv")
    for m in MODELS:
        oof[f"resid_{m}"] = oof[f"oof_{m}"] - oof["sigma_exp"]
    r = oof[[f"resid_{m}" for m in MODELS]].values
    oof["same_sign"] = (np.sign(r) == np.sign(r[:, [0]])).all(axis=1) & (r[:, 0] != 0)
    oof["min_abs_resid"] = np.abs(r).min(axis=1)
    return oof


def flag(oof, threshold):
    return oof["same_sign"] & (oof["min_abs_resid"] > threshold)


def main():
    oof = residuals()
    flagged = oof[flag(oof, THRESHOLD)].copy()

    hansch = pd.read_csv(ROOT / "csv_files" / "Chem_Rev_smiles_and_values.csv")
    entry = hansch.dropna(subset=["Substituent"]).drop_duplicates("Substituent").set_index("Substituent")["Table 1 Reference Number"]
    flagged["table1_entry"] = flagged["Substituent"].map(entry).astype("Int64")

    src = pd.read_csv(HERE / "data" / "hansch1991_sources_for_flagged_entries.csv")
    per_entry = src.groupby("table1_entry").agg(
        hansch_ref_nos=("hansch_ref_no", lambda s: ", ".join(map(str, s))),
        citations=("citation", " | ".join),
        source_classes=("source_class", lambda s: ", ".join(sorted(set(s)))))
    flagged = flagged.join(per_entry, on="table1_entry")
    flagged["flag_secondary_source"] = flagged["source_classes"].str.contains("secondary")
    flagged["flag_indirect_method"] = flagged["source_classes"].str.contains("indirect")
    cols = ["Substituent", "position", "table1_entry", "sigma_exp", "oof_cm5", "oof_mace", "oof_orb",
            "resid_cm5", "resid_mace", "resid_orb", "min_abs_resid", "hansch_ref_nos", "source_classes",
            "flag_secondary_source", "flag_indirect_method", "citations"]
    flagged = flagged.sort_values("min_abs_resid", ascending=False)[cols]
    flagged.to_csv(HERE / "results" / "consensus_outliers.csv", index=False)

    mean_sigma = oof["sigma_exp"].mean()
    rows = []
    for t in [0.10, 0.15, 0.20, 0.25]:
        f = flag(oof, t)
        shrink = np.sign(oof.loc[f, "resid_cm5"]) == -np.sign(oof.loc[f, "sigma_exp"] - mean_sigma)
        rows.append({"threshold": t, "n_pairs": int(f.sum()), "n_substituents": oof.loc[f, "Substituent"].nunique(),
                     "fraction_on_shrinkage_side": float(shrink.mean()) if f.any() else np.nan})
    pd.DataFrame(rows).to_csv(HERE / "results" / "consensus_threshold_sensitivity.csv", index=False)

    print(f"{len(flagged)} pairs flagged ({flagged['Substituent'].nunique()} substituents); "
          f"secondary source: {int(flagged['flag_secondary_source'].sum())} pairs, "
          f"indirect method: {int(flagged['flag_indirect_method'].sum())} pairs")


if __name__ == "__main__":
    main()
