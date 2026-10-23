# CM5 model vs. uMLIP embeddings, and consensus outlier screen

Code and data for Figures 9 and 10 of the manuscript. These scripts compare the one-parameter
CM5 model with ridge regression on per-atom embeddings from two universal machine-learned
interatomic potentials (uMLIPs), and screen the experimental σ values for entries that all
three models disagree with.

## Contents

| Path | Description |
|---|---|
| `01_build_structures.py` | Cuts each substituent from the benzonitrile SMILES in `csv_files/hammett_predictions.csv`, attaches it to benzene, and embeds it (RDKit ETKDGv3, seed 42, then MMFF94). Also indexes the ring atoms and assigns chemotype clusters. |
| `02_extract_embeddings.py` | Extracts ORB-v3-conservative-omol and MACE-OMOL-0 embeddings at the meta and para ring carbons for every layer. |
| `03_cross_validation.py` | Runs random grouped CV (with a layer sweep), the leave-one-chemotype-out hold-out and the σp-sign hold-out, and writes per-substituent out-of-fold predictions. |
| `04_consensus_outliers.py` | Flags pairs where all three models disagree with experiment, links each to Hansch Table 1 and its cited sources, and reports threshold sensitivity. |
| `05_figures.py` | Draws Figure 9, a calibrated (MAE) version of panel 9c, and Figure 10. |
| `structures/sub_XXX.xyz` | 370 RDKit/MMFF geometries of the monosubstituted benzenes (`XXX` = `row_idx`). |
| `data/substituent_table.csv` | One row per substituent: SMILES, CM5 charges, experimental and Eq. 2 σ, ring-atom indices, chemotype cluster. |
| `data/embeddings_orb_mace.npz` | Embeddings: `orb_para`, `orb_meta` (370 × 5 layers × 256) and `mace_para`, `mace_meta` (370 × 3 layers × 1024). |
| `data/hansch1991_sources_for_flagged_entries.csv` | Literature cited in Hansch, Leo & Taft (1991) Table 1 for each flagged entry, with a source classification. |
| `results/` | All tables used in the manuscript (see below). |
| `figures/` | PNG and SVG figures. |

## Running

```bash
conda env create -f conda_environments/umlip_comparison_environment.yml
conda activate hammett-umlip
python umlip_comparison/01_build_structures.py     # optional: structures and table are committed
python umlip_comparison/02_extract_embeddings.py   # optional: downloads model weights, ~10 min on CPU
python umlip_comparison/03_cross_validation.py
python umlip_comparison/04_consensus_outliers.py
python umlip_comparison/05_figures.py
```

Steps 01 and 02 regenerate the committed `structures/`, `data/substituent_table.csv` and
`data/embeddings_orb_mace.npz` exactly, with the package versions in the environment file.
Steps 03–05 take under a minute.

## Methods

**Geometries.** The embeddings are computed on force-field (MMFF94) geometries of the
monosubstituted benzenes PhX. These are not the DFT benzonitrile geometries used for the CM5
charges, and they were not relaxed with either uMLIP. The CM5 charges and experimental σ values
come unchanged from `csv_files/hammett_predictions.csv`.

**Embeddings.** For ORB, the node features after each of the five message-passing blocks are
captured with forward hooks (graph radius 6.0 Å, at most 120 neighbours). For MACE, the invariant
(l = 0) descriptors after each of the three interaction layers come from
`MACECalculator.get_descriptors`. Both models are run as neutral singlets. The para vector is the
para-carbon embedding; the meta vector is the mean of the two meta carbons.

**Regression.** CM5 uses ordinary least squares on one feature. The uMLIPs use `StandardScaler` +
`RidgeCV(alphas=logspace(-3, 3, 20))`, refitted inside every training fold. σm is fitted on meta
features and σp on para features; "pooled" stacks the 740 meta and para rows into one model.

**Protocols.**
- *Random CV:* 10-fold `GroupKFold` grouped by substituent. The uMLIP layer is chosen by the
  highest random-CV Spearman ρ (ORB: layer 3 for σm, layer 4 for σp and pooled; MACE: layer 2) and
  used in every protocol. Selection uses the same scores that are reported, so the uMLIP random-CV
  numbers are slightly optimistic.
- *Chemotype hold-out:* fragment Morgan fingerprints (radius 2, 1024 bits) are clustered with
  Butina at a Tanimoto distance of 0.7, giving 99 clusters. Each of the 11 largest (197
  substituents) is withheld in turn. As a baseline, random 10-fold CV over all 370 substituents
  is scored on the same 197.
- *σp-sign hold-out:* train on σp < 0 (n = 76) and test on σp ≥ 0 (n = 294; this includes 9
  substituents with σp = 0), then the reverse.

**Metrics.** Spearman ρ, Pearson r, R² (= 1 − SSE/SST on the test predictions), MAE and bias
(mean of predicted − experimental). A one-parameter linear model is a monotone function of its
descriptor, so its Spearman ρ on any test set equals |ρ(q, σ)| whatever it was trained on (the
raw-charge rows in `sigma_p_sign_holdout.csv` show this). Compare extrapolation between CM5 and
the ridge models with R², MAE and bias, not ρ.

**Consensus screen.** A pair is flagged when the out-of-fold residuals of CM5 (pooled fit), MACE
and ORB all have the same sign and the smallest |residual| exceeds 0.15. Each flag is linked to
its Hansch Table 1 entry via `csv_files/Chem_Rev_smiles_and_values.csv`. The cited sources were
checked against the reference column of Table 1 and against the reference list. A pair counts as
"secondary source" if at least one cited source is a compilation or predictive work (Perrin et
al. 1981; Charton 1981). It counts as "indirect method" if the cited source derives σ from NMR
shifts or polarographic half-wave potentials. The screen ranks entries for checking; it does not
show that a value is wrong. All three models shrink predictions towards the mean, so agreeing
residuals concentrate at the extremes of σ. `consensus_threshold_sensitivity.csv` reports the
fraction of flagged pairs on the shrinkage side at each threshold.

## Results files

| File | Contents |
|---|---|
| `random_cv_layer_sweep.csv` | Metrics for every model, layer and target. |
| `random_cv_summary.csv` | Best layer per model and target (Figure 9a). |
| `oof_predictions.csv` | One row per substituent and position: `sigma_exp`, `cm5_charge`, `oof_cm5`, `oof_mace`, `oof_orb`, and the layer used. |
| `chemotype_holdout.csv` | Leave-one-chemotype-out and the same-subset random-CV baseline (Figure 9b). |
| `sigma_p_sign_holdout.csv` | σp-sign hold-out with ρ, R², MAE and bias, plus the raw-charge ρ (Figure 9c). |
| `consensus_outliers.csv` | The 35 flagged pairs with residuals, Table 1 entry, cited sources and flags (Figure 10). |
| `consensus_threshold_sensitivity.csv` | Number of flags and fraction on the shrinkage side at thresholds 0.10–0.25. |
