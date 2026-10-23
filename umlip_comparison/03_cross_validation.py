"""Compare the one-parameter CM5 model with ridge regression on uMLIP ring-carbon embeddings.

Models
  cm5   ordinary least squares on the CM5 charge of the nitrile N (sigma = a*q + b)
  mace  StandardScaler + RidgeCV(alphas = logspace(-3, 3, 20)) on MACE-OMOL-0 embeddings
  orb   same ridge pipeline on ORB-v3-conservative-omol embeddings
The scaler and alpha are fitted on the training folds only. sigma_m uses the meta-carbon embedding,
sigma_p the para-carbon embedding. 'pooled' stacks both positions (740 rows) with one model.

Protocols
  random CV     10-fold GroupKFold grouped by substituent (scikit-learn fold assignment)
  chemotype     leave-one-chemotype-out over the 11 largest Butina clusters (197 substituents);
                compared with random 10-fold CV (KFold, shuffle, seed 0) over all 370
                substituents, scored on the same 197
  sigma_p sign  train on sigma_p < 0 (n = 76), test on sigma_p >= 0 (n = 294), and the reverse

Layer selection: for each uMLIP and target the layer with the highest random-CV Spearman rho is
used in all protocols (ORB: 3 for sigma_m, 4 for sigma_p/pooled; MACE: 2). Because selection uses
the same CV scores that are reported, the uMLIP random-CV numbers are slightly optimistic.

Metrics: Spearman rho, Pearson r, R^2 (1 - SSE/SST on the test predictions), MAE and mean signed
error (bias = predicted - experimental). For a one-parameter linear model Spearman rho on any test
set equals |rho(q, sigma)| and does not depend on the training data, so calibrated metrics
(R^2, MAE, bias) are needed to compare extrapolation between CM5 and the ridge models.

Outputs in results/:
  random_cv_layer_sweep.csv, random_cv_summary.csv, oof_predictions.csv,
  chemotype_holdout.csv, sigma_p_sign_holdout.csv
Run from the repository root:  python umlip_comparison/03_cross_validation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
POS = {"sigma_m": "meta", "sigma_p": "para"}
N_LAYERS = {"orb": 5, "mace": 3}
N_CHEMOTYPES = 11


def load():
    table = pd.read_csv(HERE / "data" / "substituent_table.csv")
    emb = np.load(HERE / "data" / "embeddings_orb_mace.npz")
    assert (emb["row_idx"] == table["row_idx"].values).all()
    return table, emb


def estimator(model):
    if model == "cm5":
        return LinearRegression()
    return Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20)))])


def features(table, emb, model, pos, layer=None):
    if model == "cm5":
        return table[[f"cm5_charge_{pos}"]].values
    return emb[f"{model}_{pos}"][:, layer, :]


def design(table, emb, model, target, layer=None):
    """Return X, y, groups for a target ('sigma_m', 'sigma_p' or 'pooled')."""
    if target == "pooled":
        X = np.vstack([features(table, emb, model, "meta", layer), features(table, emb, model, "para", layer)])
        y = np.concatenate([table["hammett_parameter_meta"], table["hammett_parameter_para"]])
        g = np.concatenate([table["Substituent"], table["Substituent"]])
        return X, y, g
    pos = POS[target]
    return features(table, emb, model, pos, layer), table[f"hammett_parameter_{pos}"].values, table["Substituent"].values


def metrics(y, p):
    return {"n": len(y), "spearman_rho": stats.spearmanr(y, p).correlation, "pearson_r": stats.pearsonr(y, p)[0],
            "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), "mae": np.mean(np.abs(y - p)),
            "bias": np.mean(p - y)}


def random_cv(table, emb):
    sweep, oof = [], {}
    for model in ["cm5", "mace", "orb"]:
        for layer in ([None] if model == "cm5" else range(N_LAYERS[model])):
            for target in ["sigma_m", "sigma_p", "pooled"]:
                X, y, g = design(table, emb, model, target, layer)
                p = cross_val_predict(estimator(model), X, y, cv=GroupKFold(n_splits=10), groups=g)
                sweep.append({"model": model, "layer": layer, "target": target, **metrics(y, p)})
                oof[(model, layer, target)] = p
    sweep = pd.DataFrame(sweep)
    best = sweep.loc[sweep.groupby(["model", "target"])["spearman_rho"].idxmax()].sort_values(["target", "model"])
    layers = {(r.model, r.target): (None if pd.isna(r.layer) else int(r.layer)) for r in best.itertuples()}
    return sweep, best, layers, oof


def oof_table(table, layers, oof):
    """Per-substituent out-of-fold predictions. CM5 from the pooled fit (as in Figure 10), uMLIPs
    from per-position fits at the selected layer."""
    rows = []
    for pos, target, sl in [("meta", "sigma_m", slice(0, len(table))), ("para", "sigma_p", slice(len(table), None))]:
        d = table[["row_idx", "Substituent", f"cm5_charge_{pos}", f"hammett_parameter_{pos}"]].copy()
        d.columns = ["row_idx", "Substituent", "cm5_charge", "sigma_exp"]
        d.insert(2, "position", pos)
        d["oof_cm5"] = oof[("cm5", None, "pooled")][sl]
        for m in ["mace", "orb"]:
            d[f"oof_{m}"] = oof[(m, layers[(m, target)], target)]
            d[f"{m}_layer"] = layers[(m, target)]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def fit_predict(model, Xtr, ytr, Xte):
    return estimator(model).fit(Xtr, ytr).predict(Xte)


def chemotype_holdout(table, emb, layers):
    cl = table["chemotype_cluster"].values
    held = cl < N_CHEMOTYPES
    rows = []
    for target, pos in POS.items():
        for model in ["cm5", "mace", "orb"]:
            X = features(table, emb, model, pos, layers.get((model, target)))
            y = table[f"hammett_parameter_{pos}"].values
            loco = np.full(len(y), np.nan)
            for c in range(N_CHEMOTYPES):
                te = cl == c
                loco[te] = fit_predict(model, X[~te], y[~te], X[te])
            # baseline: random 10-fold CV over all 370 substituents, scored on the same 197
            rcv = np.empty(len(y))
            for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
                rcv[te] = fit_predict(model, X[tr], y[tr], X[te])
            for name, p in [("leave_one_chemotype_out", loco), ("random_10fold_same_197", rcv)]:
                rows.append({"protocol": name, "target": target, "model": model,
                             "layer": layers.get((model, target)), **metrics(y[held], p[held])})
    return pd.DataFrame(rows)


def sigma_p_sign_holdout(table, emb, layers):
    neg = (table["hammett_parameter_para"] < 0).values
    rows = []
    for target, pos in POS.items():
        y = table[f"hammett_parameter_{pos}"].values
        for model in ["cm5", "mace", "orb"]:
            X = features(table, emb, model, pos, layers.get((model, target)))
            for train_name, tr in [("sigma_p<0", neg), ("sigma_p>=0", ~neg)]:
                p = fit_predict(model, X[tr], y[tr], X[~tr])
                rows.append({"train": train_name, "test": "sigma_p>=0" if train_name == "sigma_p<0" else "sigma_p<0",
                             "target": target, "model": model, "layer": layers.get((model, target)),
                             "n_train": int(tr.sum()), **metrics(y[~tr], p)})
        q = table[f"cm5_charge_{pos}"].values  # rank correlation of the raw descriptor, no fitting
        for name, mask in [("sigma_p>=0", ~neg), ("sigma_p<0", neg)]:
            rows.append({"train": "none (raw CM5 charge)", "test": name, "target": target, "model": "cm5",
                         "n": int(mask.sum()), "spearman_rho": abs(stats.spearmanr(q[mask], y[mask]).correlation)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    table, emb = load()
    sweep, best, layers, oof = random_cv(table, emb)
    sweep.to_csv(OUT / "random_cv_layer_sweep.csv", index=False)
    best.to_csv(OUT / "random_cv_summary.csv", index=False)
    oof_table(table, layers, oof).to_csv(OUT / "oof_predictions.csv", index=False)
    chemotype_holdout(table, emb, layers).to_csv(OUT / "chemotype_holdout.csv", index=False)
    sigma_p_sign_holdout(table, emb, layers).to_csv(OUT / "sigma_p_sign_holdout.csv", index=False)
    print(best[["model", "layer", "target", "spearman_rho", "r2", "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
