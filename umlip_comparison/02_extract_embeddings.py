"""Extract per-atom uMLIP embeddings at the meta and para ring carbons.

Models (neutral singlet, CPU, float32):
  ORB-v3-conservative-omol (orb-models)  -- node features after each of the 5 GNN message-passing
                                            blocks (forward hooks), 256 dimensions per layer
  MACE-OMOL-0 (mace-torch, mace_omol)    -- invariant (l = 0) descriptors after each of the 3
                                            interaction layers, 1024 dimensions per layer
For each substituent the para-carbon vector is stored, and the meta vector is the mean of the two
meta carbons. Geometries are the RDKit/MMFF structures from 01_build_structures.py, used as-is
(no uMLIP relaxation).

Output: data/embeddings_orb_mace.npz with arrays row_idx, orb_para, orb_meta (370, 5, 256) and
mace_para, mace_meta (370, 3, 1024).
Run from the repository root:  python umlip_comparison/02_extract_embeddings.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read
from mace.calculators import mace_omol
from orb_models.forcefield import pretrained
from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs

HERE = Path(__file__).resolve().parent


def main():
    table = pd.read_csv(HERE / "data" / "substituent_table.csv")

    orb = pretrained.orb_v3_conservative_omol(device="cpu")
    orb.eval()
    blocks = orb.model.gnn_stacks
    captured = {}
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: captured.__setitem__(k, o[0].detach().cpu().numpy()))
             for k, b in enumerate(blocks)]
    system_config = SystemConfig(radius=6.0, max_num_neighbors=120)

    mace = mace_omol(device="cpu", default_dtype="float32")
    n_mace = int(mace.models[0].num_interactions)
    d_mace = 1024

    n = len(table)
    orb_para = np.full((n, len(blocks), 256), np.nan, np.float32)
    orb_meta = np.full_like(orb_para, np.nan)
    mace_para = np.full((n, n_mace, d_mace), np.nan, np.float32)
    mace_meta = np.full_like(mace_para, np.nan)

    for k, r in table.iterrows():
        path = HERE / "structures" / f"sub_{int(r.row_idx):03d}.xyz"
        p, m0, m1 = int(r.para_idx), int(r.meta_idx_0), int(r.meta_idx_1)

        atoms = read(path)
        atoms.info.update(charge=0, spin=0)
        captured.clear()
        orb.predict(ase_atoms_to_atom_graphs(atoms, system_config=system_config, device=torch.device("cpu")))
        for li in range(len(blocks)):
            e = captured[li]
            orb_para[k, li] = e[p]
            orb_meta[k, li] = 0.5 * (e[m0] + e[m1])

        desc = mace.get_descriptors(read(path), invariants_only=True, num_layers=-1)
        for li in range(n_mace):
            e = desc[:, li * d_mace:(li + 1) * d_mace]
            mace_para[k, li] = e[p]
            mace_meta[k, li] = 0.5 * (e[m0] + e[m1])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{n}", flush=True)

    for h in hooks:
        h.remove()
    assert not any(np.isnan(a).any() for a in (orb_para, orb_meta, mace_para, mace_meta))
    np.savez(HERE / "data" / "embeddings_orb_mace.npz", row_idx=table["row_idx"].values,
             orb_para=orb_para, orb_meta=orb_meta, mace_para=mace_para, mace_meta=mace_meta)
    print("saved data/embeddings_orb_mace.npz")


if __name__ == "__main__":
    main()
