"""Build monosubstituted-benzene geometries and the substituent table used for the uMLIP comparison.

For each of the 370 substituents in csv_files/hammett_predictions.csv the substituent fragment is cut
from the meta-benzonitrile SMILES, re-attached to a plain benzene ring, embedded with RDKit ETKDGv3
(randomSeed 42) and relaxed with MMFF94. Ring atoms (ipso, ortho, meta, para) are indexed relative to
the substituent attachment point. Substituent chemotypes are assigned by Butina clustering of
Morgan fingerprints (radius 2, 1024 bits) of the fragment at a Tanimoto distance threshold of 0.7.

Note: these are force-field geometries of PhX, not the DFT benzonitrile geometries used for the CM5
model. The CM5 charges and experimental sigma values are copied unchanged from hammett_predictions.csv.

Outputs (relative to this folder):
  structures/sub_XXX.xyz
  data/substituent_table.csv
Run from the repository root:  python umlip_comparison/01_build_structures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

CHEMOTYPE_NAMES = {  # names assigned by inspection of the 11 largest clusters
    0: "aryl/heteroaryl",
    1: "sulfonyl/CF3 (strong EWG)",
    2: "alkyl (branched)",
    3: "acetamido/acyl/ester",
    4: "perfluoroalkyl",
    5: "silyl",
    6: "nitrile/cyano",
    7: "halomethyl/thioether/phosphine",
    8: "phosphine oxide",
    9: "amide/carboxylate",
    10: "mixed tert-C (halomethyl/nitro/tBu)",
}


def load_wide():
    hp = pd.read_csv(ROOT / "csv_files" / "hammett_predictions.csv")
    hp["position"] = hp["nitrile_id"].str.extract(r"^(meta|para)_")
    wide = hp.pivot(index="Substituent", columns="position",
                    values=["nitrile_id", "nitrile_smiles", "cm5_charge", "hammett_parameter", "predicted_hammett"])
    wide.columns = [f"{c}_{p}" for c, p in wide.columns]
    return wide.reset_index()


def substituent_fragment(nitrile_smiles):
    """Return the substituent fragment SMILES (with a dummy atom at the attachment point)."""
    mol = Chem.MolFromSmiles(nitrile_smiles)
    if mol is None:
        return None
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()
             if len(r) == 6 and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)]
    nitrile_c = [a.GetNeighbors()[0].GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "N" and a.GetDegree() == 1
                 and a.GetBonds()[0].GetBondTypeAsDouble() == 3.0 and a.GetNeighbors()[0].GetSymbol() == "C"]
    core, nc, ipso_cn = None, None, None
    for c in nitrile_c:  # the benzonitrile CN is the nitrile whose carbon is bonded to an aromatic 6-ring
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            ring = next((r for r in rings if nbr.GetIdx() in r), None)
            if ring is not None:
                core, nc, ipso_cn = ring, c, nbr.GetIdx()
                break
        if core is not None:
            break
    if core is None:
        return None
    for a in core:
        if a == ipso_cn:
            continue
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() not in core and nbr.GetIdx() != nc:
                bond = mol.GetBondBetweenAtoms(a, nbr.GetIdx())
                em = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
                for idx, frag in zip(Chem.GetMolFrags(em, sanitizeFrags=False),
                                     Chem.GetMolFrags(em, asMols=True, sanitizeFrags=False)):
                    if not set(idx) & core:
                        try:
                            Chem.SanitizeMol(frag)
                        except Exception:
                            pass
                        return Chem.MolToSmiles(frag)
    return None


def attach_to_benzene(frag_smiles):
    frag = Chem.MolFromSmiles(frag_smiles)
    dummy = next(a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0)
    benzene = Chem.MolFromSmiles("c1ccccc1")
    em = Chem.RWMol(Chem.CombineMols(benzene, frag))
    d = benzene.GetNumAtoms() + dummy
    em.AddBond(0, em.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx(), Chem.BondType.SINGLE)
    em.RemoveAtom(d)
    mol = em.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def embed_and_index(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    ring = next(r for r in mol.GetRingInfo().AtomRings()
                if len(r) == 6 and all(mol.GetAtomWithIdx(a).GetIsAromatic() and mol.GetAtomWithIdx(a).GetSymbol() == "C"
                                       for a in r))
    ipso = next((a for a in ring for n in mol.GetAtomWithIdx(a).GetNeighbors()
                 if n.GetIdx() not in ring and n.GetAtomicNum() != 1), ring[0])
    order, prev, cur = [ipso], None, ipso
    while len(order) < 6:
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors() if n.GetIdx() in ring and n.GetIdx() != prev][0]
        order.append(nxt)
        prev, cur = cur, nxt
    return mol, dict(ipso_idx=ipso, ortho_idx_0=order[1], ortho_idx_1=order[5],
                     meta_idx_0=order[2], meta_idx_1=order[4], para_idx=order[3])


def chemotype_clusters(frag_smiles):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) for s in frag_smiles]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = sorted(Butina.ClusterData(dists, len(fps), distThresh=0.7, isDistData=True), key=len, reverse=True)
    labels = np.empty(len(fps), dtype=int)
    for k, members in enumerate(clusters):
        labels[list(members)] = k
    return labels


def main():
    wide = load_wide()
    wide["frag_smiles"] = wide["nitrile_smiles_meta"].apply(substituent_fragment)
    wide.loc[wide["Substituent"] == "[AtH]", "frag_smiles"] = "[1*][H]"
    assert wide["frag_smiles"].notna().all()
    wide["benzene_smiles"] = wide["frag_smiles"].apply(attach_to_benzene)

    out_dir = HERE / "structures"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for i, r in wide.iterrows():
        res = embed_and_index(r["benzene_smiles"])
        assert res is not None, r["Substituent"]
        mol, idx = res
        pos = mol.GetConformer().GetPositions()
        with open(out_dir / f"sub_{i:03d}.xyz", "w") as f:
            f.write(f"{mol.GetNumAtoms()}\n{r['Substituent']}\n")
            for a, c in zip(mol.GetAtoms(), pos):
                f.write(f"{a.GetSymbol()} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        rows.append({"row_idx": i, "n_atoms": mol.GetNumAtoms(), **idx})

    table = wide.join(pd.DataFrame(rows).set_index("row_idx"))
    table.insert(0, "row_idx", table.index)
    table["chemotype_cluster"] = chemotype_clusters(table["frag_smiles"])
    table["chemotype_name"] = table["chemotype_cluster"].map(CHEMOTYPE_NAMES).fillna("singleton/small cluster")
    table.to_csv(HERE / "data" / "substituent_table.csv", index=False)
    print(f"{len(table)} substituents, {table['chemotype_cluster'].nunique()} chemotype clusters")


if __name__ == "__main__":
    main()
