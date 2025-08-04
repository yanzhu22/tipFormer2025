# -*- coding: utf-8 -*-
"""
Visualise key atoms and residues from cross-attention weights.
"""

import os
import pickle
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol
from PIL import ImageDraw, ImageFont

# ---------- helper functions ---------- #
def top_indices(scores, k):
    return np.argsort(scores)[-k:].tolist()

def highlight_atoms(smiles, scores, cid, top_k=5, out_dir="vis"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    n_vis = mol.GetNumAtoms()
    n_scores = len(scores)
    scores = scores[:n_vis] if n_scores > n_vis else scores
    idx = top_indices(scores, min(top_k, n_vis))
    img = Draw.MolToImage(mol, highlightAtoms=idx, highlightAtomColors={i:(1,0,0) for i in idx}, size=(600,400))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10,10), f"{cid} top-{top_k} atoms", fill="black", font=font)
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"{cid}_atoms.png"))

def highlight_residues(uniprot, scores, pid, top_k=8, out_dir="vis",
                       pdb_dir="./alphafold_structures"):
    pdb_path = os.path.join(pdb_dir, f"{uniprot}.pdb")
    if not os.path.exists(pdb_path):
        return
    idx = top_indices(scores, min(top_k, len(scores)))
    with open(pdb_path) as f:
        pdb = f.read()
    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb, 'pdb')
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}})
    for i in idx:
        view.addStyle({'resi': i+1}, {'stick': {'color': 'red'}})
    view.zoomTo()
    os.makedirs(out_dir, exist_ok=True)
    view.show()
    with open(os.path.join(out_dir, f"{pid}_residues.html"), "w") as f:
        f.write(view._make_html())

# ---------- batch visualisation ---------- #
def batch_visualise(model_path, comp_pkl, prot_pkl, csv_path,
                    out_dir="vis",
                    pdb_dir="./alphafold_structures"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device).eval()

    with open(comp_pkl, "rb") as f:
        comp_feat = pickle.load(f)
    with open(prot_pkl, "rb") as f:
        prot_feat = pickle.load(f)

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"ChemicalName": "cid", "TargetUniProtID": "pid", "SMILES": "SMILES"})
    for _, row in tqdm(df.itertuples(index=False), total=len(df)):
        cid, pid, smi = str(row.cid), str(row.pid), str(row.SMILES)
        if cid not in comp_feat or pid not in prot_feat:
            continue
        c = torch.tensor(comp_feat[cid].astype(np.float32)).unsqueeze(0).to(device)
        p = torch.tensor(prot_feat[pid].astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            _, attn_cp, attn_pc = model(c, p, return_attn=True)

        atom_scores = attn_cp.mean(dim=(0, 2)).cpu().numpy()
        res_scores  = attn_pc.mean(dim=(0, 2)).cpu().numpy()

        highlight_atoms(smi, atom_scores, cid, out_dir=out_dir)
        highlight_residues(pid, res_scores, pid, out_dir=out_dir, pdb_dir=pdb_dir)

# ---------- entry ---------- #
if __name__ == "__main__":
    batch_visualise(
        model_path="best_model.pth",
        comp_pkl="compound_atom_features.pkl",
        prot_pkl="protein_residue_features.pkl",
        csv_path="samples.csv",
        out_dir="vis",
    )
