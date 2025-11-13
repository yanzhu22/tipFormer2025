# --------------------------------------------------------
# This module adapts feature extraction strategies from:
#   https://github.com/CSUBioGroup/DTIAM/blob/main/code/data_process/extract_feature.py
#
# Original work by CSUBioGroup (DTIAM project).
# Modifications made for tipFormer.
# --------------------------------------------------------

import os
import sys
sys.path.append('./model/BerMol/')
import torch
import pickle
import dill as pickle
import json
import esm
import pandas as pd
from bermol.trainer import BerMolPreTrainer
from tqdm import tqdm
from collections import OrderedDict


def cal_comp_feat(data: pd.DataFrame, model_path: str, device: str = "cuda") -> dict:
    """
    Calculate the compound features using the compound pre-trained model
    """
    with open(model_path, "rb") as f:
        comp_model = pickle.load(f)
        comp_model.model.to(device)
        comp_model.model.eval()

    def smi_to_vec(smi):
        output = comp_model.transform(smi, device)
        return output[1].cpu().detach().numpy().reshape(-1)

    comp_feat = {}
    comp_data = data[["ChemicalName", "SMILES"]].drop_duplicates(subset=["ChemicalName"])
    
    for _, row in tqdm(comp_data.iterrows()):
        cid, smi = row[0], row[1]
        print(smi)
        comp_feat[cid] = smi_to_vec(smi)
        

    return comp_feat


def cal_prot_feat(data: pd.DataFrame) -> dict:
    """
    Calculate the protein features using the protein pre-trained model
    """
    # model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    repr_layer = model.num_layers

    def seq_to_vecs(pid, seq, max_length=1022):
        data = [
            (pid, seq[:max_length]),
        ]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
        token_representations = results["representations"][repr_layer]
        sequence_representations = token_representations[0, 1:].mean(0)
        return sequence_representations.cpu().detach().numpy().reshape(-1)

    prot_feat = {}
    prot_data = data[["UniProtKB", "Sequence"]].drop_duplicates(subset=["UniProtKB"])
    for _, row in tqdm(prot_data.iterrows()):
        pid, seq = row[0], row[1]
        prot_feat[pid] = seq_to_vecs(pid, seq)

    return prot_feat

def extract_pti() -> None:
    
    save_path = "./feature"
    os.makedirs(save_path, exist_ok=True)

    #drug_smi = pd.read_csv("./dataset/pollutants.csv", sep=",")
    drug_smi = pd.read_csv("./dataset/pollutants.csv", sep=",", usecols=['ChemicalName', 'SMILES'])

    tar_seq = pd.read_csv("./dataset/proteins.csv", sep=",", usecols=['UniProtKB', 'Sequence'])
    #tar_seq = pd.read_csv("./uniprotkb_Human_AND_model_organism_9606_2025_07_20.tsv", sep="\t", usecols=['TargetUniProtID', 'Sequence'])

    #drug_smi.columns = ["ChemicalName", "SMILES"]
    #tar_seq.columns = ["TargetUniProtID", "Sequence"]

    print(f"Extracting compound features ...")
    comp_feat = cal_comp_feat(drug_smi, bermol_model_path)
    with open(save_path + "compound_features.pkl", "wb") as f:
        pickle.dump(comp_feat, f)

    print(f"Extracting protein features ...")
    prot_feat = cal_prot_feat(tar_seq)
    with open(save_path + "protein_features.pkl", "wb") as f:
        pickle.dump(prot_feat, f)


if __name__ == "__main__":
    bermol_model_path = "./BerMolModel_base.pkl"
    extract_pti()
