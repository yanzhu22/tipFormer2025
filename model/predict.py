# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import tipFormer

# --------------------------- Dataset Definition --------------------------- #
class PredictionDataset(Dataset):
    def __init__(self, pairs_df, comp_feat, prot_feat):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.comp_feat = comp_feat
        self.prot_feat = prot_feat
        
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        cid = row['ChemicalName']
        pid = row['TargetUniProtID']
        
        # Fetch feature vectors
        comp_vector = self.comp_feat[cid].astype(np.float32)
        prot_vector = self.prot_feat[pid].astype(np.float32)
        
        return {
            'compound': comp_vector,
            'protein': prot_vector,
            'cid': cid,
            'pid': pid,
            'sequence': row['Sequence']
        }

# --------------------------- Prediction Function --------------------------- #
def predict_interactions(compound_name, protein_df, comp_feat, prot_feat, 
                         model, device, batch_size=256):
    """Predict interactions between a single compound and all proteins."""
    # Build prediction pairs
    prediction_pairs = pd.DataFrame({
        'ChemicalName': [compound_name] * len(protein_df),
        'TargetUniProtID': protein_df['TargetUniProtID'],
        'Sequence': protein_df['Sequence']
    })
    
    # Create dataset
    dataset = PredictionDataset(prediction_pairs, comp_feat, prot_feat)
    
    # Create dataloader (larger batch size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Prediction
    model.eval()
    all_pids = []
    all_sequences = []
    all_preds = []
    all_cids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {compound_name}"):
            # Move to device
            compounds = torch.tensor(batch['compound']).to(device)
            proteins = torch.tensor(batch['protein']).to(device)
            
            # Forward pass
            logits = model(compounds, proteins)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Collect results
            all_pids.extend(batch['pid'])
            all_sequences.extend(batch['sequence'])
            all_preds.extend(probs)
            all_cids.extend(batch['cid'])
    
    # Format results
    results_df = pd.DataFrame({
        'ChemicalID': all_cids,
        'ProteinID': all_pids,
        'ProteinSequence': all_sequences,
        'PredictionScore': all_preds,
        'PredictionLabel': [1 if p >= 0.5 else 0 for p in all_preds]
    }).sort_values(by='PredictionScore', ascending=False).reset_index(drop=True)
    
    return results_df

# --------------------------- Main Function --------------------------- #
def main():
    # Paths
    model_path = "./model/best_model.pth"
    chemicals_path = "./your_chemicals.csv"
    proteins_path = "./your_protens.csv"
    comp_feat_path = "./your_compound_features.pkl"
    prot_feat_path = "./your_proteins_features.pkl"
    output_dir = "./prediction_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("Loading data...")
    chemicals_df = pd.read_csv(chemicals_path)
    proteins_df = pd.read_csv(proteins_path, sep='\t')  # TSV file
    
    # Load features
    with open(comp_feat_path, 'rb') as f:
        comp_feat = pickle.load(f)
    with open(prot_feat_path, 'rb') as f:
        prot_feat = pickle.load(f)
    
    # Validate feature keys
    sample_chem = chemicals_df.iloc[0]['ChemicalName']
    if sample_chem not in comp_feat:
        print(f"Warning: compound '{sample_chem}' not found in feature dict")
        # Try fallback keys
        alt_keys = [k for k in comp_feat.keys() if str(k) == str(sample_chem)]
        if alt_keys:
            print(f"Using fallback key: {alt_keys[0]} instead of {sample_chem}")
            chemicals_df['ChemicalName'] = chemicals_df['ChemicalName'].apply(
                lambda x: alt_keys[0] if x == sample_chem else x
            )
        else:
            print(f"Error: no fallback key found for compound '{sample_chem}'")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    
    # Initialize model with saved config
    config = checkpoint['config']
    model = tipFormer(
        comp_dim=config['comp_dim'],
        prot_dim=config['prot_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Parameter validation
    print("\nModel parameter validation:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Config: {config}")
    
    # Predict for each compound
    print(f"\nStarting predictions for {len(chemicals_df)} compounds...")
    for _, row in chemicals_df.iterrows():
        compound_name = row['ChemicalName']
        print(f"\nProcessing compound: {compound_name}")
        
        # Check compound feature existence
        if compound_name not in comp_feat:
            print(f"Error: features for compound {compound_name} not found, skipping...")
            continue
        
        # Predict
        results_df = predict_interactions(
            compound_name=compound_name,
            protein_df=proteins_df,
            comp_feat=comp_feat,
            prot_feat=prot_feat,
            model=model,
            device=device,
            batch_size=256
        )
        
        # Save results
        safe_name = "".join(c for c in compound_name if c.isalnum() or c in " _-")
        output_path = os.path.join(output_dir, f"{safe_name}_predictions.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Print top-5 predictions
        top_preds = results_df.head(5)
        print("Top 5 predictions:")
        print(top_preds[['ProteinID', 'PredictionScore']])
    
    print("\nAll predictions completed!")

if __name__ == "__main__":
    main()
