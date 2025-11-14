import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from sklearn import metrics
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Optional, List, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# Global settings
torch.backends.cudnn.benchmark = True

# --------------------------- Custom Dataset Class --------------------------- #
class DTIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 comp_feat: Dict[str, np.ndarray], 
                 prot_feat: Dict[str, np.ndarray],
                 augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.comp_feat = comp_feat
        self.prot_feat = prot_feat
        
        sample_comp = next(iter(comp_feat.values()))
        sample_prot = next(iter(prot_feat.values()))
        self.comp_dim = sample_comp.shape[-1]
        self.prot_dim = sample_prot.shape[-1]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid = str(row['cid'])
        pid = str(row['pid'])
        
        compound = torch.tensor(self.comp_feat[cid].astype(np.float32))
        protein = torch.tensor(self.prot_feat[pid].astype(np.float32))

        return {
            'compound': compound,
            'protein': protein,
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'cid': cid,
            'pid': pid
        }
    
    def __len__(self):
        return len(self.df)


# --------------------------- Encoder Block --------------------------- #
class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


# --------------------------- tipFormer Model --------------------------- #
class tipFormer(nn.Module):
    def __init__(self, 
                 comp_dim: int, 
                 prot_dim: int, 
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.4,
                 comp_noise: float = 0.05,
                 prot_noise: float = 0.025):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_noise = comp_noise
        self.prot_noise = prot_noise
        
        # Initial linear projection (Eq. 3-4)
        self.comp_proj_init = nn.Linear(comp_dim, hidden_dim)
        self.prot_proj_init = nn.Linear(prot_dim, hidden_dim)
        
        # Encoder: 3 stacked blocks (Eq. 5)
        self.comp_encoder = nn.Sequential(
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout)
        )
        self.prot_encoder = nn.Sequential(
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout)
        )
        
        # Cross-attention (Eq. 6-9)
        self.comp_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.prot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.comp_norm = nn.LayerNorm(hidden_dim)
        self.prot_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion layer (Eq. 11)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # MLP classifier (Eq. 12-13)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def add_noise(self, x, noise_level):
        if self.training and noise_level > 0:
            noise = torch.randn_like(x) * noise_level
            return x + noise
        return x

    def forward(self, compound: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        # Feature noise injection
        compound = self.add_noise(compound, self.comp_noise)
        protein = self.add_noise(protein, self.prot_noise)
        
        # Initial projection (Eq. 3-4)
        comp_emb = self.comp_proj_init(compound)
        prot_emb = self.prot_proj_init(protein)
        
        # Encoder blocks (Eq. 5)
        comp_enc = self.comp_encoder(comp_emb)
        prot_enc = self.prot_encoder(prot_emb)
        
        # Add dummy sequence dim (B, 1, H)
        comp_seq = comp_enc.unsqueeze(1)
        prot_seq = prot_enc.unsqueeze(1)
        
        # Bidirectional cross-attention (Eq. 6-9)
        comp_attn_out, _ = self.comp_attn(query=comp_seq, key=prot_seq, value=prot_seq)
        comp_updated = self.comp_norm(comp_attn_out.squeeze(1) + comp_enc)
        
        prot_attn_out, _ = self.prot_attn(query=prot_seq, key=comp_seq, value=comp_seq)
        prot_updated = self.prot_norm(prot_attn_out.squeeze(1) + prot_enc)
        
        # Concatenate and fuse (Eq. 10-11)
        concat = torch.cat([comp_updated, prot_updated], dim=1)
        fused = self.fusion(concat)
        
        # Classifier (Eq. 12-13)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'comp_dim': self.comp_proj_init.in_features,
                'prot_dim': self.prot_proj_init.in_features,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.comp_attn.num_heads,
                'dropout': self.comp_encoder[0].dropout.p,
                'comp_noise': self.comp_noise,
                'prot_noise': self.prot_noise
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)


# --------------------------- Evaluation Function --------------------------- #
def calculate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"ROC AUC": 0.0, "PR AUC": 0.0, "Sensitivity": 0.0, "Specificity": 0.0,
                "Precision": 0.0, "Accuracy": 0.0, "MCC": 0.0, "F1 Score": 0.0}
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = metrics.auc(recall, precision)
    
    # Use 0.5 threshold for standard metrics (as in paper evaluation)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    
    return {
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_val,
        "Accuracy": accuracy,
        "MCC": mcc,
        "F1 Score": f1,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall
    }


# --------------------------- Training Function --------------------------- #
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                num_epochs: int = 150,
                patience: int = 15,
                output_dir: str = "./outputs/models",
                pos_weight: Optional[torch.Tensor] = None) -> Tuple[nn.Module, dict]:
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight if pos_weight is not None else None)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=2e-3
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=15,
        T_mult=2,
        eta_min=1e-7
    )
    
    scaler = GradScaler(enabled=True)
    
    best_val_auroc = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            compound = batch['compound'].to(device, non_blocking=True)
            protein = batch['protein'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(compound, protein)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for batch in val_loader:
                compound = batch['compound'].to(device, non_blocking=True)
                protein = batch['protein'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                logits = model(compound, protein)
                val_loss += criterion(logits, labels).item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
        val_auroc = metrics['ROC AUC']
        
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUROC={val_auroc:.4f}")
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            no_improve = 0
            model.save(os.path.join(output_dir, "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break
        
        scheduler.step()
    
    model = tipFormer.load(os.path.join(output_dir, "best_model.pth"), device)
    return model, history


# --------------------------- Testing Function --------------------------- #
def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device, output_dir: str = "./outputs/results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_labels, all_probs = [], []
    
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for batch in tqdm(test_loader, desc="Testing"):
            compound = batch['compound'].to(device, non_blocking=True)
            protein = batch['protein'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            logits = model(compound, protein)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'pred_prob': all_probs,
        'pred_label': (np.array(all_probs) >= 0.5).astype(int)
    })
    results_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
    
    print("\nTest Results:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    if 'fpr' in metrics:
        plot_roc_curve(metrics['fpr'], metrics['tpr'], output_dir)
        plot_pr_curve(metrics['recall'], metrics['precision'], output_dir)
    
    return {'metrics': metrics}


# --------------------------- Visualization Functions --------------------------- #
def plot_roc_curve(fpr, tpr, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {metrics.auc(fpr, tpr):.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(); plt.savefig(os.path.join(output_dir, 'roc_curve.png')); plt.close()

def plot_pr_curve(recall, precision, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR AUC = {metrics.auc(recall, precision):.4f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(); plt.savefig(os.path.join(output_dir, 'pr_curve.png')); plt.close()


# --------------------------- Main Function --------------------------- #
def main():
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # Load data (paths unchanged)
    data_dir = "./datasETS/"
    features_dir = os.path.join(data_dir, "features")
    with open(os.path.join(features_dir, "compound_features.pkl"), "rb") as f:
        comp_feat = pickle.load(f)
    with open(os.path.join(features_dir, "protein_features.pkl"), "rb") as f:
        prot_feat = pickle.load(f)
    
    data_dir = "./datasets/"
    train_df = pd.read_csv(os.path.join(data_dir, "train_1_1.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_1_1.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_1_1.csv"))
    
    col_mapping = {'ChemicalName': 'cid', 'PollutantName': 'cid', 'compound_id': 'cid',
                   'UniProtKB': 'pid', 'ProteinID': 'pid', 'target_id': 'pid',
                   'Label': 'label', 'affinity': 'label'}
    for df in [train_df, val_df, test_df]:
        df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns}, inplace=True)
    
    # Create datasets
    train_dataset = DTIDataset(train_df, comp_feat, prot_feat)
    val_dataset = DTIDataset(val_df, comp_feat, prot_feat)
    test_dataset = DTIDataset(test_df, comp_feat, prot_feat)
    
    # Sampler
    labels = train_df['label'].values
    pos_weight = torch.tensor([np.sum(labels == 0) / np.sum(labels == 1)], dtype=torch.float32).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = tipFormer(
        comp_dim=train_dataset.comp_dim,
        prot_dim=train_dataset.prot_dim,
        hidden_dim=256,
        num_heads=8,
        dropout=0.4,
        comp_noise=0.05,
        prot_noise=0.025
    ).to(device)
    
    # Train
    trained_model, _ = train_model(
        model, train_loader, val_loader, device,
        num_epochs=150, patience=15,
        output_dir="./outputs/models",
        pos_weight=pos_weight
    )
    
    # Test
    test_model(trained_model, test_loader, device, "./outputs/results")

if __name__ == "__main__":
    main()
