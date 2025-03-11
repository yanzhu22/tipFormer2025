import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor
from model import TipFormer

class AirPollutionDataset(Dataset):
    def __init__(self, smiles_list, protein_seqs, labels):
        self.smiles_list = smiles_list
        self.protein_seqs = protein_seqs
        self.labels = labels
        self.feature_extractor = FeatureExtractor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        chem_emb = self.feature_extractor.extract_chem_features(self.smiles_list[idx])
        protein_emb = self.feature_extractor.extract_protein_features(self.protein_seqs[idx])
        return {
            'chem_emb': chem_emb.squeeze(),
            'protein_emb': protein_emb.squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['chem_emb'], batch['protein_emb']).squeeze()
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch['label'].size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['chem_emb'], batch['protein_emb']).squeeze()
                loss = criterion(outputs, batch['label'])
                val_loss += loss.item() * batch['label'].size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_dataset = AirPollutionDataset(
        smiles_list=train_data['smiles'].tolist(),
        protein_seqs=train_data['protein_sequence'].tolist(),
        labels=train_data['label'].tolist()
    )
    val_dataset = AirPollutionDataset(
        smiles_list=val_data['smiles'].tolist(),
        protein_seqs=val_data['protein_sequence'].tolist(),
        labels=val_data['label'].tolist()
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = TipFormer()
    train_model(model, train_loader, val_loader)
