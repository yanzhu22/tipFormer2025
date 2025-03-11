import torch
from feature_extractor import FeatureExtractor
from model import TipFormer

def predict(smiles, protein_seq):
    feature_extractor = FeatureExtractor()
    chem_emb = feature_extractor.extract_chem_features(smiles)
    protein_emb = feature_extractor.extract_protein_features(protein_seq)

    model = TipFormer()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    with torch.no_grad():
        output = model(chem_emb, protein_emb).squeeze()
        prediction = (output >= 0.5).item()
    return prediction

if __name__ == "__main__":
    smiles = "O=C(O)c1ccccc1"
    protein_seq = "MAEGEITTFTALTEKFQNKALGPGADLQ"
    result = predict(smiles, protein_seq)
    print(f"Prediction: {result}")
  
