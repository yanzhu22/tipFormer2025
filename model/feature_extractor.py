from transformers import AutoTokenizer, AutoModel
import torch

class FeatureExtractor:
    def __init__(self, chem_model_name="seyonec/PubChem10M_SMILES_BPE_450k", protein_model_name="Rostlab/prot_bert"):
        self.chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
        self.chem_model = AutoModel.from_pretrained(chem_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
        self.protein_model = AutoModel.from_pretrained(protein_model_name)

    def extract_chem_features(self, smiles):
        inputs = self.chem_tokenizer(smiles, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = self.chem_model(**inputs)
        return outputs.last_hidden_state

    def extract_protein_features(self, protein_seq):
        inputs = self.protein_tokenizer(protein_seq, padding='max_length', truncation=True, max_length=1024, return_tensors='pt')
        with torch.no_grad():
            outputs = self.protein_model(**inputs)
        return outputs.last_hidden_state
