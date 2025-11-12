![abstract](abstract.png)
# tipFormer
Transformer-Based Pollutant-Protein Interaction Analysis Prioritizes Airborne Components with Potential Adverse Health Effects
![overview](tipformerFigure1.jpg)
This project implements a TipFormer model for predicting pollutant-protein interactions. It includes functions such as feature extraction using pre-trained language models, model definition, training, and prediction for new samples. The following is a detailed usage guide and an introduction to the project structure.

## Project Structure
```
.
├── feature_extractor.py  # Feature extraction using pre-trained language models
├── model.py              # Model definition
├── train.py              # Model training
├── predict.py            # Prediction for new samples
├── best_model.pth        # Trained model weights (generated after training)
└── README.md             # This instruction file
```

## Environment Requirements
Make sure you have installed the following Python libraries:
- `torch`
- `transformers`
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`

You can install them using the following command:
```bash
pip install torch transformers pandas numpy matplotlib scikit-learn
```

## Data Preparation
The dataset used in this project can be obtained from the [T3DB database](http://www.t3db.ca) <sup>[1]</sup>.
Create a file named `air_pollution_data.csv` which should contain three columns: `smiles`, `protein_sequence`, and `label`. An example data format is as follows:
```csv
smiles,protein_sequence,label
O=C(O)c1ccccc1,MAEGEITTFTALTEKFQNKALGPGADLQ,1
CCOc1ccccc1,MAEGEITTFTALTEKFQNKALGPGADLQ,0
```

## Code Usage Instructions

### 1. Feature Extraction using Pre-trained Language Models (`feature_extractor.py`)
This file defines a `FeatureExtractor` class for extracting features from SMILES strings and protein sequences. It uses the pre-trained ChemBERTa and ProtBert models.
```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
chem_emb = extractor.extract_chem_features("O=C(O)c1ccccc1")
protein_emb = extractor.extract_protein_features("MAEGEITTFTALTEKFQNKALGPGADLQ")
```

### 2. Model Definition (`model.py`)
Defines the structure of the `TipFormer` model, including feature encoders for chemical molecules and proteins, a cross-attention mechanism, and a prediction head.
```python
from model import TipFormer

model = TipFormer()
```

### 3. Model Training (`train.py`)
Run this file to train the model. During the training process, validation will be performed, and the model weights with the minimum validation loss will be saved to the `best_model.pth` file.
```bash
python train.py
```

### 4. Prediction for New Samples (`predict.py`)
Use the trained model to predict new SMILES strings and protein sequences.
```bash
python predict.py
```

## Notes
- Since the pre-trained models are relatively large, the model files will be automatically downloaded during the first run. Please ensure a stable network connection.
- The length of the input sequence will affect memory usage. It is recommended to adjust the `max_length` parameter according to your hardware configuration.
- The training time may be long. It is recommended to use GPU acceleration. You can add the following code to use the GPU:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
And move the data to the corresponding device during data processing and model inference.

## Scalability
You can adjust the model parameters, data pre-processing flow, and evaluation metrics according to your actual needs. For example, modify the `hidden_dim` and `num_heads` parameters in `model.py`, or adjust the number of training epochs and the learning rate in `train.py`.

## Contribution
If you find any problems or have suggestions for improvement, please feel free to contact us.

## Acknowledgements

This project uses feature extraction logic inspired by / adapted from the `extract_feature.py` script in the [DTIAM](https://github.com/CSUBioGroup/DTIAM) repository (CSUBioGroup, 2023). We gratefully acknowledge their open-source contribution.

If you use this code in your research, please also consider citing the original DTIAM work:

> CSUBioGroup. (2023). *DTIAM: A unified framework for predicting drug-target interactions, binding affinities and activation/inhibition mechanisms*. GitHub repository. https://github.com/CSUBioGroup/DTIAM

## References
[1] Wishart, D., Arndt, D., Pon, A., Sajed, T., Guo, A. C., Djoumbou, Y., Knox, C., Wilson, M., Liang, Y., Grant, J., Liu, Y., Goldansaz, S. A., & Rappaport, S. M. (2015). T3DB: the toxic exposome database. Nucleic acids research, 43(Database issue), D928–D934. https://doi.org/10.1093/nar/gku1004
