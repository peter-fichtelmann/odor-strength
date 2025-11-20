from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from .predictors import Predictor


class RuleOfThreePredictor(Predictor):
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray|None = None, y_val:np.ndarray|None = None, **kwargs):
        pass

    def predict(self, X: np.ndarray):
        smiles_list = X.tolist()
        prediction = np.zeros((len(smiles_list)))
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mw = Descriptors.MolWt(mol)
            n_hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
            if mw > 30 and mw < 300 and n_hetero_atoms < 3:
                prediction[i] = 1
        return prediction

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def reset(self):
        pass