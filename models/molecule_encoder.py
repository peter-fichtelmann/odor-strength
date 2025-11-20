from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, MACCSkeys, Descriptors
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import gc
import os
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data import build_dataloader
from chemprop.models.model import MPNN
from chemprop.nn.message_passing.base import BondMessagePassing, AtomMessagePassing
from chemprop.nn.agg import MeanAggregation, SumAggregation, NormAggregation, AttentiveAggregation
from chemprop.nn import BinaryClassificationFFN



class MoleculeEncoder(ABC):
    """
    Abstract base class for encoding molecules.
    """
    @abstractmethod
    def encode(self, smiles_list):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class MoleculeEncoderBase(MoleculeEncoder):
    """Base class for molecule encoders that provides common functionality.

     Methods:
        encode_single_mol(mol): Abstract method to encode a single molecule.
        smiles_to_mol(smiles): Converts a SMILES string to an RDKit molecule object.
        smiles_list_to_mol_list(smiles_list): Converts a list of SMILES strings to a list of RDKit molecule objects.
        encode(smiles_list): Encodes a list of SMILES strings into a numpy array of encodings.
    """
    def encode_single_mol(self, mol):
        pass

    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    
    def smiles_list_to_mol_list(self, smiles_list: list[str]) -> list:
        return [self.smiles_to_mol(smiles) for smiles in smiles_list]

    def encode(self, smiles_list) -> np.ndarray:
        mol_list = self.smiles_list_to_mol_list(smiles_list)
        return np.array([self.encode_single_mol(mol) for mol in mol_list])
    
    def save(self, path: str):
        """Saves the encoder to the specified path."""
        import joblib
        joblib.dump(self, path)

    def load(self, path: str):
        """Loads the encoder from the specified path."""
        import joblib
        loaded_encoder = joblib.load(path)
        if not isinstance(loaded_encoder, MoleculeEncoderBase):
            raise ValueError(f"Loaded object is not an instance of MoleculeEncoderBase: {type(loaded_encoder)}")
        return loaded_encoder


# class EncodingCombiner(MoleculeEncoderBase):
#     def __init__(self, *encoders):
#         self.encoders = encoders

#     def encode_single_mol(self, mol):
#         return np.array([encoder.encode_single_mol(mol) for encoder in self.encoders])
    

class NativeEncoder(MoleculeEncoderBase):
    """No encoding, just returns the list of SMILES as a numpy array."""
    def __init__(self):
        super().__init__()

    def encode(self, smiles_list: list[str]) -> np.ndarray:
        return np.array(smiles_list)


class Fingerprints(MoleculeEncoderBase):
    """Base class for generating molecular fingerprints using RDKit.
    
    Parameters:
        count: If True, returns the count of each molecular subgroup in the fingerprint. Otherwise, returns the presence/absence of molecular subgroups."""
    def __init__(self, count: bool):
        super().__init__()
        self.count = count  # To be defined in subclasses

    def encode_single_mol(self, mol) -> np.ndarray|None:
        if self.count:
            return np.array(self.fingerprint_generator.GetCountFingerprintAsNumPy(mol)) if mol else None
        return np.array(self.fingerprint_generator.GetFingerprint(mol)) if mol else None
    
    def save(self, path: str):
        """Saves the encoder to the specified path, handling non-picklable fingerprint generator."""
        import joblib
        # Store the fingerprint generator parameters instead of the object itself
        state = {
            'count': self.count,
            'init_params': getattr(self, '_init_params', {})
        }
        joblib.dump(state, path)
    
    def load(self, path: str):
        """Loads the encoder from the specified path and recreates the fingerprint generator."""
        import joblib
        state = joblib.load(path)
        # Recreate the encoder with the saved parameters
        encoder = self.__class__(**state['init_params'])
        encoder.count = state['count']
        return encoder


class MorganFp(Fingerprints):
    def __init__(self, count: bool=False, atomInvariantsGenerator: bool=False, **kwargs):
        super().__init__(count)
        if atomInvariantsGenerator:
            invgen = AllChem.GetMorganFeatureAtomInvGen()
            kwargs['atomInvariantsGenerator'] = invgen
        self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(**kwargs)
    

# class FeatureMorganFp(Fingerprints):
#     def __init__(self, count: bool=False, **kwargs):
#         super().__init__(count)
#         invgen = AllChem.GetMorganFeatureAtomInvGen()
#         self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(atomInvariantsGenerator=invgen, **kwargs)
    

class RDKitFp(Fingerprints):
    def __init__(self, count: bool=False, atomInvariantsGenerator=False, **kwargs):
        super().__init__(count)
        if atomInvariantsGenerator:
            invgen = AllChem.GetRDKitAtomInvGen()
            kwargs['atomInvariantsGenerator'] = invgen
        self.fingerprint_generator = rdFingerprintGenerator.GetRDKitFPGenerator(**kwargs)

    
class TopologicalTorsionFp(Fingerprints):
    def __init__(self, count: bool=False, **kwargs):
        super().__init__(count)
        self.fingerprint_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(**kwargs)


class AtomPairFp(Fingerprints):
    def __init__(self, count: bool=False, atomInvariantsGenerator=False, **kwargs):
        super().__init__(count)
        if atomInvariantsGenerator:
            invgen = AllChem.GetAtomPairAtomInvGen()
            kwargs['atomInvariantsGenerator'] = invgen
        self.fingerprint_generator = rdFingerprintGenerator.GetAtomPairGenerator(**kwargs)


class MACCSKeysFp(Fingerprints):
    def __init__(self, count: bool=False, **kwargs):
        super().__init__(count)
    
    def encode_single_mol(self, mol):
        return np.array(MACCSkeys.GenMACCSKeys(mol)) if mol else None

    
class RDKitDescriptors(MoleculeEncoderBase):
    def __init__(self):
        super().__init__()

    def encode_single_mol(self, mol: Chem.Mol):
        if mol is None:
            return None
        return Descriptors.CalcMolDescriptors(mol)
    
    def encode(self, smiles_list: list[str]) -> pd.DataFrame:
        mol_list = self.smiles_list_to_mol_list(smiles_list)
        return pd.DataFrame([self.encode_single_mol(mol) for mol in mol_list])


class ChemBerta(MoleculeEncoderBase):
    """Encodes SMILES strings using the ChemBERTa model from Hugging Face.
    
    Parameters:
        target_layer: The layer of the model to extract features from.
        pooling: {'mean', 'cls'} The pooling method to apply to the features.
        batch_size: The batch size for processing SMILES strings.
    """
    def __init__(self, target_layer: int, pooling: str = 'mean', encoded_database_path: str | None = None, batch_size: int = 64): # up to 6 target layers
        # Store initialization parameters for save/load
        self._init_params = {
            'target_layer': target_layer,
            'pooling': pooling,
            'encoded_database_path': encoded_database_path,
            'batch_size': batch_size
        }
        self.batch_size = batch_size
        
        if not isinstance(encoded_database_path, str) or not os.path.exists(encoded_database_path):
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.memory_reserved() > 8*1e9:
                print('Warning: more than 8 GB gpu memory reserved')  # Check if more than 8GB is reserved
            self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            self.target_layer = target_layer
            self.pooling = pooling
            self.encoded_database = None
            if self.model.config.num_hidden_layers < target_layer:
                raise ValueError(f"Target layer {target_layer} exceeds the number of hidden layers in the model: {self.model.config['num_hidden_layers']}.")
        else:
            self.encoded_database = pd.read_csv(encoded_database_path, index_col=0)
            self.target_layer = target_layer
            self.pooling = pooling

    def encode(self, smiles_list: list[str]) -> np.ndarray:
        if not isinstance(self.encoded_database, pd.DataFrame):
            # Process in batches to manage memory
            all_encodings = []
            for i in range(0, len(smiles_list), self.batch_size):
                batch_smiles = smiles_list[i:i + self.batch_size]
                if len(batch_smiles) > 0:
                    batch_encoding = self._encode_batch(batch_smiles)
                    all_encodings.append(batch_encoding)
                
            return np.vstack(all_encodings)
        else:
            encoding = self.encoded_database.loc[smiles_list].values
            return encoding
    
    def _encode_batch(self, batch_smiles: list[str]) -> np.ndarray:
        """Encode a single batch of SMILES strings."""
        for smiles in batch_smiles:
            if not isinstance(smiles, str):
                raise ValueError(f'Provided non string SMILES: {smiles}')
        tokenized_smiles = self.tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized_smiles['input_ids'].to(self.model.device)  # Move input_ids to the same device as the model
        attention_mask = tokenized_smiles['attention_mask'].to(self.model.device)  # Important for masking padded values
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # outputs = self.model(input_ids)
            # print(outputs.last_hidden_state.shape)
            if self.pooling == 'cls':
                encoding = outputs.hidden_states[self.target_layer][:, 0, :]
                # encoding = outputs.last_hidden_state[:, 0, :]
            elif self.pooling == 'mean':
                encoding = outputs.hidden_states[self.target_layer].mean(dim=1)
                # encoding = outputs.last_hidden_state.mean(dim=1)
                # Use attention mask to correctly compute the mean only over non-padded tokens
                expanded_mask = attention_mask.unsqueeze(-1).expand(outputs.hidden_states[self.target_layer].shape)
                sum_embeddings = (outputs.hidden_states[self.target_layer] * expanded_mask).sum(dim=1)
                sum_mask = expanded_mask.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
                encoding = sum_embeddings / sum_mask
            else:
                raise ValueError("Invalid pooling method. Choose from 'cls', 'mean'.")
        
        return encoding.cpu().numpy()


class ChemPropMPNN(MoleculeEncoderBase):
    def __init__(self,
                predictor: object
                 ):
        self.predictor = predictor

    def encode(self, smiles_list: list[str]) -> np.ndarray:
        if len(smiles_list) == 1:
            smiles_list.append('C')  # Add a dummy SMILES to avoid empty batch
        data_points = [MoleculeDatapoint.from_smi(smiles) for smiles in smiles_list]
        dataset = MoleculeDataset(data_points, featurizer=self.predictor.featurizer)
        dataloader = build_dataloader(dataset, batch_size=self.predictor.batch_size, shuffle=False)
        with torch.no_grad():
            fingerprints = [
                self.predictor.model.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)
                for batch in dataloader
            ]
            fingerprints = torch.cat(fingerprints, 0)
        fingerprints = fingerprints.cpu().numpy()
        if smiles_list[1] == 'C':
            fingerprints = fingerprints[0]  # Remove the dummy SMILES encoding
        return fingerprints





