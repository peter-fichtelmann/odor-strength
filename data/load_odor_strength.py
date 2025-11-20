import pandas as pd
import numpy as np
from .odor_strength_data_cleaner import OdorStrengthDataCleaner
from models.molecule_encoder import MorganFp
from .molecules.substance_cleaner import SubstanceCleaner

import networkx as nx
from tqdm import tqdm

substance_cleaner = SubstanceCleaner()


def get_odor_strength_data_cleaner(df_pubchem, unified_labels):
    odor_strength_data_cleaner = OdorStrengthDataCleaner()
    odor_strength_data_cleaner.clean_pubchem_odor_strength(df_pubchem)
    odor_strength_data_cleaner.clean_odor_strength_goodscents(unified_labels)
    odor_strength_data_cleaner.compare_odor_strength_datasets()
    return odor_strength_data_cleaner

def load_odor_strength(df_goodscents, df_pubchem, target_dataset):
    odor_strength_data_cleaner = get_odor_strength_data_cleaner(df_pubchem, df_goodscents)
    odor_strength_data_cleaner.merge_odor_strength_datasets()
    df_odor_strength = odor_strength_data_cleaner.odor_strength
    category_dict = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 3}
    print('Molecules with very high odor strength')
    print(df_odor_strength[df_odor_strength['odor_strength'] == 'very high'][['name', 'cas', 'canonical_smiles']].to_latex())
    df_odor_strength.loc[:,'numerical_strength'] = np.array([category_dict[odor_strength] for odor_strength in df_odor_strength['odor_strength'].tolist()])
    df_odor_strength['has_odor'] = (df_odor_strength['numerical_strength'] > 0) * 1
    print('Number of SMILES with dots:', df_odor_strength['canonical_smiles'].str.contains('\.', regex=True).sum())
    df_odor_strength = substance_cleaner.remove_metal_ion_substances(df_odor_strength, 'canonical_smiles')
    df_odor_strength = substance_cleaner.remove_F_Br_I_substances(df_odor_strength, 'canonical_smiles')
    df_odor_strength = substance_cleaner.correct_smiles_dots(df_odor_strength, 'name', 'canonical_smiles')
    df_odor_strength = substance_cleaner.remove_smiles_dots_compounds(df_odor_strength, 'canonical_smiles')
    df_odor_strength = df_odor_strength[~df_odor_strength['canonical_smiles'].duplicated(keep='first')]
    

    def check_for_outlier_molecules(smiles_list_1, smiles_list_2):
        morgan_fingerprint = MorganFp(radius=3, fpSize=2048)
        morgan_fingerprints_1 = morgan_fingerprint.encode(smiles_list_1)
        morgan_fingerprints_2 = morgan_fingerprint.encode(smiles_list_2)
        # morgan_similarities = {smiles_1: np.sum(np.logical_and(np.array(morgan_fingerprints_2), fp1), axis=1) / len(fp1) for smiles_1, fp1 in tqdm(zip(smiles_list_1, morgan_fingerprints_1))}
        morgan_similarities = {smiles_1: np.sum(np.logical_and(np.array(morgan_fingerprints_2), fp1), axis=1) / np.sum(np.logical_or(np.array(morgan_fingerprints_2), fp1), axis=1) for smiles_1, fp1 in tqdm(zip(smiles_list_1, morgan_fingerprints_1))}

        # morgan_similarities = {smiles_2: [jaccard_score(fp1, fp2) for smiles_1, fp1 in zip(smiles_list_1, morgan_fingerprints_1)] for smiles_2, fp2 in tqdm(zip(smiles_list_2, morgan_fingerprints_2))}
        df_morgan_similarities = pd.DataFrame(morgan_similarities, index=smiles_list_2)
        return df_morgan_similarities
    
    def filter_formulas(df_odor_strength, df_compare_smiles):
        df_morgan_similarities = check_for_outlier_molecules(df_compare_smiles.tolist(), df_odor_strength['canonical_smiles'].tolist())
        filtered_smiles = df_morgan_similarities[df_morgan_similarities.max(axis=1) > 0.2].index.tolist()
        df_odor_strength = df_odor_strength[df_odor_strength['canonical_smiles'].isin(filtered_smiles)]
        return df_odor_strength
    
    df_odor_strength = filter_formulas(df_odor_strength, df_odor_strength[df_odor_strength['source'] == target_dataset]['canonical_smiles'])
    smiles_list = df_odor_strength['canonical_smiles'][~df_odor_strength['canonical_smiles'].duplicated(keep='first')].tolist()
    df_morgan_similarities_odor_strength = check_for_outlier_molecules(smiles_list, smiles_list)
    dist_matrix = df_morgan_similarities_odor_strength.values

    def get_similar_groups(dist_matrix, threshold):
        # Create a graph where nodes are points and edges are below threshold
        n_points = dist_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_matrix[i, j] >= threshold:
                    G.add_edge(i, j)

        components = list(nx.connected_components(G))
        groups = np.zeros(n_points, dtype=int)
        for cluster_id, nodes in enumerate(components, start=0):
            for node in nodes:
                groups[node] = cluster_id
        print("Connected components (clusters):", components)
        print(np.max([len(component) for component in components]), "cluster max size")
        print(len(groups))
        print("Cluster groups:", groups)
        return groups

    groups = get_similar_groups(dist_matrix, 0.8)
    return df_odor_strength, groups