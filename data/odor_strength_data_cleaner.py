


import pandas as pd
import re
import numpy as np

class OdorStrengthDataCleaner:
    def __init__(self):
        self.pubchem_strength = pd.DataFrame()
        self.goodsents_strength = pd.DataFrame()
        self.odor_strength = pd.DataFrame()

    def clean_pubchem_odor_strength(self, df_pubchem):
        # string = 'There is no odor. So it is odorless or odourless. A very faint or very weak odor.'
        odorless_pattern = r'odou?rless|no .* odou?r|very faint|very weak|very mild'
        low_odor_pattern = r'faint|weak'
        high_odor_pattern = r'strong|intense|powerful'
        very_high_odor_pattern = r'very strong|very intense|very powerful|very pungent|very aromatic'
        # odorless_matches = re.findall(odorless_pattern, string, flags=re.IGNORECASE)
        # print(odorless_matches)
        df_pubchem_odorless = df_pubchem[(df_pubchem['raw_description'].str.contains(odorless_pattern, regex=True, flags=re.IGNORECASE)) & (df_pubchem['canonical_smiles'].notna())]
        df_pubchem_odorless['odor_strength'] = 'none'
        df_pubchem_low_odor = df_pubchem[(df_pubchem['raw_description'].str.contains(low_odor_pattern, regex=True, flags=re.IGNORECASE)) & (df_pubchem['canonical_smiles'].notna())]
        df_pubchem_low_odor['odor_strength'] = 'low'
        df_pubchem_high_odor = df_pubchem[(df_pubchem['raw_description'].str.contains(high_odor_pattern, regex=True, flags=re.IGNORECASE)) & (df_pubchem['canonical_smiles'].notna())]
        df_pubchem_high_odor['odor_strength'] = 'high'
        df_pubchem_very_high_odor = df_pubchem[(df_pubchem['raw_description'].str.contains(very_high_odor_pattern, regex=True, flags=re.IGNORECASE)) & (df_pubchem['canonical_smiles'].notna())]
        df_pubchem_very_high_odor['odor_strength'] = 'very high'
        concat_list = [
            df_pubchem_odorless,
            df_pubchem_low_odor.loc[~df_pubchem_low_odor.index.isin(df_pubchem_odorless.index)],
            df_pubchem_high_odor.loc[~df_pubchem_high_odor.index.isin(df_pubchem_odorless.index.tolist() + df_pubchem_low_odor.index.tolist())],
            df_pubchem_very_high_odor.loc[~df_pubchem_very_high_odor.index.isin(df_pubchem_odorless.index.tolist() + df_pubchem_low_odor.index.tolist() + df_pubchem_high_odor.index.tolist())],
        ]
        self.pubchem_strength = pd.concat(concat_list, axis=0)
        self.pubchem_strength = self.pubchem_strength[['Name', 'canonical_smiles', 'raw_description', 'odor_strength']].rename(columns={'Name': 'name'})
        self.pubchem_strength['source'] = 'pubchem'
        self.pubchem_strength = self.pubchem_strength[~self.pubchem_strength.duplicated(subset=['canonical_smiles'], keep='first')]
        df_pubchem_very_high_odor[['Name', 'canonical_smiles', 'raw_description', 'odor_strength']]

    def clean_odor_strength_goodscents(self, unified_labels):
        self.goodsents_strength = unified_labels[unified_labels['odor_strength'].notna() & (unified_labels['canonical_smiles'].notna())]
        df_odorless_descriptions = unified_labels[unified_labels['raw_description'].str.contains('odorless').fillna(False) & (unified_labels['canonical_smiles'].notna())]
        df_odorless_descriptions['odor_strength'] = 'none'
        self.goodsents_strength = pd.concat([self.goodsents_strength, df_odorless_descriptions[~df_odorless_descriptions['canonical_smiles'].isin(self.goodsents_strength['canonical_smiles'])]])
        self.goodsents_strength = self.goodsents_strength[~self.goodsents_strength.duplicated(subset=['canonical_smiles'], keep='first')]
        self.goodsents_strength = self.goodsents_strength[['name', 'cas', 'canonical_smiles', 'odor_strength', 'source', 'logP (o/w):', 'Vapor Pressure:', 'Boiling Point:', 'Boiling Point:_2']]
        self.goodsents_strength['odor_strength'] = self.goodsents_strength['odor_strength'].str.split(' ,').str[0]
        self.goodsents_strength = self.goodsents_strength[~self.goodsents_strength['name'].isin(['carob bean absolute', 'galbanum resinoid'])]


    def compare_odor_strength_datasets(self):
        df_compare = self.goodsents_strength[self.goodsents_strength['canonical_smiles'].isin(self.pubchem_strength['canonical_smiles'])]
        df_compare.index = df_compare['canonical_smiles']
        self.pubchem_strength.index = self.pubchem_strength['canonical_smiles']
        df_compare['pubchem_odor_strength'] = self.pubchem_strength.loc[df_compare.index, 'odor_strength']
        df_compare['pubchem_description'] = self.pubchem_strength.loc[df_compare.index, 'raw_description']
        return df_compare

    def merge_odor_strength_datasets(self):
        self.odor_strength = pd.concat([self.goodsents_strength, self.pubchem_strength.loc[~self.pubchem_strength.index.isin(self.goodsents_strength['canonical_smiles'])]], axis=0)

