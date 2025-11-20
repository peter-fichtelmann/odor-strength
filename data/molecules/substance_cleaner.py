from .dot_correction import dot_correction

class SubstanceCleaner:
    def remove_metal_ion_substances(self, df, smiles_col_name):
        metal_pattern = r'\[(?![^]]*(?:@H|C@|nH|N\+|O-))[^]]*\]'
        df = df.loc[~df[smiles_col_name].str.contains(metal_pattern, regex=True, na=False)]
        return df
    
    def remove_F_Br_I_substances(self, df, smiles_col_name):
        halogen_pattern = r'F|Br|I'
        df = df.loc[~df[smiles_col_name].str.contains(halogen_pattern, regex=True, na=False)]
        return df
    
    def remove_wrong_compound_smiles(self, df, name_col_name):
        df = df.loc[~df[name_col_name].isin(['carob bean absolute', 'galbanum resinoid'])]
        return df
    
    def correct_smiles_dots(self, df, name_col_name, smiles_col_name):
        df.loc[df[name_col_name].isin(dot_correction.keys()), smiles_col_name] = df[name_col_name].map(dot_correction)
        df = df.explode(smiles_col_name, ignore_index=True)
        return df
    
    def remove_smiles_dots_compounds(self, df, smiles_col_name):
        df = df.loc[~df[smiles_col_name].str.contains(r'\.', regex=True, na=False)]
        return df
