import json
from rdkit import Chem
import requests
import os
from bs4 import BeautifulSoup
from .html_getter import get_static_html

class SmilesCanonicalizer:
    def canonicalize_smiles(self, raw_smiles_list):
        canonical_smiles_list = []
        for smiles in raw_smiles_list:
            if not smiles or smiles == 'None':
                canonical_smiles_list.append(None)
            else:
                try:
                    mol = Chem.MolFromSmiles(smiles)  # Parse smiles_list into an RDKit molecule
                    if mol:
                        # Draw.MolToImage(mol)  # Generate 2D coordinates image for the molecule
                        canonical_smiles_list.append(Chem.MolToSmiles(mol, canonical=True))  # Generate canonical_smiles_list
                    else:
                        canonical_smiles_list.append(None)  # Handle invalid smiles_list strings
                except Exception as e:
                    print(f"Error processing smiles_list '{smiles}': {e}")
                    canonical_smiles_list.append(None)
        return canonical_smiles_list


class SmilesConverter:
    def cas_to_pubchem_smiles(self, cas):
        try:
            # Search PubChem by CAS
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                properties = data["PropertyTable"]["Properties"]
                if properties:
                    smiles = properties[0].get("CanonicalSMILES", cas)
                else:
                    return None
            else:
                return None
        except requests.exceptions.RequestException:
            return None
        return smiles
        
    def cas_to_goodscents_smiles(self, cas):
        try:
            smiles_url = f'http://www.thegoodscentscompany.com/opl/{cas}.html'
            html = get_static_html(smiles_url)
            soup = BeautifulSoup(html, "html.parser")
            # print(soup.find(string='SMILES :'))
            div_smiles = soup.find(string='SMILES :').parent
            # div_smiles = soup.find(string=lambda text: "SMILES" in text if text else False).parent
            smiles = div_smiles.find(class_='mrado1').get_text()
        except:
            return None
        return smiles

    def cas_to_cas_smiles(self, cas):
        try:
            smiles_url = f'https://commonchemistry.cas.org/detail?cas_rn={cas}'
            html = get_static_html(smiles_url)
            soup = BeautifulSoup(html, "html.parser")
            smiles = soup.find(string='SMILES').parent.next_sibling.next_sibling.get_text()
        except:
            return None     
        return smiles

    def get_smiles_list(self, cas_list, cas_to_smiles):
        smiles_list = []
        for cas in cas_list:
            if not cas or cas == 'None':
                smiles = None
            else:
                smiles = cas_to_smiles(cas)
            smiles_list.append(smiles)
        return smiles_list

    def cas_to_smiles_converter(self, cas_list, converter):
        if converter == 'goodscents':
            smiles_list = self.get_smiles_list(cas_list, self.cas_to_goodscents_smiles)
        elif converter == 'pubchem':
            smiles_list = self.get_smiles_list(cas_list, self.cas_to_pubchem_smiles)
        elif converter == 'cas':
            smiles_list = self.get_smiles_list(cas_list, self.cas_to_cas_smiles)
        else:
            raise ValueError(f"Converter '{converter}' not supported")
        return smiles_list
