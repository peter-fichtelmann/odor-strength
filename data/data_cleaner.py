from abc import ABC, abstractmethod
import pandas as pd
from .molecules.smiles_converter import SmilesConverter, SmilesCanonicalizer
from .goodscents_one_page_parser import GoodScentsOnePageParser
import requests
import os
from bs4 import BeautifulSoup
import requests
from tqdm.auto import tqdm


class DataCleaner(ABC):
    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame()

    @abstractmethod
    def clean_molecules(self):
        pass
        

class GoodScentsDataCleaner(DataCleaner):
    def __init__(self, data: pd.DataFrame|None = None):
        super().__init__()
        self.data = data if data is not None else pd.DataFrame()
        
    def get_links(self):
        base_url = 'http://www.thegoodscentscompany.com/allprod-{}.html'
        links = []
        all_pages = ['a','b','c','d','e','f','g','h','i','jk','l','m','n','o','p','q','r','s','t','u','v','wx','y','z']
        for letter in all_pages:
            url = base_url.format(letter)
            resp = requests.get(url)
            html = resp.content
            soup = BeautifulSoup(html,"html.parser")
            for a in soup.find_all('a', href=True):
                if a['href'] == "#" and a.get('onclick'):
                    alink = str(a["onclick"])
                    alink = alink.replace("openMainWindow('","").replace("');return false;","")
                    links.append(alink)
        links = list(set(links))
        return links

    def crawl_data(self):
        links = self.get_links()
        results = []
        for link in tqdm(links):
            try:
                link = 'http://www.thegoodscentscompany.com/' + link
                results.append(GoodScentsOnePageParser().parse_one_page(link))
            except Exception as e:
                print(f'Error in {link}: {e}')
        self.data = pd.DataFrame(results)
        self.data = self.data.query('useful')
        self.data = self.data.explode(['raw_description', 'source_detail', 'name2'], ignore_index=True)
        self.data.drop(inplace=True, columns=['useful'])
        self.data['concentration in %'] = self.data['raw_description'].str.extract(r"(\d+(?:\.\d+)?\s?%(?!\s?purity))")

    def clean_molecules(self):
        self.data['raw_pubchem_smiles'] = SmilesConverter().cas_to_smiles_converter(self.data['cas'], converter='pubchem')
        self.data['raw_goodscents_smiles'] = SmilesConverter().cas_to_smiles_converter(self.data['cas'], converter='goodscents')
        self.data['raw_cas_smiles'] = SmilesConverter().cas_to_smiles_converter(self.data['cas'], converter='cas')
        self.data['raw_smiles'] = self.data['raw_cas_smiles'].fillna(self.data['raw_goodscents_smiles']).fillna(self.data['raw_pubchem_smiles'])
        self.data['canonical_smiles'] = SmilesCanonicalizer().canonicalize_smiles(self.data['raw_smiles'])


class PubChemDataCleaner(DataCleaner): 
    def __init__(self, df_odor_cids):
        self.data = df_odor_cids


    def crawl_data(self):
        """
        Crawl PubChem for odor descriptions and thresholds based on CIDs in self.data.
        """
        def find_dict_with_toc_heading(data, toc_heading):
            """
            Recursively search a nested structure (dicts/lists) and return the first dict 
            where 'TOCHeading' == toc_heading.
            """
            if isinstance(data, dict):
                if data.get('TOCHeading') == toc_heading:
                    return data
                for value in data.values():
                    result = find_dict_with_toc_heading(value, toc_heading)
                    if result:
                        return result

            elif isinstance(data, list):
                for item in data:
                    result = find_dict_with_toc_heading(item, toc_heading)
                    if result:
                        return result

            return None  # Not found

        def get_odor_data_from_dict(pubchem_dict, toc_heading):
            odor_dict = find_dict_with_toc_heading(pubchem_dict, toc_heading)
            odor_list = []
            odor_reference_list = []

            if odor_dict and 'Information' in odor_dict:
                for data in odor_dict['Information']:
                    if 'Value' in data and 'StringWithMarkup' in data['Value'] and len(data['Value']['StringWithMarkup']) > 0:
                        odor_list.append(data['Value']['StringWithMarkup'][0]['String'])
                    if 'Reference' in data:
                        odor_reference_list.append(data['Reference'][0])
            
            odor_data = ' // '.join(odor_list)
            odor_reference = ' // '.join(odor_reference_list)
            return odor_data, odor_reference

        odor_descriptions = []
        odor_descriptions_reference = []
        odor_thresholds = []
        odor_thresholds_reference = []
        for cid in tqdm(self.data.index):
            response = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/')
            if response.status_code == 200:
                pubchem_dict = response.json()  # Automatically parses JSON into a Python dict
            odor_data, odor_reference = get_odor_data_from_dict(pubchem_dict, 'Odor')
            odor_threshold_data, odor_threshold_reference = get_odor_data_from_dict(pubchem_dict, 'Odor Threshold')
            odor_descriptions.append(odor_data)
            odor_descriptions_reference.append(odor_reference)
            odor_thresholds.append(odor_threshold_data)
            odor_thresholds_reference.append(odor_threshold_reference)
        self.data['raw_description'] = odor_descriptions
        self.data.loc[:,'source_detail'] = odor_descriptions_reference
        self.data.loc[:,'odor_threshold'] = odor_thresholds
        self.data.loc[:,'odor_threshold_reference'] = odor_thresholds_reference

    def clean_molecules(self):
        self.data['canonical_smiles'] = SmilesCanonicalizer().canonicalize_smiles(self.data['SMILES'])

        