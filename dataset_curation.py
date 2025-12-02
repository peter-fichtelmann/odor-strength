#!/usr/bin/env python
# coding: utf-8


# In[2]:


# tested with python==3.12.2


# ## Dataset Curation and Analysis

# In[1]:


import os
import pandas as pd
from data.data_cleaner import GoodScentsDataCleaner, PubChemDataCleaner
from data.load_odor_strength import load_odor_strength


# ### Web Scraping

# In[2]:


GOODSCENTS_PATH = 'data/goodscents/'
PUBCHEM_PATH = 'data/pubchem/'
# pubchem csv from pubchem classification browser
PUBCHEM_PATH_ODOR_DESCRIPTION_CSV = PUBCHEM_PATH + 'pubchem_odor_descriptions_cids.csv'
if not os.path.exists(GOODSCENTS_PATH):
    os.makedirs(GOODSCENTS_PATH)
if not os.path.exists(PUBCHEM_PATH):
    os.makedirs(PUBCHEM_PATH)

# In[3]:


if not os.path.exists(GOODSCENTS_PATH + 'goodscents.csv'):
    goodscents_data_cleaner = GoodScentsDataCleaner()
    goodscents_data_cleaner.crawl_data()
    goodscents_data_cleaner.clean_molecules()
    goodscents_data_cleaner.data['source'] = [goodscents_data_cleaner.__class__.__name__.replace('DataCleaner', '') for i in range(len(goodscents_data_cleaner.data))]
    goodscents_data_cleaner.data.to_csv(GOODSCENTS_PATH + 'goodscents.csv')
df_goodscents = pd.read_csv(GOODSCENTS_PATH + 'goodscents.csv', index_col=0)


# In[4]:


if not os.path.exists(PUBCHEM_PATH + 'pubchem.csv'):
    df_odor_descriptions_cids = pd.read_csv(PUBCHEM_PATH_ODOR_DESCRIPTION_CSV, index_col=0)
    pubchem_datacleaner = PubChemDataCleaner(df_odor_descriptions_cids)
    pubchem_datacleaner.crawl_data()
    pubchem_datacleaner.clean_molecules()
    pubchem_datacleaner.data.to_csv(PUBCHEM_PATH + 'pubchem.csv')
df_pubchem = pd.read_csv(PUBCHEM_PATH + 'pubchem.csv', index_col=0)


# #### Data cleaning

# In[5]:


df_odor_strength, groups = load_odor_strength(df_goodscents, df_pubchem, target_dataset='GoodScents')


# In[8]:


df_odor_strength.head(5)


# In[11]:


print(df_odor_strength.shape)
print(df_odor_strength['odor_strength'].value_counts(), '\n')
print(df_odor_strength['numerical_strength'].value_counts(), '\n')
print(df_odor_strength.groupby('source')['numerical_strength'].value_counts())


# In[10]:


df_odor_strength.to_csv('data/df_odor_strength.csv')


# In[7]:


pd.DataFrame(groups).to_csv('data/odor_strength_groups.csv')


# In[ ]:




