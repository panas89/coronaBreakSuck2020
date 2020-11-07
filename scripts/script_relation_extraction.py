"""
This python script is prepared to automatically run the relation extraction for the classes 
and subclasses. The resulting file will be a .csv file

This script follows notebooks/main_kch_relation_extraction.ipynb
"""
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from covid.models.relation.extraction import RelationExtractor

# Note
# for how the data files look like, 
# checkout here: https://drive.google.com/drive/u/1/folders/1LSNRnk24Uiqb-l2Sm_0Uz-Fa2hI5YzYi

# initiate the extractor
print('----- Initiating the extractor -----')
covidre = RelationExtractor(km_path='../covid/models/paperclassifier/interest.yaml')

# load the paper-classified csv <-- paper that have been identified with classes
print('----- Loading the paper-classified csv -----')
df = pd.read_csv('../data/paperclassifier/classified_merged_covid.csv')

# extraction all the information
# it will take a while, ~10mins
print('----- Extracting the relation information -----')
relations = covidre.extract_all(df)
df['relations'] = relations

# save the information
print('----- saving -----')
df.to_csv('../data/paperclassifier/classified_merged_covid_relation.csv')

