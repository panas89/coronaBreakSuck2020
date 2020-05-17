import numpy as np
import pandas as pd
from covid.models.paperclassifier.paperclassifier import PaperClassifier

# Load the data
DATA_DIR = '../data'
FILE_PATH = '/raw/merged_raw_data.csv'
USE_COLS = ['sha', 'title', 'abstract_x', 'text', 'publish_time']
df = pd.read_csv(DATA_DIR + FILE_PATH, usecols=USE_COLS)\
       .rename({'abstract_x': 'abstract'}, axis=1)
NUM_PAPERS = len(df) 

# Load the paperclassifier
pc = PaperClassifier(km_path='../models/paperclassifier/interest.yaml')

# Preprocess the dataframe text
df_p = pc.preprocess(df)

# Classify papers: add tags
df_p = pc.classify_all(df_p)

# Select covid data
df_covid = df_p.loc[df_p['covid_related']==1].drop(['covid_related'], axis=1)
NUM_COVID_PAPERS = len(df_covid) 

print("Fraction of covid papers: {}/{}".format(NUM_COVID_PAPERS, NUM_PAPERS))

# Save data
df_covid.to_csv(DATA_DIR + '/paperclassifier/classified_merged_covid.csv', index=False)