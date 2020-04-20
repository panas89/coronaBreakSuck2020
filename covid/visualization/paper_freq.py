import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from covid.data.constants import *
from covid.models.query_model import regexQueryDf,PatternGenerator
from covid.models.paperclassifier.frontpaperclassifier import FrontPaperClassifier

YAML_PATH = '../covid/models/paperclassifier/interest.yaml'
USECOLS = ['title', 'abstract']

# Instantiate FrontPaperClassifier; to be used for keywords retrieval
fpc = FrontPaperClassifier(km_path=YAML_PATH)

def compute_paper_freq(file_path,subclass):
    """ Method to compute annual frequencies of paper publications that much the keywords
        of the input subclass. 
        
        Input:
            - file_path: path to filtered data
        Output:
            df with:
                - index = years 
                - data = no of papers published per year that also much subclass's keywords
    """
    
    df = pd.read_csv(file_path,parse_dates=['publish_time'])
    NO_PAPERS = len(df)
    num_nans = df.publish_time.isnull().sum()

    df_filtered = df[~df.publish_time.isnull()].reset_index(drop=True)

    print("Dropped papers with MISSING dates: {}/{}"
          .format(num_nans, NO_PAPERS))
    print("Num of COVID papers that match {}: {}/{}"
         .format(subclass.upper(), len(df_filtered), len(df)))

    return  df_filtered.set_index('publish_time')\
            .groupby(lambda x: x.year)[['abstract']]\
            .count()\
            .rename({'abstract': 'paper_freq'}, axis=1)


def visualize_paper_freq(file_paths, subclass, from_year=1978):
    """ Method to visualize annual frequencies of paper publications that much the keywords
        of the input subclass.

        Input:
            - file_path
            - subclass
            - from_year: {int} consider only year values equal or greater than input
        Output:
            barplot of annual paper frequencies
    """

    to_concat = []
    for file_path in file_paths:
            to_concat.append(compute_paper_freq(file_path, subclass)['paper_freq'])
    
    df_total_paper_freq = pd.concat(to_concat, axis=1)\
                            .sum(axis=1)\
                            .astype('int')

    df_total_paper_freq = df_total_paper_freq[df_total_paper_freq.index >= from_year]
    
    df_total_paper_freq.plot(kind='bar', figsize=(18,7))
    plt.title(f"Annual Count of Publications about {subclass.upper()}", 
              fontdict = {'fontsize' : 18})

    missing_years = sorted(set(np.arange(from_year, 2021)) - set(df_total_paper_freq.index)) 

    return print(f"No papers were found for years: {missing_years}")




        
