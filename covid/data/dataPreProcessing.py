
############################### dependencies ##############################

import glob, os
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from covid.data.nltkPreProc import *

###########################################################################

def nltkPreProcDf(df,cols):
    """Method to preprocess a dataframe using the preProcessPipline mthod from the nltkPreProc modeule.
       Returns a DF."""
    for col in cols:
        df[col] = df[col].progress_apply(lambda x: preProcessPipeline(x))
    return df


if __name__ == "__main__":
    tqdm.pandas()

    files_with_text = False #only files with text
    all_files = False #both files with text and metadata
    metdata_files = True #only metadata


    if files_with_text:
        cols_to_pre_proc = ['title','abstract','text']

        ###files with the main text
        files = ['bioarxiv_papers.csv','comm_use_papers.csv','cust_lic_papers.csv','non_comm_use_papers.csv']

        for file in files:
            df = pd.read_csv('./Data/'+file)
            df = nltkPreProcDf(df,cols_to_pre_proc)
            df.to_csv('./Data/'+file.replace('.csv','')+'_pre_proc.csv',index=False)

        print('Analyzed corpora with texts!')
    elif all_files:

        cols_to_pre_proc = ['title','abstract','text']

        ###files with the main text
        files = ['bioarxiv_papers.csv','comm_use_papers.csv','cust_lic_papers.csv','non_comm_use_papers.csv']

        for file in files:
            df = pd.read_csv('./Data/'+file)
            df = nltkPreProcDf(df,cols_to_pre_proc)
            df.to_csv('./Data/'+file.replace('.csv','')+'_pre_proc.csv',index=False)

        print('Analyzed corpora with texts!')
    
        ##preprocessing the metadata csv which containts all papers info except main text
        cols_to_pre_proc = ['title','abstract']
        file = 'metadata.csv'
        df = pd.read_csv('..?data/'+file)
        df = nltkPreProcDf(df,cols_to_pre_proc)
        df.to_csv('./Data/'+file.replace('.csv','')+'_pre_proc.csv',index=False)


        print('Analyzed corpora metadata with titles and abstracts only!')
    elif metdata_files:

        ##preprocessing the metadata csv which containts all papers info except main text
        cols_to_pre_proc = ['title','abstract']
        file = 'metadata.csv'
        df = pd.read_csv('../../data/'+file)
        df = nltkPreProcDf(df,cols_to_pre_proc)
        df.to_csv('../../Data/'+file.replace('.csv','')+'_pre_proc.csv',index=False)


        print('Analyzed corpora metadata with titles and abstracts only!')
