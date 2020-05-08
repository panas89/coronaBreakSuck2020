
############################### dependencies ##############################

import glob, os
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm

###########################################################################

############################### reading files #############################

def getListOfFiles(path, extention=None):
    """Method that get path to directory and extention and returns list of file names."""
    if extention != None:
        return [file for file in os.listdir(path) if extention in file]
    else:
        return os.listdir(path)
###########################################################################


############################### paths to files ############################

path_all_archives = './'

path_bioarxiv = 'biorxiv_medrxiv/biorxiv_medrxiv/'
path_comm_use = 'comm_use_subset/comm_use_subset/'
path_cust_lic = 'custom_license/custom_license/'
path_non_comm_use = 'noncomm_use_subset/noncomm_use_subset/'

###########################################################################

############################### JSON to csv ############################

# code is from this Kaggle kernel

###########################################################################

import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm




def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return " ; ".join(text)

def format_location(authors):
    text = []
    for author in authors:
        affiliation = author['affiliation']
        if affiliation:
            location = affiliation.get('location')
            if location:
                text.extend(list(affiliation['location'].values()))
            else:
                text.append(' ')
        else:
            text.append(' ')
        
    return " ; ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return " ; ".join(name_ls)


def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_location(file['metadata']['authors']),               
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'location', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df

############################### Pattern matching ##########################

###########################################################################


def getPattern(x,pattern):
    "Method that returns regular expression patterns"

    findings = re.findall(pattern,x)

    if len(findings) > 0:
        return findings[0]
    else:
        return np.nan


def getCSVPapers(filenames,relative_path):

    files = []

    for i in range(len(filenames)):
        file = json.load(open(path_all_archives+relative_path+ filenames[i]))
        
        files.append(file)

    df = generate_clean_df(files)

    tqdm.pandas()

    pattern = r"\b(https://doi.org/10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>])\S)+)\b"

    links = df['text'].progress_apply(lambda x: getPattern(x,pattern) ) 
    
    df['links'] = links

    return df



if __name__ == "__main__":

    filenames_bioarxiv = getListOfFiles(path_all_archives+path_bioarxiv,extention='json')
    df_bioarxiv = getCSVPapers(filenames_bioarxiv,path_bioarxiv)
    df_bioarxiv.to_csv('./Data/bioarxiv_papers.csv',index=False)
    print('Finished analysis of bioarxiv!')

    filenames_comm_use = getListOfFiles(path_all_archives+path_comm_use,extention='json')
    df_comm_use = getCSVPapers(filenames_comm_use,path_comm_use)
    df_comm_use.to_csv('./Data/comm_use_papers.csv',index=False)
    print('Finished analysis of comm use!')

    filenames_cust_lic = getListOfFiles(path_all_archives+path_cust_lic,extention='json')
    df_cust_lic = getCSVPapers(filenames_cust_lic,path_cust_lic)
    df_cust_lic.to_csv('./Data/cust_lic_papers.csv',index=False)
    print('Finished analysis of custom lic!')

    filenames_non_comm_use = getListOfFiles(path_all_archives+path_non_comm_use,extention='json')
    df_non_comm_use = getCSVPapers(filenames_non_comm_use,path_non_comm_use)
    df_non_comm_use.to_csv('./Data/non_comm_use_papers.csv',index=False)
    print('Finished analysis of non comm use!')