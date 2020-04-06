import nltk 
from nltk.corpus import wordnet
import re
import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()


#################################### methods


def getSynonymsAntonymns(word):
    synonyms = [] 
    antonyms = [] 

    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name())

    return set(synonyms),set(antonyms)

def getSynonymsAntonymnsListOfWords(words,bagged=True):
    synonyms = [] 
    antonyms = []

    for word in words:
        word_synonyms, word_antonyms = getSynonymsAntonymns(word)
        synonyms += [x.lower() for x in word_synonyms] + [word]
        antonyms += [x.lower() for x in word_antonyms] + [word]

    if bagged:
        return list(set(synonyms + antonyms))
    else:
        return list(set(synonyms)), list(set(antonyms))

def regexQueryDf(df,cols,patterns,operatorPattern='AND',operatorColumn='OR'):
    """Method that based on the columns you input checks for a list of rgular expressions

        operatorPattern: regex pattern must be satisfied for all combinations of patters
        operatorColumn: columns on which regex list must be satisfied all or at least 1.
    
    """

    if operatorPattern=='AND' and operatorColumn=='AND':
        cond = pd.Series([True]*len(df))
        for pattern in patterns: 
            cond = cond & (df.progress_apply(lambda row: all(row[cols].str.contains(pattern)),axis=1))
        return cond
    elif operatorPattern=='OR' and operatorColumn=='AND':
        cond = pd.Series([False]*len(df))
        for pattern in patterns: 
            cond = cond | (df.progress_apply(lambda row: all(row[cols].str.contains(pattern)),axis=1))
        return cond
    elif operatorPattern=='AND' and operatorColumn=='OR':
        cond = pd.Series([True]*len(df))
        for pattern in patterns: 
            cond = cond & (df.progress_apply(lambda row: any(row[cols].str.contains(pattern)),axis=1))
        return cond
    elif operatorPattern=='OR' and operatorColumn=='OR':
        cond = pd.Series([False]*len(df))
        for pattern in patterns: 
            cond = cond | (df.progress_apply(lambda row: any(row[cols].str.contains(pattern)),axis=1))
        return cond

############################################################################################################

if __name__ == "__main__":

    ####### gender related words

    gender_words = ['male','female','sex','gender']
    pattern_gender = ' | '.join(getSynonymsAntonymnsListOfWords(gender_words))
    pattern_gender = pattern_gender.replace(' |','|(?i)') #regex


    ############################### corona virus string match
    # keywords from here https://coviz.apps.allenai.org/bc5cdr/?d=bc5cdr&l=40&ftm=11444


    pattern_COVID = 'respiratory tract infection |virus infection |respiratory syncytial virus | \
                    lipopolysaccharide |death |acute respiratory distress syndrome |acute respiratory failure | \
                    H1N1 viral infection |rubella virus infection |influenza virus infection |human immunodeficiency virus | \
                    irritation of the respiratory tract |Zika Virus Infection |Ebola and Zika virus infection | \
                    porcine reproductive and respiratory syndrome |TAP |influenza virus A |Thrombocytopenia Syndrome Virus Infection | \
                    SARS-CoV-2 |respiratory syncitial virus |skin or mucous membrane lesions |upper respiratory infection | \
                    H5N1 viral infection |herpes simplex virus type 1 |human immunodeficiency virus type 1 |gastrointestinal viral infection | \
                    reproductive and respiratory syndrome virus infection |porcine reproductive and respiratory syndrome virus | \
                    hepatitis A virus |acquired immunodeficiency syndrome |parainfluenza virus 3 | \
                    nosocomial viral respiratory infections |coronavirus OC43 infection |IFN |H3N2 virus infection | \
                    dsRNA |dsDNA |long QT syndrome |liver cell necrosis |latent TB infection |Pulmonary Coronavirus Infection | \
                    Dengue virus Type |neurotropic coronavirus virus |Leukocyte adhesion deficiency II syndrome | \
                    Human T-cell leukemia virus type 1 |Human T-cell leukemia virus type | \
                    infection of the central nervous system |infection of the pulmonary parenchyma'

    pattern_COVID = pattern_COVID.replace(' |','|(?i)')


    ######## joining paterns
    file = 'metadata_pre_proc.csv'

    df = pd.read_csv('../Data/'+file,nrows=500) #sample of 500 papers

    ######## all combinations of all patterns to be satisfied in all columns
    cols = ['title','abstract']

    cond_COVID_gender = regexQueryDf(df,cols,[pattern_COVID,pattern_gender],operatorPattern='AND',
                                                                                operatorColumn='OR')

    print('Number of papers found: ', np.sum(cond_COVID_gender))

    cond_COVID_gender = regexQueryDf(df,cols,[pattern_COVID,pattern_gender],operatorPattern='AND',
                                                                                operatorColumn='AND')

    print('Number of papers found: ', np.sum(cond_COVID_gender))

    cond_COVID_gender = regexQueryDf(df,cols,[pattern_COVID,pattern_gender],operatorPattern='OR',
                                                                                operatorColumn='AND')

    print('Number of papers found: ', np.sum(cond_COVID_gender))

    cond_COVID_gender = regexQueryDf(df,cols,[pattern_COVID,pattern_gender],operatorPattern='OR',
                                                                                operatorColumn='OR')

    print('Number of papers found: ', np.sum(cond_COVID_gender))



    df = df[cond_COVID_gender].reset_index(drop=True)

    df.to_csv('../Data/gender_papers.csv')

