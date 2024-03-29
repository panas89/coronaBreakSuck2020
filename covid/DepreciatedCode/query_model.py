import nltk 
from nltk.corpus import wordnet
import re
import pandas as pd
import numpy as np
from covid.models.paperclassifier.frontpaperclassifier import FrontPaperClassifier

from tqdm import tqdm
tqdm.pandas()

#-------------------- GLOBAL VARS -----------------------------

from covid.data.constants import COVID_WORDS

YAML_PATH = '../covid/models/paperclassifier/interest.yaml'
fpc = FrontPaperClassifier(km_path=YAML_PATH)

# TO-BE DEPRECIATED!!!
risk_factor_words = 'male | female | sex | gender '.split('| ')

covid_words = 'respiratory tract infection |virus infection |respiratory syncytial virus | \
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
                    infection of the central nervous system |\
                    infection of the pulmonary parenchyma |covid | coronavirus'.split(' |')

#-------------------- METHODS -----------------------------


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
        # print(cond)
        for pattern in patterns:
            # print(df.progress_apply(lambda row: row[cols].str.contains(pattern).fillna(False).all(),axis=1))
            # print(df.progress_apply(lambda row: row[cols].str.contains(pattern).fillna(False),axis=1))
            cond = cond & (df.progress_apply(lambda row: row[cols].str.contains(pattern).fillna(False).all(),axis=1))
        #     print(cond)
        # print(cond)
        return cond
    elif operatorPattern=='OR' and operatorColumn=='AND':
        cond = pd.Series([False]*len(df))
        for pattern in patterns: 
            cond = cond | (df.progress_apply(lambda row: row[cols].str.contains(pattern).fillna(False).all(),axis=1))
        return cond
    elif operatorPattern=='AND' and operatorColumn=='OR':
        cond = pd.Series([True]*len(df))
        for pattern in patterns: 
            cond = cond & (df.progress_apply(lambda row: row[cols].str.contains(pattern).fillna(False).any(),axis=1))
        return cond
    elif operatorPattern=='OR' and operatorColumn=='OR':
        cond = pd.Series([False]*len(df))
        for pattern in patterns: 
            cond = cond | (df.progress_apply(lambda row: row[cols].str.contains(pattern).fillna(False).any(),axis=1))
        return cond




#-------------------- CLASSES -----------------------------

class DataSearchByQueryEngine:
    """
    Class for reading, processing, and writing data from the
    Semantics scholar website.



    """
    def __init__(self,filename,cols_to_query,patterns):
        self.filename=filename
        self.df = None
        self.cols_to_query = cols_to_query
        self.df_filtered = None
        self.patterns = patterns
        self.operatorColumn = 'AND'
        self.operatorPattern = 'AND'

    def read_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        self.df = pd.read_csv(raw_data_path)

    def query_data(self):
        """Process raw data into useful files for model."""
        cond = regexQueryDf(self.df,self.cols_to_query,
                            self.patterns,operatorPattern=self.operatorPattern,
                            operatorColumn=self.operatorColumn)
        self.df_filtered = self.df[cond].reset_index(drop=True)

    def write_data(self, df, processed_data_path):
        """Write processed data to directory."""
        
        df.to_csv(processed_data_path+self.filename,
                    index=False)



class FrontDataSearchByQueryEngine(DataSearchByQueryEngine):

    def __init__(self, filename, cols_to_query, subclass):

        self.subclass = subclass
        self.keywords = self._get_keywords_from_subclass()
        self.patterns = [self._generatePattern(COVID_WORDS), 
                         self._generatePattern(self.keywords)]

        super().__init__(filename, cols_to_query, self.patterns)


    def _generatePattern(self, words):
        pattern = ' | '.join(words)
        return pattern.replace(' |','|(?i)') #regex

    def _get_keywords_from_subclass(self):
        return fpc.get_keywords(self.subclass)

        

class PatternGenerator:

    def __init__(self,words):
        self.words = words
        self.pattern = None

    def addSynonyms(self):
        self.words,_ = getSynonymsAntonymnsListOfWords(self.words,bagged=False)

    def addAntonyms(self):
        _,self.words = getSynonymsAntonymnsListOfWords(self.words,bagged=False)

    def addSynonymsAntonyms(self):
        self.words = getSynonymsAntonymnsListOfWords(self.words,bagged=True)

    def generatePattern(self):
        pattern = ' | '.join(self.words)
        self.pattern = pattern.replace(' |','|(?i)') #regex

    def getPattern(self):
        return self.pattern








############################################################################################################

if __name__ == "__main__":
    pass

