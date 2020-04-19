from nltk.corpus import wordnet 
from tqdm.notebook import tqdm
from collections import defaultdict

import yaml
import pandas as pd
import numpy as np
import nltk


class PaperClassifier(object):
    def __init__(self, km_path='interest.yaml'):
        """
        This class is to classify the type of information that a paper MAY contains.
        
        ----------
        v1 is based on keyword matches (not comparing the word2vec semantics yet).
        The approach here is to create a knowledge map, which define the hierarchy of
        information that we are interested in as well as keywords and their synonyms. 
        Then we search through the title/abstrac to classify each papers. 
        
        :param km_path (string): file path for the knowledge map of subject of interests 
        """
        # nltk configure
        self._configure_nltk()
        
        # Load the basic synonyms that we define
        with open(km_path) as f:
            self.km = yaml.load(f, Loader=yaml.FullLoader)
            
        # Further expand the defined km keywords using nltk
        self._expand_keyword_lists()
        
        
    def preprocess(self, df):
        """
        Preprocess the title and abstract dataframe particularly for the meta information. 
        """
        # 
        pass
        
        
    
    def classify_all(self, df):
        """
        Classify a dataframe of abstracts into differenent categories based on keyword search
        
        :param df (pandas): the dataframe for all the abstracts. It should contain
                            the 'title' and 'abstract' columns
        """
        # loop each row in the dataframe and do the classification
        for i in tqdm(range(0, df.shape[0])):
            s = df.iloc[i, :]
            
            # classify
            classes, kws = self.classify(s)
            
        
        
                
            
    def classify(self, s):
        """
        Classify a document the class and the subclass of it. As well as
        providing the keywords that link to it for the subclass
        
        Class & subclasses: 
            risk_factor: gender, age, etc
            diagnostic
            treatment_and_vaccine
            outcome
            
        multiple steps to classify a paper:
            1. Must contains coronavirus disease name in title or abtract
            2. Search for the keywords for that appear in the abstract, and then return
                all the keywords find
            
        :param s (pandas series): the pandas series for the paper information. It should contain
                            the 'title' and 'abstract' columns
        """
        classes = []
        kws = []
        print(s['abstract'])

        # check if the abstract fullfill the criteria
        disease_names = self.km['disease_name']['common_name']
        if (any(word in s['title'] for word in disease_names) or
            any(word in s['abstract'] for word in disease_names)):
            pass
        
        
        return classes, kws
    
    
    def match_keywords(s, kws):
        """
        Give a string s, dind the keyword(s) that appear in kws and caclulate the occurance as well.
        
        :param s (string): a sentece
        :param kws (list): a list of keywords
        """
        kws_match = defaultdict()
        for token in s:
            print(token)
            
        
        
        
        
      
            
    def get_km(self):
        """
        Return the knowledge map (km)
        """
        return self.km
    
    
    def expand_keyword(self, kw):
        """
        Expand a keyword using NLTK synonyms.
        https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/
        
        :param kw (string): expand a keyword using nltk
        """
        synonyms = [kw]
        for syn in wordnet.synsets(kw): 
            for l in syn.lemmas(): 
                synonyms.append(l.name().lower()) 
        synonyms = list(set(synonyms))
        return synonyms
        
        
    def _configure_nltk(self):
        """
        Download all necessary libraries for NLTK
        """
        nltk.download('wordnet')
        
        
    def _expand_keyword_lists(self, ):
        """
        Expand the keyword lists using NLTK.
        
        This is highly dependendt on how we know about the knowledge map.
        Also, we don't expand all the keyword list because NLTK cannot
        handle some keywords well. Therefore, we will just manually include them
        in the yaml, and make the flag 'allow_nltk_expand'=False
        
        TODO: assign from using NLTK, we can use other technique, such as using
        embedding similarity calculation to find the keywords as well. We can
        expand this method later.
        
        Level: class --> subclass --> keywords
        """
        for aclass in self.km:
            subclasses = self.km[aclass]
            for sc in subclasses:
                if subclasses[sc]['allow_nltk_expand']:
                    # Retreives synonymps for keywords
                    kws = [w.lower() for w in subclasses[sc]['kw']]
                    kws_new = []
                    for kw in kws:
                        kws_new += self.expand_keyword(kw)

                    # Update the keywords list
                    subclasses[sc] = kws_new
                else:
                    subclasses[sc] = [w.lower() for w in subclasses[sc]['kw']]
    
    
        
        
        
        






