from nltk.corpus import wordnet 
from tqdm.notebook import tqdm
from collections import defaultdict, OrderedDict

import yaml
import pandas as pd
import numpy as np
import nltk
import re


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
        # drop na for particular columns
        cols = ['title', 'abstract']
        df = df.dropna(subset=cols)
        df = df.reset_index(drop=True)
        
        return df
        
        
    def classify_all(self, df):
        """
        Classify a dataframe of abstracts into differenent categories based on keyword search
        
        There is a performance issue for using regular expression in panda dataFrae.str.contains().
        See: https://stackoverflow.com/questions/37894003/how-to-make-pandas-dataframe-str-contains-search-faster. 
        I will change the way we use the regexQueryDf in query_model.py
        
        throughout the processing, we will KEEP the same entries despite it does not fit
        certain criteria (e.g., not covid-related). This is to ensure consistency in the
        dataframe array.
        
        :param df (pandas): the dataframe for all the abstracts. It should contain
                            the 'title' and 'abstract' columns
        """
        # initial parameters
        classes, subclasses, subclasses_kws, kws_all = self.get_km_info()
        
        # deep copy the dataframe first
        df = df.copy(deep=True)
        
        # ---------------- Identify if the paper is related to covid19 first
        kws = self.km['disease_name']['disease_common_name']
        has_dnames = []
        for i in tqdm(range(0, df.shape[0])):
            row = df.iloc[i]
              
            if (self._find_kws(row['title'], kws) or
                self._find_kws(row['abstract'], kws)):
            # if (any(w in row['title'] for w in kws) or
            #    any(w in row['abstract'] for w in kws)):
                    has_dnames.append(True)
            else:
                has_dnames.append(False)
        df['covid_related'] = has_dnames
        
        # ---------------- Classify paper into categories
        # ----- create new columns for classes, subclasses, keywords in dataframe first
        cols = classes + list(subclasses.keys()) + ['keywords']
        df_cols = pd.DataFrame(0, index=list(range(0, df.shape[0])), 
                               columns=cols)
        df = pd.concat([df, df_cols], axis=1)
        
        # ----- classify
        for i in tqdm(range(0, df.shape[0])):
            kws_found = self._find_kws(df.loc[i, 'abstract'], kws_all)
            if kws_found:
                # assign the label into the df 
                relevant_cols = self._get_relevant_info(kws_found, subclasses, 
                                                        subclasses_kws,)
                df.loc[i, 'keywords'] = ",".join(kws_found)
                df.loc[i, relevant_cols] = 1
        return df
                                   

    def get_km_info(self):
        """
        Obtain structured information from the knowledge map yaml file.
        
        This information is somewhat the "reverse" structure of the 
        knowledge map. E.g., {male:gender}, {gender:risk_factor}, {risk_factor}
        
        Note: there should not be overlap between keywords (kws) between
             classes-subclasses. If so, that mean the manually defined 
             knowledge map is not good. And we need to change the yaml.
        
        return classes (list): the major classes, e.g., risk_factor
                subclasses (dict): the subclasses and which classes each belong to,
                            e.g., {gender:risk_factor}
                subclasses_kws (dict): the subclasses' that a knowledge map to, 
                            e.g., {male:gender}
                kws (list): a list of all possible keywords
        """
        # classes info
        classes = [x for x in list(self.km.keys()) if x!='disease_name']
        
        # subclasses info
        subclasses = OrderedDict()
        subclasses_kws = OrderedDict()
        kws = []
        for c in classes:
            for sc in self.km[c].keys():
                # assign a class to a subclass
                subclasses[sc] = c

                # assign a kw to a subclass
                for kw in self.km[c][sc]:
                    subclasses_kws[kw] = sc

                # add keywords to the kws list
                kws += self.km[c][sc]

        return classes, subclasses, subclasses_kws, kws
     
    
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
        
        # remove '_' from the words since syn use 
        # _ to represent separation, e.g., 'female_person'
        synonyms = [w.replace("_", " ") for w in synonyms]
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

                               
    def _get_relevant_info(self, kws, subclasses, subclasses_kws,):
        """
        Given a list of keywords, find the associated subclasses' and classes' labels
        so that we can use that to update the dataframe
        
        :param kws (list): a list of keywords that can be found in the
                            value of the dict sublcasses_kws
                subclasses (dict): the subclasses and which classes each belong to,
                            e.g., {gender:risk_factor}
                subclasses_kws (dict): the subclasses' that a knowledge map to, 
                                        e.g., {male:gender}
        :return a list of class and subclass names that are relvant to the list of keywords
        """
        c_set = set()
        sc_set = set()
        for kw in kws:
            sc_set.add(subclasses_kws[kw])
        for sc in list(sc_set):
            c_set.add(subclasses[sc])
        return list(c_set) + list(sc_set)
    
    
    def _find_kws(self, s, kws):
        """
        Identity what keywords are present in a sentence s
        given the keyword list (kws)
        
        :param s (string): a sentence
        :param kws (list): a list of keywords want to identify
        """
        def is_phrase_in(text, phrase):
            """
            Identify the exact phrase in text. 
            We cannot use "if phrase in text" because this will check subwords.
            For example, "age is in teenage". We need to find exact match. 
            """
            return re.search(r"\b{}\b".format(phrase), text, re.IGNORECASE) is not None
        
        # check
        kws_found = []
        for kw in kws:
            if is_phrase_in(s, kw):
                kws_found.append(kw)
        return kws_found
    

# ======================================== Soon to be abandon functions
#     def classify(self, s):
#         """
#         Classify a document the class and the subclass of it. As well as
#         providing the keywords that link to it for the subclass
        
#         Class & subclasses: 
#             risk_factor: gender, age, etc
#             diagnostic
#             treatment_and_vaccine
#             outcome
            
#         multiple steps to classify a paper:
#             1. Must contains coronavirus disease name in title or abtract
#             2. Search for the keywords for that appear in the abstract, and then return
#                 all the keywords find
            
#         :param s (pandas series): the pandas series for the paper information. It should contain
#                             the 'title' and 'abstract' columns
#         """
#         classes = []
#         kws = []
#         print(s['abstract'])

#         # check if the abstract fullfill the criteria
#         disease_names = self.km['disease_name']['common_name']
#         if (any(word in s['title'] for word in disease_names) or
#             any(word in s['abstract'] for word in disease_names)):
#             pass
        
        
#         return classes, kws
    
    
#     def match_keywords(s, kws):
#         """
#         Give a string s, dind the keyword(s) that appear in kws and caclulate the occurance as well.
        
#         :param s (string): a sentece
#         :param kws (list): a list of keywords
#         """
#         kws_match = defaultdict()
#         for token in s:
#             print(token)
        
        
        






