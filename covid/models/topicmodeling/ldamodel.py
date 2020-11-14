# Misc
import os
import time
from pathlib import Path

# Arrays & Dataframes
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, Phrases, TfidfModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.matutils import corpus2dense, corpus2csc

# Visualization
import pyLDAvis.gensim

# Sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA,IncrementalPCA,SparsePCA,TruncatedSVD
from sklearn.mixture import GaussianMixture


# **Note**: LdaModel trasforms docs and tokens to the (num_topics)-dimensional latent space of topics
# LdaModel[corpus][i][0] = list((topic_no, probability))
# LdaModel[corpus][i][1] = list((token_vocab_key, [topic_no])
# LdaModel[corpus][i][2] = list((token_vocab_key, [(topic_no, probability)])

CUR_DIR = str(Path(__file__).parents[0])
MALLET_DIR= CUR_DIR + '/Mallet/'

class LDAModel:

    def __init__(self, text_data, test_data=None):
        """
        Class Object to train and evaluate LDA gensim models.
        Input:
            - text_data: list of docs, i.e. list of lists of tokens
        """

        # Create Dictionary
        self.id2word = corpora.Dictionary(text_data)

        # Convert each doc to bag-of-words
        self.corpus = self.create_corpus(text_data)  
        self.test_corpus = self.create_corpus(test_data) if test_data else None 

        # Model-related Attributes
        self.lda_model = None
        self.params = None
        self.coherence_score = None

    #----------------------- Training Methods ------------------------
    
    def create_corpus(self, text_data):
        """
        Convert each document (a list of tokens) in text_data (list of docs)
        into the bag-of-words format: list of (token_id, token_count) 2-tuples.
        """

        return [self.id2word.doc2bow(text) for text in text_data]

    def create_tfidf_corpus(self):

        # Initialize gensim's TF-IDF
        tfidf = TfidfModel(self.corpus, self.id2word)

        # Convert corpus to tfidf-corpus
        self.corpus = tfidf[self.corpus]

        return self

    def getLowerTopicBound(self):
        """
        Method that computes the lower bound of topics using, tfidf then PCA, and mixture of gaussian with BIC to estimate optimal number of gaussians (i.e., topics).

        Output:
          - Integer: lower bound number of topics.

        """
        # Initialize gensim's TF-IDF
        tfidf = TfidfModel(self.corpus, self.id2word)
        num_docs = self.id2word.num_docs
        num_terms = len(self.id2word.keys())
        
        # batch_size = 10000

        # pca = IncrementalPCA(n_components=200,batch_size=250)
        # for i in tqdm(range(0,len(self.corpus),batch_size)):
        #     corpus_tfidf = tfidf[self.corpus[i:i+batch_size]]

        #     #to numpy array
        #     corpus_tfidf_dense = corpus2dense(corpus_tfidf, num_terms, len(corpus_tfidf))[:,:1000].T
            
        #     #PCA decomposition
        #     pca.partial_fit(corpus_tfidf_dense)

        #     if i==0:
        #         corpus_pca = pca.transform(corpus_tfidf_dense)
        #     else:
        #         corpus_pca = np.vstack((corpus_pca,pca.transform(corpus_tfidf_dense)))

        pca = TruncatedSVD(n_components=100)
        corpus_tfidf = tfidf[self.corpus]
        corpus_sparse = corpus2csc(corpus=corpus_tfidf, num_terms=num_terms, num_docs=len(corpus_tfidf))
        pca.fit(corpus_sparse)
        corpus_pca = pca.transform(corpus_sparse)

        #Mixture of Gaussians and BIC
        gm = GaussianMixture(n_components=1,covariance_type='diag')

        print('Finding the lower bound of optimal number of topics ...')

        bics = [GaussianMixture(n_components=num_comp,covariance_type='diag').fit(corpus_pca).bic(corpus_pca) for num_comp in tqdm(range(2,12))]

        print('Lower bound: ', np.argmin(bics) + 2)

        return np.argmin(bics) + 2 #counting at 2 + 1 for the index to get number of components


    def filter_low_tfidf(self, text_data, min_tfidf_score=0.01, keep_fraction=None):
        """
        Filter out tokens in the dictionary (id2word) by their tf-idf score.
        Parameters:
            - text_data
            - min_tfidf_score (float)
        Output:
            - updates self.id2word
            - updates self.corpus
            - returns self  
        """

        # Initialize gensim's TF-IDF
        tfidf = TfidfModel(self.corpus, self.id2word)
        
        if keep_fraction:
            # compute the distribution of scores
            id_to_scores = {}
            for bow in tfidf[self.corpus]:
                for tupl in bow:
                    ID,score = tupl
                    id_to_scores[ID] = score

            # find the score above which lies the 'keep_fraction' of the tokens
            min_tfidf_score = np.percentile(list(id_to_scores.values()), (1-keep_fraction)*100)

        # filter out tokena with scores below min_tfidf_score
        low_score_tokens = []
        for bow in self.corpus:
            low_score_tokens += [ID for ID,score in tfidf[bow] if score < min_tfidf_score]

        # update dictionary (id2word) and corpus
        self.id2word.filter_tokens(bad_ids=low_score_tokens)
        self.corpus = self.create_corpus(text_data)            

        return self


    def filter_extremes(self, text_data, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        """
        Filter out tokens in the dictionary (id2word) by their frequency.
        Parameters:
            - text_data
            - no_below (int, optional): Keep tokens which are contained in at least no_below documents.
            - no_above (float, optional): Keep tokens which are contained in no more than no_above documents 
            (fraction of total corpus size, not an absolute number).
            - keep_n (int, optional): Keep only the first keep_n most frequent tokens.
            - keep_tokens (iterable of str): Iterable of tokens that must stay in dictionary after filtering.
        Notes:
        This removes all tokens in the dictionary that are:
            1. Less frequent than no_below documents (absolute number, e.g. 5) or
            2. More frequent than no_above documents (fraction of the total corpus size, e.g. 0.3).
            3. After (1) and (2), keep only the first keep_n most frequent tokens (or keep all if keep_n=None).

        Output:
            - updates self.id2word
            - updates self.corpus
            - returns self  
         """

        self.id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None)
        self.corpus = self.create_corpus(text_data)

        return self

    


    def build_model(self, params, lda_class='single'):
        """
        Method for building an LDA model.
        Input:
            - params: dict of (parameter name (string), value) pairs; see 
            https://radimrehurek.com/gensim/models/ldamodel.html, etc, for documentation
            - lda_class: (string) specify the gensim model to be used for training;
            at the moment {'single': LdaModel, 'multi': LdaMulticore}
        Output:
            Trains a gensim LDA model, creates two class object attributes self.{params, lda_model} 
            and returns self
        """

        self.lda_class = lda_class
        self.params = params

        if params.get('per_word_topics'):
            raise ValueError('Omit using parameter per_word_topics because it creates buggy models.')

        if lda_class == 'single':
            self.lda_model = LdaModel(corpus=self.corpus, id2word=self.id2word, **params)
        elif lda_class == 'multi':
            self.lda_model = LdaMulticore(corpus=self.corpus, id2word=self.id2word, **params)
        elif lda_class == 'mallet':
            os.environ.update({'MALLET_HOME': MALLET_DIR}) #'../covid/models/topicmodeling/mallet-2.0.8/'
            mallet_path = MALLET_DIR + 'bin/mallet'
            mallet_model = LdaMallet(mallet_path, corpus=self.corpus, id2word=self.id2word, **params)
            self.lda_model = malletmodel2ldamodel(mallet_model)

        return self


    def grid_search(self, text_data, param_grid, lda_class='single', scorers=['coherence_score']):
        """
        Input:
            - text_data: list of docs, i.e. list of lists of tokens
            - param_grid: dict of (parameter name (string), list of values); see 
            https://radimrehurek.com/gensim/models/ldamodel.html, etc, for documentation
            - lda_class: (string) specify the gensim model to be used for training;
            at the moment {'single': LdaModel, 'multi': LdaMulticore}
        Output:
            Best model's attributes are assigned to the class object's attributes,
            i.e. self.{coherence_score, params, lda_model},  
            AND returns a dict of dicts with:
                - Outer dict's keys: 'model_i' where 0<= i < number of grids
                - Inner dict's keys: {'coherence_score', 'params', 'lda_model'}
        """
        start_time = time.time()

        # Instantiate a sklearn ParameterGrid
        parameter_grid = ParameterGrid(param_grid)

        # Loop through all grids of params and store results to models dict
        models = {}
        best_score = 0
        best_model_key =''
        for i,params in enumerate(tqdm(parameter_grid)):
            
            # Train model
            self.build_model(params, lda_class=lda_class)

            # Store params, model, score in dict
            key = f'model_{i}'
            models[key] = {}
            models[key]['params'] = params
            models[key]['lda_model'] = self.lda_model

            for s in scorers:
                if s == 'coherence_score':
                    models[key][s] = self.compute_coherence_score(text_data)
                elif s == 'f1_score':
                    models[key][s] = self.compute_f1_score(average='macro')
            
            # Find best model's key
            score = models[key][scorers[0]]  # use first scorer in scorers to find the best model
            if best_score < score:
                best_score = score
                best_model_key = key

        # Update class attributes to best model's attributes
        self.params = models[best_model_key]['params']
        self.lda_model = models[best_model_key]['lda_model']
        for s in scorers:
            self.coherence_score = models[best_model_key].get('coherence_score',np.nan)
            self.f1_score = models[best_model_key].get('f1_score',np.nan)
        
        print("Training lasted: {} sec".format(round(time.time() - start_time,3)))

        return models

    def compute_coherence_score(self, text_data, verbose=False):
        """Input:
                - text_data: list of lists of tokens
                - verbose: Boolean - if True, print coherence score
            Output:
                Creates a class object attribute self.coherence_score
                and returns its value
        """

        coherence_model_lda = CoherenceModel(model=self.lda_model, 
                                             texts=text_data, 
                                             dictionary=self.id2word, 
                                             coherence='c_v')

        self.coherence_score = coherence_model_lda.get_coherence()

        if verbose:
            print('\nCoherence Score: ', self.coherence_score)

        return self.coherence_score


    def compute_f1_score(self, average='macro', verbose=False):

        if self.test_corpus is None:
            raise ValueError('self.test_corpus is None')

        if len(self.test_corpus)!= len(self.corpus):
            raise ValueError('test_corpus and corpus must have the same lengths')
        
        test_preds = self.predict_topics(self.test_corpus)
        text_preds = self.predict_topics(self.corpus)

        self.f1_score = f1_score(text_preds, test_preds, average=average)

        if verbose:
            print('\nf1-score: ', self.f1_score)

        return self.f1_score



    def predict_topics(self, corpus=None):
        corpus = corpus if corpus is not None else self.corpus

        preds = []
        for doc in self.lda_model[corpus]:
            dominant_topic,_ = max(doc, key=lambda x: x[1])
            preds.append(dominant_topic)

        return preds
            

    #----------------------- Post-Training Methods ------------------------

    def get_topic_keywords(self, topic_no):
        """
        Return the top ten keywords of the input topic: topic_no (int).
        """

        return [word for word, prob in self.lda_model.show_topic(topic_no)]

    def create_topic_keywords_df(self):
        """
        Method to visualize top keywords per topic.
        Output: 
            a df with:
                - data = top ten keywords per topic
                - cols = topic numbers
                - index = range(0,10)
        """
        num_topics = self.params['num_topics']
        df = pd.DataFrame.from_dict(
                                dict( (i,self.get_topic_keywords(i)) for i in range(num_topics) )
                                )
        df.columns.name = 'topic_no'
        df.index.name = 'top_keywords'

        return df
    
    def create_top_topic_per_doc_df(self, other_corpus=None):
        """
        Use the trained model to find the dominant topic and the correpsonding probability,
        for each doc in the input corpus. 
        Input:
            - other_corpus: gensim corpus; if None, self.corpus will be used, else, specified 
            other_corpus will be used 
        Output:
            a df with:
                - cols = 'dominant_topic', 'topic_probability', 'topic_keywords'
                - index = corpus ids, range(0, len(corpus))
        """
        # Set the corpus var
        corpus = self.corpus
        if other_corpus: # if other_corpus is specified use that corpus instead
            corpus = other_corpus

        # For each doc in corpus find the dominant topic, and the corresponding probability
        to_concat = []    
        for doc in self.lda_model[corpus]:
            dominant_topic, topic_prob = max(doc, key=lambda x: x[1])
            topic_kws = ", ".join(self.get_topic_keywords(dominant_topic))
            temp_df = pd.DataFrame(data=[dominant_topic, round(topic_prob,4), topic_kws]).T
            to_concat.append(temp_df)
            # df = df.append(temp_series, ignore_index=True)
        
        # Create df
        df = pd.concat(to_concat, axis=0, ignore_index=True)
        df.columns = ['dominant_topic', 'topic_probability', 'topic_keywords']
        df.dominant_topic = df.dominant_topic.astype('int')
        df.index.name = "corpus_id"

        return df

    def find_topic_to_rep_id(self):
        """
        Method to find the representative doc of each topic.
        Output: 
            a dict of (topic_no, corpus_id) pairs
        """
        # Initialize a dict with keys=topic_no and values=(doc_id,topic_prob)
        num_topics = self.params['num_topics']
        topic_to_id_prob = dict((i,(0,0)) for i in range(num_topics))
        
        # Loop through corpus and update dict whenever a doc has the max topic_prob
        for ID, doc in enumerate(self.lda_model[self.corpus]):
            dominant_topic, topic_prob = max(doc, key=lambda x: x[1])
            if topic_prob > topic_to_id_prob[dominant_topic][1]:
                topic_to_id_prob[dominant_topic] = (ID, topic_prob)

        return dict((topic_no, ID) for topic_no,(ID, prob) in topic_to_id_prob.items())        

    def getLDAVisualization(self):
        """
        Method to get the LDA visualization object.
        """

        return pyLDAvis.gensim.prepare(self.lda_model, corpus=self.corpus, dictionary=self.id2word)

    