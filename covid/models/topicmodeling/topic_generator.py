# Misc
import os
import pickle
from tqdm import tqdm
from pathlib import Path

# Arrays & Dataframes
import numpy as np
import pandas as pd

# Utilitieswhic
from covid.models.topicmodeling.ldamodel import LDAModel
from covid.models.topicmodeling.utilities.constants import COMMON_PHRASES_REGEX, COMMON_WORDS
from covid.models.topicmodeling.utilities.functions import *

# Gensim
from gensim.models import Phrases

#-------------------------- Global Vars ----------------------------

TOP_DIR = str(Path.cwd().parents[2])

#-------------------------- 1. Prepare Text Data -------------------------

print("1. Preparing Text Data\n")

file_path = TOP_DIR + '/data/paperclassifier/classified_merged_covid.csv'
df = pd.read_csv(file_path, parse_dates=['publish_time'])
df = process_pcf_data(df, 
                    bad_phrases=COMMON_PHRASES_REGEX, 
                    bad_tokens=COMMON_WORDS, 
                    drop_nan_text=True, 
                    from_date='2020-01-01')


text_data = df['clean_text'].values.tolist()

# Add bigrams to text data
bigram_model = Phrases(text_data, min_count=3, threshold=50) # higher threshold fewer phrases.
text_data = [bigram_model[doc] for doc in text_data]

#-------------------------- 2. Train LDA Models -------------------------

print("\n\n2. Starting Training\n")
# Choose training parameters
param_grid_mallet = {
    'num_topics': list(range(5,6,1)),
    'iterations': [1000],
    'random_seed': [100],
    'workers': [1]
}

param_grid_mallet = {
    'num_topics': list(range(5,6,1)),
    'passes': [3],
}

tfidf_grid = [0.25, 0.50, 0.75, 0.90]

scorers = ['coherence_score'] 

# Train Mallet LDA models
all_models = []
for fraction in tqdm(tfidf_grid):
    # initialize LDA model
    temp_lda = LDAModel(text_data, test_data=None).filter_low_tfidf(text_data, keep_fraction=fraction)
    # run grid-search: automatically assigns best model to LDAModel
    _ = temp_lda.grid_search(text_data, 
                             param_grid=param_grid_mallet, 
                             lda_class='single', 
                             scorers=scorers
                             )

    all_models.append(temp_lda)

# Print results
for i, model in enumerate(all_models):
    print(f'\nTF-IDF fraction: {tfidf_grid[i]}')
    print("- Coherence score: {}\n- Number of topics: {}"
          .format(model.coherence_score, model.params['num_topics'])
        )

# Find best LDA model
all_models.sort(key=lambda model: model.coherence_score)
best_model = all_models[-1]
print("\nSelected Model has\n- Coherence score: {}\n- Number of topics: {}"
      .format(best_model.coherence_score, best_model.params['num_topics'])
    )

#-------------------------- 3. Add Topic-Tags -------------------------

df_top_topic_per_doc = best_model.create_top_topic_per_doc_df(best_model.corpus)

df_classified = pd.merge(df, df_top_topic_per_doc, left_index=True, right_index=True)

print('\n3. Added topic-tags')

#-------------------------- 4. Save Data -------------------------

SAVE_DIR = TOP_DIR + '/data/topicmodels/testing/' 
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

df_classified.to_csv(SAVE_DIR + 'data.csv', index=False)

with open(SAVE_DIR + 'model.pickle', 'wb') as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f'\n4. Saved Classified Papers and Best LDA Models in {SAVE_DIR}')