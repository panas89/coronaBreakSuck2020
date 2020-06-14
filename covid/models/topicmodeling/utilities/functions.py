# Misc
from functools import partial
from collections import defaultdict

# Array & Dataframes
import pandas as pd
import numpy as np

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
my_pal = sns.color_palette(n_colors=20)

# Utilities
from covid.data.nltkPreProc import preprocess_text

# Sklearn
from sklearn.metrics import confusion_matrix


#------------------------ PreProcessing Methods -------------------------

def load_paper_data(file_path, class_cols, bad_phrases, bad_tokens, drop_nan_text=False, from_date='2020-01-01'):
    
    # Load relevant data
    paper_info_cols = ['sha', 'title', 'abstract', 'text', 'publish_time']
    df = pd.read_csv(file_path, 
                     usecols=paper_info_cols + class_cols, 
                     parse_dates=['publish_time'])
    NUM_COVID_PAPERS = len(df)
    
    # Select only papers published from/after 'from_date'
    df = df[df.publish_time >= from_date]

    # Query class cols and then drop them
    for c in class_cols:
        df = df.loc[df[c]==1]
    df = df.drop(class_cols, axis=1)
    
    # Treat NaNs
    if drop_nan_text:
        df.dropna(subset=['text'], axis=0, inplace=True)
    df[['title', 'abstract', 'text']] = df[['title', 'abstract', 'text']].fillna('')
    
    # Create meta col
    df['abstract'] = df['abstract'].apply(lambda x: x[len('abstract'):] 
                                          if x[:len('abstract')].lower() == 'abstract' 
                                          else x) # remove string 'abstract' from abstract col
    df['meta'] = df['title'] + ' ' + df['abstract']
    
    # Clean cols
    text_cleaner = partial(preprocess_text,
                           bad_phrases=bad_phrases, 
                           bad_tokens=bad_tokens, 
                           min_char=3, 
                           pos_tags=['v', 'n'], 
                           remove_dig=True, 
                           replace_num=False,
                           replace_contr=False)

    df['clean_title'] = df['title'].apply(lambda x: text_cleaner(x))
    df['clean_abstract'] = df['abstract'].apply(lambda x: text_cleaner(x))
    df['clean_text'] = df['text'].apply(lambda x: text_cleaner(x))
    df['clean_meta'] = df['meta'].apply(lambda x: text_cleaner(x))

    print('Fraction of selected papers: {}/{}'.format(len(df), NUM_COVID_PAPERS))
    
    return df.reset_index(drop=True)


#------------------------ DataFrame Methods -------------------------


def process_pcf_data(df, bad_phrases, bad_tokens, drop_nan_text=False, from_date='2020-01-01'):
    
    NUM_PAPERS = len(df)
    
    # Select only papers published from/after 'from_date'
    df = df[df.publish_time >= from_date]
    
    # Treat NaNs
    if drop_nan_text:
        df.dropna(subset=['text'], axis=0, inplace=True)
    df.loc[:,['title', 'abstract', 'text']] = df[['title', 'abstract', 'text']].fillna('')
    
    # Create meta col
    df.loc[:,'abstract'] = df.loc[:,'abstract'].apply(lambda x: x[len('abstract'):] 
                                          if x[:len('abstract')].lower() == 'abstract' 
                                          else x) # remove string 'abstract' from abstract col
    df.loc[:,'meta'] = df['title'] + ' ' + df['abstract']
    
    # Clean cols
    text_cleaner = partial(preprocess_text,
                           bad_phrases=bad_phrases, 
                           bad_tokens=bad_tokens, 
                           min_char=3, 
                           pos_tags=['v', 'n'], 
                           remove_dig=True, 
                           replace_num=False,
                           replace_contr=False)

    df.loc[:,'clean_title'] = df['title'].apply(lambda x: text_cleaner(x))
    df.loc[:,'clean_abstract'] = df['abstract'].apply(lambda x: text_cleaner(x))
    df.loc[:,'clean_text'] = df['text'].apply(lambda x: text_cleaner(x))
    df.loc[:,'clean_meta'] = df['meta'].apply(lambda x: text_cleaner(x))

    print('Fraction of selected papers: {}/{}'.format(len(df), NUM_PAPERS))
    
    return df #.reset_index(drop=True)


#------------------------ DataFrame Methods -------------------------


def gs_models_to_df(gs_models, scorers):
    data = defaultdict(list)
    for model, d in gs_models.items():
        data['model'].append(model)
        for param_name, value in d['params'].items():
            data[param_name].append(value)
        for s in scorers:
            data[s].append(d[s])
        

    df_gs = pd.DataFrame.from_dict(data)\
              .set_index('model')\
              .sort_values(by=scorers[0], ascending=False)

    return df_gs

def gs_models_to_df_multi(gs_models_dict, scorers):
    to_concat = []
    for name, gs_models in gs_models_dict.items():
        temp_df = gs_models_to_df(gs_models, scorers).reset_index()
        temp_df['name'] = name
        to_concat.append(temp_df)

    df = pd.concat(to_concat,axis=0).set_index(['name', 'model'])
    return df



#------------------------ Plotting Methods -------------------------


def print_evaluation_graph(gs_models, scorers):
    data = defaultdict(list)
    for model, d in gs_models.items():
        data['model'].append(model)
        for param_name, value in d['params'].items():
            data[param_name].append(value)
        for s in scorers:
            data[s].append(d[s])
            
    fig, axs = plt.subplots(2,1,sharex=True, figsize=(10,7))
    fig.suptitle('Model Evaluation', fontsize=16)
    
    for i,s in enumerate(scorers):
        axs[i].plot(data['model'], data[s], '-o', color=my_pal[i])
        axs[i].set_ylabel(f'{s}', size=14)
    
    return


def print_evaluation_graph_multi(gs_models_dict, scorers):
    n_rows = len(scorers)
    fig, axs = plt.subplots(n_rows, 1, sharex=False, figsize=(10,10))
    fig.suptitle('Model Evaluation', fontsize=16)
    # loop over name of lda type and gs_models for that lda
    for i,(lda_name,gs_models) in enumerate(gs_models_dict.items()):
        # convert data in gs_models into appropriate format for plotting
        data = defaultdict(list)
        for model, d in gs_models.items():
            data['model'].append(model)
            for param_name, value in d['params'].items():
                data[param_name].append(value)
            for s in scorers:
                data[s].append(d[s])
                
        # if n_rows=1 then axs is NOT an array
        if n_rows == 1:
            s = scorers[0]
            axs.plot(data['model'], data[s], '-o', color=my_pal[i], label=lda_name)
            axs.set_ylabel(f'{s}', size=14)
        else:
            for j,s in enumerate(scorers):
                axs[j].plot(data['model'], data[s], '-o', color=my_pal[i], label=lda_name)
                axs[j].set_ylabel(f'{s}', size=14)
       
    plt.legend(loc='right', prop={'size': 15})
    plt.xticks(rotation=90)
        
    return


def print_topic_freq(df_top_topic_per_doc):
    
    topic_freq = df_top_topic_per_doc.dominant_topic.value_counts().sort_index()
    plt.figure(figsize=(10,7))
    barplot = plt.bar(list(topic_freq.index), list(topic_freq.values))

    ax = plt.gca()
    for i,bar in enumerate(barplot):
        bar.set_color(my_pal[i])
        height=bar.get_height()
        x = bar.get_x()
        width=bar.get_width()
        ax.text(x + 0.4*width, height+2,f'{height}',color='black')

    plt.xlabel('topic no', size=14); plt.ylabel('counts', size=14)
    plt.title("Top-Topic Frequencies", size=18)
    plt.xticks(rotation=0)
    
    return


def plot_confusion_matrix(lda):
    cf_matrix = confusion_matrix(lda.predict_topics(lda.corpus), lda.predict_topics(lda.test_corpus))
    plt.figure(figsize=(10,7))
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    return 




# def print_evaluation_graph_multi(gs_models_dict, scorers):
#     n_rows = len(scorers)
#     fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(10,10))
#     fig.suptitle('Model Evaluation', fontsize=16)
#     for i,(lda_class,gs_models) in enumerate(gs_models_dict.items()):
#         data = defaultdict(list)
#         for model, d in gs_models.items():
#             data['model'].append(model)
#             for param_name, value in d['params'].items():
#                 data[param_name].append(value)
#             for s in scorers:
#                 data[s].append(d[s])

#         for j,s in enumerate(scorers):
#             axs[j].plot(data['model'], data[s], '-o', color=my_pal[i], label=lda_class)
#             axs[j].set_ylabel(f'{s}', size=14)
            
#     plt.legend(loc=(1,1.9), prop={'size': 15})
#     plt.xticks(rotation=90)
        
#     return
