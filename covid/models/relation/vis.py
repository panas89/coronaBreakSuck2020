"""
This is the module to store all the specialized functions for relation visualization
"""
import pandas as pd
import ast

def preprocess_df(df_r):
    """
    Prepare the raw relation dataframe from the data file (i.e., classified_merged_covid_relation.csv)
    to several dataframe for visualization
    
    :param df_r (dataframe): raw dataframe
    """
    # ========================================
    # preprocess
    # ========================================

    # ---------- 1. only select the paper published after covid time in 2020 Feb
    df_r['publish_time'] = pd.to_datetime(df_r['publish_time'])
    df_p = df_r.loc[df_r['publish_time'] > '2020-02-01', :]
    df_p = df_p.drop(columns='Unnamed: 0')
    df_p.head()

    # ---------- 2. Reformulate the dataframe
    # a. use a unique keyterm and a unique relation as a row, along with the paper title, sha, publish time, location.
    # This mean that multiple rows can have the same paper title. 
    # b. only keep the useful relations that make sense to readers. Rename the relation
    # parameters
    df_new = []
    # accept_relations = {'has part': 'is related to',
    #                     'part of': 'is related to',
    #                     'said to be the same as': 'is', 
    #                     'instance of': 'is'}
    accept_relations = {'has part': 'is related to',
                        'part of': 'is related to',}
    for i in range(0, df_p.shape[0]):
        s = df_p.iloc[i,:]

        # basic info 
        sha = s['sha']
        title = s['title']
        publish_time = s['publish_time']
        location = s['location']

        # extract relations
        rs = s['relations']
        if isinstance(rs, str):
            rx = ast.literal_eval(rs)

            # create content string
            for c in rx:
                if c[2][0] in accept_relations.keys():
                    content = [c[1], accept_relations[c[2][0]], c[2][1], 
                               sha, title, publish_time, location]
                    df_new.append(content)

    # create dataframe
    df_new = pd.DataFrame(df_new, columns=['keyword', 'relation', 'probability', 'sha',
                                           'title', 'publish_time', 'location'])
    return df_new
    
    
def preproces_for_kwspecific_plot(df_new, kw, relations=['is related to']):
    """
    Preprocess the dataframe for keyword-specific plot
    
    :param df_new (dataframe): the preprocessed dataframe
    :param kw (string): the keywrod string that the user is interested in
    :param relations (list): a list of relations that are interested. We will 
                            stick with the default value
    """
    # create groups
    df_grps = []
    for relation in relations:
        # sub-df
        df_sg = df_new.loc[(df_new['keyword'] == kw) & (df_new['relation']==relation), :]

        # aggregated data
        grp = df_sg.groupby(df_sg['publish_time'].dt.strftime('%B'))['probability']
        statistics = [grp.mean(), grp.std(), grp.sem(), grp.count()]
        df_grp = pd.DataFrame(statistics).transpose()
        df_grp.columns = ['proba_mean', 'proba_std', 'proba_stderr', 'n']
        df_grp = df_grp.fillna(0)

        # new column
        df_grp['publish_month'] = df_grp.index
        df_grp['relation'] = relation
        df_grp['proba_min'] = df_grp['proba_mean'] - df_grp['proba_stderr']
        df_grp['proba_max'] = df_grp['proba_mean'] + df_grp['proba_stderr']

        # append
        df_grps.append(df_grp)
    df_grps = pd.concat(df_grps)
    return df_grps
    

def preprocess_for_multiple_kw_visualization(df_new, kw_interest):
    """
    Preprocess the dataframe for a list of interested keywords visualization.
    :param df_new (dataframe): the preprocessed dataframe
    :param kw_interest (list): a list of interested keywords to be visualized
    """
    df_new_p = df_new.copy()
    df_new_p = df_new_p[df_new_p['keyword'].isin(kw_interest)]
    df_new_p.head()
    return df_new_p
    
    
    
    
    