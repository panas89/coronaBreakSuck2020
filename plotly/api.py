import pickle
import dash_html_components as html 
import pandas as pd
import dash_bootstrap_components as dbc 
#----------------------- DATA --------------------------------
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_bar_data(df, subclass_to_ids, subclass):
    
    df_agg = df[df.cord_uid.isin(subclass_to_ids[subclass])].set_index('publish_time')\
            .groupby(lambda x: x.year)[['title']]\
            .count()\
            .rename({'title': 'paper_freq'}, axis=1)
    
    x = df_agg.index.to_list()
    y = list(df_agg.paper_freq.values)
    
    return x,y

def filter_bar_data(x, y, from_year):
    
    i=0
    year = x[0]
    while year < from_year:
        year = x[i]
        i += 1
    
    return x[i:], y[i:]

def create_df_freq(df, queries_to_ids):
    to_concat = []
    for ids in queries_to_ids.values():
        df_agg = df[df.cord_uid.isin(ids)].set_index('publish_time')\
                .groupby(lambda x: x.year)[['abstract']]\
                .count()\
                .rename({'abstract': 'paper_freq'}, axis=1)
        to_concat.append(df_agg)
    df_concat = pd.concat(to_concat, axis=1).fillna(0).astype('int')
    df_concat.columns = list(queries_to_ids.keys())
    
    return df_concat


def create_pie_chart_df(df_freq, year):
    
    return df_freq[df_freq.index==year]\
                 .T\
                 .reset_index()\
                 .rename({'index':'Risk Factor', year:'Num'}, axis=1)

#------------------------ HTML -------------------------------- 

def create_dropdown_list(strings):

    return list({'label': s.title(), 'value': s} for s in strings)

def create_boostrap_dropdown_list(strings):

    return [dbc.DropdownMenuItem({'child': s.title(), 'value': s}) for s in strings]


def generate_table(df, max_rows=50):

    return [
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))
        ])
    ]

def df_to_table(df):
    return html.Table(
        [html.Tr([html.Th(col) for col in df.columns])]
        + [
            html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
            for i in range(len(df))
        ]
    )