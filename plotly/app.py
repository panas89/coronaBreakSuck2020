import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from covid.models.paperclassifier.frontpaperclassifier import FrontPaperClassifier
from api import *
import plotly.express as px
import json
import dash_bootstrap_components as dbc

#------------------------------  Global Vars  ------------------------------ 

fpc = FrontPaperClassifier(km_path='../covid/models/paperclassifier/interest.yaml')
CLASSES = fpc.classes
SUBCLASSES = fpc.get_subclasses('risk_factor')
RISKFACTORS_TO_IDS = load_pickle('./data/rf_to_ids_metadata.pkl')

df = pd.read_csv('./data/clean_metadata.csv',
                 usecols=['title','year','cord_uid', 'url', 'abstract', 'publish_time'], 
                 parse_dates=['publish_time'])

df_freq = create_df_freq(df, queries_to_ids=RISKFACTORS_TO_IDS)


# Markdown
markdown_project_description = '''
### Project Description
Say a few things about the project (e.g. scope, sources, updates, etc)
'''


# Instantiate Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# Define server for deployment with Heroku
server = app.server





#---------------------------- APP LAYOUT ----------------------------

app.layout = html.Div(children=[
    html.Div(children=[    
        html.H1(children='A Search Engine for Covid-19 Publications'),
        dcc.Markdown(children=markdown_project_description),
        html.Div(children=[
            dcc.Graph(
                id='bar-plot', 
                style={'width': '50%', 'display': 'inline-block'}
                ),
            dcc.Graph(
                id='pie-chart', 
                style={'width': '50%', 'display': 'inline-block'}
                )
        ]),
        html.Div(children=[
            dbc.Row([
                dbc.Col(html.H4('Domain')),
                dbc.Col(html.H4('Topic'))
            ]),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='domain',
                    options=create_dropdown_list(CLASSES),
                    value='risk_factor',
                    style={'width': '50%', 'display': 'inline-block'}
                    )
                ),
                dbc.Col(dcc.Dropdown(
                    id='subclass',
                    options=create_dropdown_list(SUBCLASSES),
                    value='gender',
                    style={'width': '50%', 'display': 'inline-block'}
                    )
                )
            ])
        ]),
        html.Table(id='table')
    ])
],className="col-10 offset-1")


#-------------------------  CALLBACKS --------------------------

@app.callback(
    Output('bar-plot', 'figure'),
    [Input('domain', 'value')])
def update_bar_plot(domain):
    x,y = create_bar_data(df, RISKFACTORS_TO_IDS, 'gender')
    x,y = filter_bar_data(x,y, 2015)

    return {'data': [{'x': x, 'y': y, 'type': 'bar', 'name': domain}],
            'layout': {'title': f'Publication Frequencies By {domain.title()}',
                       'clickmode': 'event+selected'}
        }
    
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('bar-plot', 'hoverData')])
def update_pie_chart(hoverData):
    try:
        year = hoverData['points'][0]['x']
    except:
        year=2020
    df_pie = create_pie_chart_df(df_freq, year)
    fig = px.pie(df_pie, values='Num', names='Risk Factor', 
             title=f'Number of Publications per Risk Factor in {year}',
             color_discrete_sequence=px.colors.sequential.RdBu
            )
    return fig
    

@app.callback(
    Output('table', 'children'),
    [Input('subclass', 'value')])
def updata_table(subclass):
    ids = RISKFACTORS_TO_IDS[subclass]
    df_filtered = df[df.cord_uid.isin(ids)].loc[:, ['year', 'title', 'abstract']]

    return generate_table(df_filtered)


#---------------------------  MAIN  ------------------------------------ 

if __name__ == '__main__':
    app.run_server(debug=True)