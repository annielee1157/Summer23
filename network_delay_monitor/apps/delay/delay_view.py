from ast import literal_eval
import pandas as pd
import numpy as np
import datetime as dt
import plotly
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash, time, math
from dash import dcc
from dash import html

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from pathlib import Path

from app import app
import config

# if this doesn't work, pip install dash-cytoscape==0.2.0

import dash_cytoscape as cyto 
import plotly.express as px

import matplotlib.cm as cm 


airports = pd.read_csv('data/airport_locations.csv')

# placeholders
carrier = 'AAL'
date = '2023-02-22'

selectData = pd.read_csv(f'data/historical_view/{carrier}_{date}_processed_data.csv')

weights = selectData.groupby('arrival_aerodrome_icao_name').sum(numeric_only=True).loc[:, 'arrival_delay_time']



def cmap_at_scale(cmap, bins):

    # cmap:  mpl or cmocean colormap 
    # bins :  ascending ordered list  of floats in [0,1] that defines the scale for a plotly colorscale derived from cmap
    # returns: scaled list of hexcolors, one for each bin
    
    if not isinstance(bins, (list, np.ndarray)):
        raise ValueError('bins should be a 1d list or  an array')
    if not  (0 <= np.asarray(bins).min() <= 1) or  not  (0 <= np.asarray(bins).max() <= 1):
        raise ValueError('The elements in bins should be in [0,1]')
    scale  = sorted(bins)
    colors = cmap(scale)[:, :3]
    colors = (255*colors+0.5).astype(np.uint8)
    hexcolors = [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in colors]
    return hexcolors

colormap = cmap_at_scale(cm.Blues, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


def findColor(name, weightarr):
    """
    changes every update; the purpose is to find the stylesheet class name to style a node to
    name: name of the airport
    weightarr: series with airport names as indexes in format 'KXXX', and total delay as values
    returns: corresponding bin name for the color of the node based on the airports delay weight
    """
    minW = weightarr.min()
    maxW = weightarr.max()

    if name not in weightarr.keys():
         return 'Blank'
    
    score = weightarr.get(name)

    zscore = (score-minW)/(maxW-minW)
    if 0 <= zscore and zscore < 0.1:
        return 'bin1'
    elif zscore < 0.2:
        return 'bin2'
    elif zscore < 0.3:
        return 'bin3'
    elif zscore < 0.4:
        return 'bin4'
    elif zscore < 0.5:
        return 'bin5'
    elif zscore < 0.6:
        return 'bin6'
    elif zscore < 0.7:
        return 'bin7'
    elif zscore < 0.8:
        return 'bin8'
    else:
        return 'bin9'
    


# colorselectors is a list of ditionaries to pass into the stylesheet for the network graph
colorselectors = [{
    'selector': 'node',
    'style': {
    'content': 'data(label)',
    'height': 70,
    'width': 70,
    }
    }, 
    {
    'selector': '.Blank',
    'style': {
        'height': 5,
        'width': 5,
        'background-color': 'Gray',
        'line-color': 'Gray'
    }
}]

for i in range(len(colormap)):
    mapped = {
        'selector': '.' + 'bin' + str(i),
        'style': {
            'background-color': colormap[i],
            'line-color': colormap[i]
        }
    }
    colorselectors.append(mapped)



# set up nodes and edges to pass into graph 
nodes = []
for i in range(len(airports['icao'])):
    temp = {'data': {'id': airports['icao'].get(i), 'label': airports['icao'].get(i)[1:]}, 
        'position': {'x': airports['long'].get(i) * 100, 'y': airports['lat'].get(i) * -100},
        'classes': 'bin8',
        'classes': findColor(airports['icao'].get(i), weights),
        'locked': True,
    }
    nodes.append(temp)
edges = []
for i in range(len(selectData['gufi'])):
    temp = {'data': {'source': selectData['departure_aerodrome_icao_name'].get(i), 'target': selectData['arrival_aerodrome_icao_name'].get(i)}}
    edges.append(temp)
ele = nodes + edges



# Set the layout
def render() -> html.Div:
	return html.Div([

        
		#html.Div(children='App'),
        html.H4(
            'Delay Map',
            className = 'text-left mr-5'),

		dbc.Col(html.H6('Carrier', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = '3'),
		dbc.Col(html.Div([dcc.Dropdown(['DAL', 'UAL', 'AAL', 'RPA'], placeholder="Select a carrier", id='select-carrier', value='AAL'), html.Div(id='dd-output-container')]), 
                                        style = {'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}),

        dbc.Col(html.H6('Date', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = '3'),

        dbc.Col(
            [
                dcc.DatePickerSingle(
                    id="pick-date",
                    min_date_allowed=dt.date(2023, 1, 1),
                    max_date_allowed=dt.date(2023, 6, 1),
                    date=dt.date(2023, 3, 1),
                    style={'maxWidth': '300px'},
                    clearable=False,
                ),
            ], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}, width = 'auto'),

		cyto.Cytoscape(
            id='network-graph',
            layout={'name': 'preset', 'plot_bgcolor': 'Black',},
            style={'width': '100%', 'height': '700px'},
            elements= ele,
            stylesheet= colorselectors
        ),
        

	
	
    ], className='six columns'),

	

    

# Callback function updates display.
@app.callback(
    Output('network-graph','elements'),
    Input('pick-date', 'date'),
    Input('select-carrier', 'value'),
)
def update_carrier(date, value):
    carrier = value
    date = date
    selectData = pd.read_csv(f'data/historical_view/{carrier}_{date}_processed_data.csv')

    # debugging statements to show which data file is being used, feel free to comment out
    print('\nUPDATE')
    print(f'data/historical_view/{carrier}_{date}_processed_data.csv')

    weights = selectData.groupby('arrival_aerodrome_icao_name').sum(numeric_only=True).loc[:, 'arrival_delay_time']

    nodes = []

    for i in range(len(airports['icao'])):
      temp = {'data': {'id': airports['icao'].get(i), 'label': airports['icao'].get(i)[1:]},
              'position': {'x': airports['long'].get(i) * 100, 'y': airports['lat'].get(i) * -100},
              'classes': 'bin8',
              'classes': findColor(airports['icao'].get(i), weights),
              'locked': True,
              
              }
      nodes.append(temp)

    edges = []

    for i in range(len(selectData['gufi'])):
        temp = {'data': {'source': selectData['departure_aerodrome_icao_name'].get(i), 'target': selectData['arrival_aerodrome_icao_name'].get(i)}}
        edges.append(temp)
    
    elements = nodes + edges
    return elements

