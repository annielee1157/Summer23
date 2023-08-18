# from ast import literal_eval
import pandas as pd
# import numpy as np
import datetime as dt
# import plotly
# from datetime import datetime
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import dash, time, math
from dash import dcc
from dash import html

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
# from pathlib import Path

from app import app
# import config

import dash_cytoscape as cyto # if this doesn't work, pip install dash-cytoscape==0.2.0
# import plotly.express as px

# import matplotlib.cm as cm # if it doesn't work, pip install matplotlib


airports = pd.read_csv('data/airport_locations.csv')

carrier = 'UAL'
startdate = pd.to_datetime('2023-01-15')
enddate = pd.to_datetime('2023-03-15')

dataConstraints = pd.read_csv('data/historical_view/raw_queried_data_w_constraints.csv')




def update_elements(carrier, start, end):
	"""
	carrier: string
	start: datetime object; use pd.to_datetime()
	end: datetime object
	returns: elements of graph with corresponding stylesheet
	"""

	dataConstraints = pd.read_csv('data/historical_view/raw_queried_data_w_constraints.csv')
	filtered = dataConstraints[dataConstraints['carrier'] == carrier]
	filtered = filtered[filtered['departure_runway_metered_time_value'].notna()]
	filtered['departure_runway_metered_time_value'] = pd.to_datetime(filtered['departure_runway_metered_time_value'])
	
	filtered = (filtered[filtered['departure_runway_metered_time_value'] >= start])
	filtered = (filtered[filtered['departure_runway_metered_time_value'] <= end])
	apreqNodes = filtered['arrival_aerodrome_icao_name']
	


	nodes = []

	stylesheet = [{
	'selector': 'node',
	'style': {
	'content': 'data(label)',
	'height': 70,
	'width': 70
	}
	}, 
	
	{
	'selector': '.regular',
	'style': {
		'height': 5,
		'width': 5,
		'content': 'data(label)',
		'color': '#D3D3D3',
		'background-color': 'Gray',
		'line-color': 'Gray'
	}
	}]

	
	sizes = {}
	

	for airport in apreqNodes:
		if airport in sizes.keys():
			sizes.update({airport: sizes.get(airport) + 1})
		else:
			sizes.update({airport: 1})


	apreqNodes = filtered['arrival_aerodrome_icao_name'].unique()
	# update stylesheet
	for each in sizes.keys():
		weightedsize = {
		'selector': '.' + 'bin' + str(sizes.get(each)),
		'style': {
			'height': .35 * sizes.get(each),
			'width': .35 * sizes.get(each),
			'background-color': ' #964B00'
			}
		}

		stylesheet.append(weightedsize)
	
	for i in range(len(airports['icao'])):
		if airports['icao'].get(i) not in apreqNodes:
			temp = {'data': {'id': airports['icao'].get(i), 'label': airports['icao'].get(i)[1:], 'apreq': str(0)}, 
				'position': {'x': airports['long'].get(i) * 100, 'y': airports['lat'].get(i) * -100},
				'classes': 'regular',
				'locked': True,
			}

			
			nodes.append(temp)
		else:
			temp = {'data': {'id': airports['icao'].get(i), 'label': airports['icao'].get(i)[1:], 'apreq': str(sizes.get(airport))}, 
				'position': {'x': airports['long'].get(i) * 100, 'y': airports['lat'].get(i) * -100},
				'classes': 'bin' + str(sizes.get(airport)),
				'locked': True,
			}

			nodes.append(temp)
	
	return nodes, stylesheet

ele, style = update_elements(carrier, startdate, enddate)


# Set the layout
def render() -> html.Div:
	return html.Div([

		html.H4(
			'Delay Map',
			className = 'text-left mr-5'),

		dbc.Col(html.H6('Carrier', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px', 'value': 'DAL'}), width = '3'),
		dbc.Col(html.Div([dcc.Dropdown(['DAL', 'UAL', 'AAL', 'RPA'], placeholder="Select a carrier", id='select-carrier', value='AAL'), html.Div(id='dd-output-container')]), 
										style = {'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}),

		
		dbc.Row([

			dbc.Col(html.H6('Start Date', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = '3'),

			dbc.Col(
				[
					dcc.DatePickerSingle(
						id="pick-start-date",
						min_date_allowed=dt.date(2023, 1, 1),
						max_date_allowed=dt.date(2023, 6, 1),
						date=dt.date(2023, 1, 1),
						style={'maxWidth': '300px'},
						clearable=False,
					),
				], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}, width = 'auto'),
			
			dbc.Col(html.H6('End Date', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = '3'),

			dbc.Col(
				[
					dcc.DatePickerSingle(
						id="pick-end-date",
						min_date_allowed=dt.date(2023, 1, 1),
						max_date_allowed=dt.date(2023, 6, 1),
						date=dt.date(2023, 8, 10),
						style={'maxWidth': '300px'},
						clearable=False,
					),
				], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}, width = 'auto'),


		]),

		cyto.Cytoscape(
			id='network-graph-apreq',
			layout={'name': 'preset', 'plot_bgcolor': 'Black',},
			style={'width': '100%', 'height': '700px'},
			elements= ele,
			stylesheet= style
		),
		html.P(id='cytoscape-tapNodeData-output-apreq'),
		
	], className='six columns'),


#Callback function updates display.
@app.callback(
	Output('network-graph-apreq','elements'),
	Output('network-graph-apreq','stylesheet'),
	Input('select-carrier', 'value'),
	Input('pick-start-date', 'date'),
	Input('pick-end-date', 'date'),

)
def update_output(value, startdate, enddate):
	start = pd.to_datetime(startdate)
	end = pd.to_datetime(enddate)
	elements, stylesheet = update_elements(value, start, end)
	return elements, stylesheet

@app.callback(Output('cytoscape-tapNodeData-output-apreq', 'children'),
			  Input('network-graph-apreq', 'tapNodeData'))
def displayTapNodeData(data):
	if data:
		return "The selected airport, " + data['label'] + ", has " + data['apreq'] + " APREQs in the selected date range." 
