import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import os
import datetime as dt
import pandas as pd
import numpy as np
import torch
from kedro_datasets.pickle import PickleDataSet
from torch_geometric.data import Data
import joblib

import plotly.io as pio
pio.templates.default = "plotly_white"
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import apps.historical_view.query_hist_graph as query_utils
import apps.prediction.query_prediction as prediction_utils
from app import app
import config

NAS_8 = ('KEWR','KJFK','KLGA','KATL','KBOS','KIAD','KBWI','KDCA','KDFW','KSFO','KLAX','KSEA')

NAS_30 = ('KEWR', 'KJFK', 'KLGA', 'KATL', 'KBOS', 'KIAD', 'KBWI', 'KDCA', 'KSFO', 'KLAX', 'KDTW', 'KDFW', 'KSEA', 'KSLC', 'KCLT', 'KDEN', 'KORD', 'KLAS', 'KMCO', 'KMIA', 'KPHX', 'KIAH', 'KFLL', 'KMSP', 'KPHL', 'KSAN', 'KTPA', 'KAUS', 'KBNA', 'KMDW')

LOCATIONS = pd.read_csv('data/airport_locations.csv', index_col=0)

DATA_START_DATE = config.PARAMS['start_time']
DATA_END_DATE = config.PARAMS['end_time']

PAGE_TITLE = 'Cumulative Arrival Delay Prediction - Sliding Window'

PAGE_DESCRIPTION = '''
This page displays the prediction of cumulative arrival delay of a given date. The cumulative arrival delay is predicted from a trajectory of previous cumulative arrival delay. The cumulative arrival delay is predicted on selected airports from the national airspace system (NAS). On this page, the user may select the date and the hour to visualize the network delay prediction.
'''

PAGE_NAME = 'prediction'

CARRIER = config.PARAMS['carrier']
DE_PARAMS = config.PARAMS['de_params']
MODEL_NAME = config.PARAMS['model_name']
MODEL_PARAMS = config.PARAMS['model_params']

# Set the layout
def render() -> html.Div:
    
    # row element for the title
    title_row = dbc.Row(
        dbc.Col(
        [
            dcc.Markdown(f'## {PAGE_TITLE}'),
            dcc.Loading(id='loading-prediction'),
        ],
        style={'display': 'flex'},
        )
    )

    # row element for the description of the page
    descr_row = dbc.Row(
        dbc.Col(dcc.Markdown(f'*{PAGE_DESCRIPTION}*')),
        style={'margin-top': '10px'},
    )

    # row element for the selection objects
    selection_row = dbc.Row(
        [
            dbc.Col(
                dcc.Markdown('##### Prediction Date')
            ),
            dbc.Col([
                dcc.DatePickerSingle(
                    id='pred-date',
                    min_date_allowed=DATA_START_DATE,
                    max_date_allowed=DATA_END_DATE,
                    date=dt.date(2023,3,1),
                    style={'maxWidth': '300px'},
                    clearable=False,
                ),
            ]),
            dbc.Col(
                dcc.Markdown('##### Hour of Prediction Date')
            ),
            dbc.Col([
                dcc.Dropdown(
                    list(range(24)),
                    value=12,
                    id='pred-hour',
                    clearable=False,
                ),
            ]),
        ],
        style={'margin-top': '10px'},
    )

    # row element for a printout of the selected values
    print_select_row = dbc.Row(
        [
            dbc.Col(html.Div(id='chosen-date-pred')),
            dbc.Col(html.Div(id='chosen-hour-pred')),
        ], 
        style={'margin-top': '10px'},
    )

    return html.Div([
        title_row,
        descr_row,
        selection_row,
        print_select_row,

        dbc.Row(dcc.Markdown('', id='valid-inputs'), style={'margin-top': '10px'}),
        dbc.Row(
            dbc.Col(dcc.Graph(id='graph-compare')),
            style={'margin-top': '10px'},
            ),
        dbc.Row(dcc.Markdown('', id='pred-info')),
        dbc.Row(dcc.Graph(id='data-table')),
        dbc.Row(dcc.Graph(id='graph-traj'),
                style={'margin-top': '10px'},
            ),
    ])

def _append_statement(original, new):
    original.append(new)
    original.append('\n')
    return original

def update_pred_figure(date, hour, carrier=CARRIER, de_params=DE_PARAMS, model_name=MODEL_NAME, model_params=MODEL_PARAMS):

    # check if the date selected is in the appropriate range (model predictions 
    # do not rely on this, but comparison to stored data does)
    date_obj = dt.date.fromisoformat(date)
    if date_obj <= DATA_START_DATE:
        if date_obj == DATA_START_DATE:
            if (hour-n_prev) < 0:
                return 
        else:
            return
    if date_obj >= DATA_END_DATE:
        if date_obj == DATA_END_DATE:
            if (hour+n_pred) > 24:
                return
        else:
            return

    model_pred_stats = []
    model_pred_stats = _append_statement(model_pred_stats, '**Model Prediction Information:**')

    hr_shift = de_params['hr_shift']
    model_pred_stats = _append_statement(model_pred_stats, f'- This model started the cumulative arrival delay computation at **{hr_shift}** hours after midnight.')

    n_prev = de_params['n_prev']
    model_pred_stats = _append_statement(model_pred_stats, f'- This model used a history of **{n_prev}** hours (includes the hour selected).')

    n_pred = de_params['n_pred']
    model_pred_stats = _append_statement(model_pred_stats, f'- This model predicts the cumulative arrival delay at each airport **{n_pred}** hours into the future.')


    # specify name of the raw data file depending on if the carrier is specified
    if carrier is not None:
        raw_data_file = f'./data/{PAGE_NAME}/{carrier}_{DATA_START_DATE}_{DATA_END_DATE}_raw_queried_data.csv'
    else:
        raw_data_file = f'./data/{PAGE_NAME}/{DATA_START_DATE}_{DATA_END_DATE}_raw_queried_data.csv'

    # query raw data if file does not already exist
    if not os.path.exists(raw_data_file):
        df_raw = query_utils.raw_query(
            NAS_30, 
            DATA_START_DATE, 
            DATA_END_DATE, 
            carrier=carrier,
            tab=PAGE_NAME,
        )
    # otherwise, read data from existing file
    else:
        df_raw = pd.read_csv(raw_data_file, index_col=0, parse_dates=['arrival_stand_scheduled_time', 'arrival_stand_actual_time'])
    print("\nRAW data:\n", df_raw.head())

    # compute the arrival delay times from each flight and group by hour
    delay_df = query_utils.compute_delay_from_raw_data(df_raw, carrier=carrier, tab=PAGE_NAME)
    print("\nProcessed data:\n", delay_df.head())

    # compute the cumulative arrival delay for the graph representation
    graph_info, _ = prediction_utils.create_node_data_around_date(
        delay_df,
        date_obj,
        hr_shift,
    )
    print("\nGraph Data:\n", graph_info.head())
    n_nodes = len(NAS_30)

    if model_name == "GraphSAGE":
        from utils.my_model import GraphSAGE
        model_input = graph_info.iloc[24-hr_shift+hour-n_prev+1:24-hr_shift+hour+1].to_numpy()

        with open(f"./apps/prediction/models/{model_name}_scaler.pkl", 'rb') as f:
            scaler = joblib.load(f)
        model_input = scaler.transform(model_input)
        model_input = torch.tensor(model_input, dtype=torch.float32)
        # print("Model Input:\n", model_input)
        
        true_output = torch.tensor(graph_info.iloc[24-hr_shift+hour+n_pred-1])
        # print("True Output:\n", true_output)

        pkl_data = PickleDataSet(filepath='./apps/prediction/models/NAS-de_dataset.pkl')
        model_data = pkl_data.load()
        edge_index = model_data['edge_index']
        
        model = GraphSAGE(
            in_channels=n_prev,
            out_channels=1,
            **model_params,
            )
        model.load_state_dict(torch.load('./apps/prediction/models/simple-gnn_best_model.pt'))
        data_input = Data(x=model_input.permute(1,0), edge_index=edge_index)
        model_output = model.predict(data_input).reshape(-1,n_nodes)
        print("Model Output:\n", model_output)
        model_output = scaler.inverse_transform(model_output).reshape(-1)
        print("Model Output:\n", model_output)
        d = {}
        loc_list = list(NAS_30)
        loc_list.sort()
        d['icao'] = loc_list
        d['model_output'] = model_output
        d['true_output'] = true_output
        df_model = pd.DataFrame(d)
    else:
        return
    
    df_loc_red = LOCATIONS[LOCATIONS['icao'].isin(NAS_30)]
    df_merge = df_model.merge(df_loc_red, on='icao')
    df_merge['difference'] = df_merge['model_output'] - df_merge['true_output']
    pos_mask = df_merge['difference']>=0
    neg_mask = ~pos_mask
    df_merge.loc[pos_mask, 'positive'] = np.abs(df_merge.loc[pos_mask, 'difference'])
    df_merge.loc[neg_mask, 'positive'] = 0.0
    df_merge.loc[neg_mask, 'negative'] = np.abs(df_merge.loc[neg_mask, 'difference'])
    df_merge.loc[pos_mask, 'negative'] = 0.0
    
    display_table = df_merge[['icao', 'true_output', 'model_output', 'difference']].round()

    n_cols = 3
    fig = make_subplots(
        rows=1, 
        cols=n_cols, 
        specs=[[{'type': 'mapbox'} for _ in range(n_cols)]],
        horizontal_spacing=0.01,
        subplot_titles=['True', 'Predicted', '(Predicted-True)']
    )
    size_scale = 0.1

    mapbox_args = {
        'mode': 'markers',
        'lat': df_merge['lat'],
        'lon': df_merge['long'],
    }
    # add mapbox for the true values
    fig.add_trace(
        go.Scattermapbox(
            name='True',
            marker={
                'size': df_merge['true_output']*size_scale*1.2,
                'color': 'darkorchid',
                'opacity': 0.8,
            },
            **mapbox_args,
            ),
        row=1,
        col=1,
    )
    # add mapbox for the predicted values
    fig.add_trace(
        go.Scattermapbox(
            name='Predicted',
            marker={
                'size': df_merge['model_output']*size_scale,
                'color': 'darkturquoise',
            },
            **mapbox_args,
            ),
        row=1,
        col=2,
    )
    # add mapbox for the difference (positive difference is in blue)
    fig.add_trace(
        go.Scattermapbox(
            name='Positive Difference',
            marker={
                'size': df_merge['positive']*size_scale,
                'color': 'navy',
            },
            **mapbox_args,
            ),
        row=1,
        col=3,
    )
    # add mapbox for the difference (negative difference is in blue)
    fig.add_trace(
        go.Scattermapbox(
            name='Negative Difference',
            marker={
                'size': df_merge['negative']*size_scale,
                'color': 'gold'
            },
            **mapbox_args,
            ),
        row=1,
        col=3,
    )
    view_loc = {
        'center': {'lat': 38, 'lon': -94},
        'zoom': 2.1,
    }

    mapbox_kwargs = {f'mapbox{i+1}': view_loc for i in range(n_cols)}
    mapbox_style_kwargs = {f'mapbox{i+1}_style': 'open-street-map' for i in range(n_cols)}
    fig.update_layout(
        autosize=True,
        title_text="Cumulative Arrival Delay",
        **mapbox_kwargs,
        **mapbox_style_kwargs,
    )

    window_data = graph_info.iloc[24-hr_shift+hour-n_prev+1:24-hr_shift+hour+1].reset_index()
    window_data = window_data.melt(id_vars='timestamp')
    window_data.rename(columns={'value': 'delay'}, inplace=True)
    window_data['type'] = 'history'

    predict_data = graph_info.iloc[24-hr_shift+hour+n_pred-1].to_frame().T
    predict_data.reset_index(inplace=True)
    predict_data.rename(columns={'index': 'timestamp'}, inplace=True)
    predict_data = predict_data.melt(id_vars='timestamp')
    predict_data.rename(columns={'value': 'delay'}, inplace=True)
    predict_data['type'] = 'truth'
    
    facet_data = pd.concat([window_data, predict_data])

    facet_fig = px.line(
        facet_data,
        x='timestamp',
        y='delay',
        facet_col='airport',
        color='type',
        facet_col_wrap=5,
        height=800,
        markers=True,
    )
    facet_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig, model_pred_stats, display_table, facet_fig


# Callback function updates display.
@app.callback(
    [
        Output('chosen-date-pred', 'children'), 
        Output('chosen-hour-pred', 'children'),
        Output('graph-compare', 'figure'), 
        Output('valid-inputs', 'children'),
        Output('pred-info', 'children'),
        Output('data-table', 'figure'),
        Output('graph-traj', 'figure'),
    ], 
    [
        Input('pred-date', 'date'), 
        Input('pred-hour', 'value'), 
    ],
)
def update_prediction_page(date, hour):
    if date is not None:
        date_obj = dt.date.fromisoformat(date)
        date_str = date_obj.strftime('%b %d, %Y')
        fig, model_pred_stats, display_table, facet_fig = update_pred_figure(date, hour)

        table = go.Figure(
            data=[
                go.Table(
                    header=dict(values=list(display_table.columns), fill_color='paleturquoise', align='center'), 
                    cells=dict(values=[display_table.icao, display_table.true_output, display_table.model_output, display_table.difference], align='center'),
                ),
            ],
        )

        if fig is not None:
            valid_fig = '*Valid Inputs for Comparison*'
        else:
            valid_fig = '*Invalid Inputs for Comparison*'
        return [
            f'You have selected the date: {date_str}',
            f'You have selected the hour: {hour}',
            fig,
            valid_fig,
            model_pred_stats,
            table,
            facet_fig,
        ]
    else:
        return dash.no_update
