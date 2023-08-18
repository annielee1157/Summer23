import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import datetime as dt
import pandas as pd
import plotly.io as pio
pio.templates.default = "plotly_white"
import plotly.express as px
import os

import apps.historical_view.query_hist_graph as query_utils
from app import app
import config

NAS_8 = ('KEWR','KJFK','KLGA','KATL','KBOS','KIAD','KBWI','KDCA','KDFW','KSFO','KLAX','KSEA')

NAS_30 = ('KEWR', 'KJFK', 'KLGA', 'KATL', 'KBOS', 'KIAD', 'KBWI', 'KDCA', 'KSFO', 'KLAX', 'KDTW', 'KDFW', 'KSEA', 'KSLC', 'KCLT', 'KDEN', 'KORD', 'KLAS', 'KMCO', 'KMIA', 'KPHX', 'KIAH', 'KFLL', 'KMSP', 'KPHL', 'KSAN', 'KTPA', 'KAUS', 'KBNA', 'KMDW')

LOCATIONS = pd.read_csv('data/airport_locations.csv', index_col=0)

DATA_START_DATE = config.PARAMS['start_time']
DATA_END_DATE = config.PARAMS['end_time']

PAGE_TITLE = 'Daily Cumulative Arrival Delay'

PAGE_DESCRIPTION = '''
This page displays the cumulative arrival delay of a given date. The daily cumulative arrival delay is computed from historical data of selected airports from the national airspace system (NAS). On this page, the user may select the previous date and the carrier to visualize the network delay. Additionally, the user may select an **hour shift**, which designates the hour after midnight to begin computing the cumulative arrival delay. Note that this page converts all flight data to US Eastern Time Zone. Finally, the user can select (via slider) to view the particular hour after the beginning of the cumulative delay start.
'''

# Set the layout
def render() -> html.Div:
    return html.Div([
        dbc.Row(
                [
                dbc.Col([

                        dbc.Row(
                                dbc.Col(
                                    [
                                        dcc.Markdown(f'## {PAGE_TITLE}'),
                                        dcc.Loading(
                                            id='loading-historical-data-by-date'
                                        )
                                    ],
                                    style = {'display':'flex'}
                                )
                        ),

                        dbc.Row(
                                dbc.Col(dcc.Markdown(f'*{PAGE_DESCRIPTION}*'))
                        ),

                        dbc.Row(
                                [
                                dbc.Col(html.H6('Date', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = '3'),

                                dbc.Col(
                                    [
                                        dcc.DatePickerSingle(
                                            id="hist-date",
                                            min_date_allowed=DATA_START_DATE,
                                            max_date_allowed=DATA_END_DATE,
                                            date=dt.date(2023, 3, 1),
                                            style={'maxWidth': '300px'},
                                            clearable=False,
                                        ),
                                    ], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}, width = 'auto'),
                                
                                dbc.Col(html.H6('Carrier', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = 'auto'),

                                dbc.Col(html.Div([
                                    dcc.Dropdown(
                                        ['DAL', 'UAL', 'AAL', 'RPA', 'None'], 
                                        clearable=False,
                                        value='DAL', 
                                        id='carrier'
                                        ), 
                                    html.Div(id='dd-output-container'),
                                    ]), style = {'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}),

                                dbc.Col(html.H6('Hour Shift', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = 'auto'),

                                dbc.Col(html.Div([
                                    dcc.Dropdown(
                                        list(range(10)), 
                                        value=4, 
                                        id='hourshift',
                                        ), 
                                    html.Div(id='dd-output-container'),
                                    ]), style = {'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}),
                            ]
                        ),
                        
                        dbc.Row(
                            html.Div(id='chosen-date'),
                            style = {'margin-left':'10px', 'margin-top': '20px'}
                        ),
                        
                        dbc.Row(
                                [
                                    dbc.Col(html.Div(
                                            dcc.Slider(id='hour-slider', 
                                                       min=0,
                                                       max=23,
                                                       step=1,
                                                       value=6,
                                                       ),
                                                    style = {'margin-top':'30px', 'margin-left':'10px'}), width=10)
                                ]
                        ),
            
                        dbc.Row(
                            html.Div(id='chosen-hour'),
                            style = {'margin-left':'10px', 'margin-top': '20px'}
                        ),
                ]
                )
            ]
        ), 

        dbc.Row(
                dbc.Col(dcc.Graph(
                            id='graph-network-by-date-hour',
                            )), 
                        style = {'margin-top':'20px'}
        ),

        dbc.Row(
                dbc.Col(dcc.Graph(
                            id='graph-total-delay',
                            )), 
                        style = {'margin-top':'20px'}
        ),
    ])


def update_bubble_map(date, hour, carrier, hr_shift):
    """
    helper function to update a bubble map
    
    Inputs(s):
    ----------
    date        the chosen date
    hour        the chosen hour after the start of the cumulative arrival delay 
                computation
    carrier     the chosen carrier
    hr_shift    the hour to start the computation of the cumulative arrival
                delay
    
    Output(s):
    ----------
    fig         the figure that is created/updated when the parameters above 
                change
    graph_info  a DataFrame that contains the node feature information for 
                creating a graph (in this case, the cumulative delay per 
                airport per hour)
    """
    date_obj = dt.date.fromisoformat(date)

    # specify name of the raw data file depending on if the carrier is specified
    if carrier is not None:
        raw_data_file = f'data/historical_view/{carrier}_{DATA_START_DATE}_{DATA_END_DATE}_raw_queried_data.csv'
    else:
        raw_data_file = f'data/historical_view/{DATA_START_DATE}_{DATA_END_DATE}_raw_queried_data.csv'
    print("\nRaw Data File: ", raw_data_file)

    # query raw data if file does not already exist
    if not os.path.exists(raw_data_file):
        df_raw = query_utils.raw_query(
            NAS_30, 
            DATA_START_DATE, 
            DATA_END_DATE, 
            carrier=carrier,
        )
    # otherwise, read data from existing file
    else:
        df_raw = pd.read_csv(raw_data_file, index_col=0, parse_dates=['arrival_stand_scheduled_time', 'arrival_stand_actual_time'])
    print("\nRAW data:\n", df_raw.head())

    # compute the arrival delay times from each flight and group by hour
    delay_df = query_utils.compute_delay_from_raw_data(df_raw, carrier=carrier)
    print("\nProcessed data:\n", delay_df.head())

    # compute the cumulative arrival delay for the graph representation
    graph_info, _ = query_utils.create_node_data_for_date(
        delay_df,
        date_obj,
        hr_shift,
    )
    print("\nGraph Data:\n", graph_info.head())
    max_delay = graph_info.values.max()
    min_delay = graph_info.values.min()

    # locate the specified data from the hour given
    df_hour = graph_info.iloc[hour].to_frame(name='cumulative_delay').reset_index()
    df_hour = df_hour.rename(columns={'airport':'icao'})
    df_hour['scaled_size'] = (df_hour['cumulative_delay'] - min_delay) / (max_delay-min_delay)
    df_loc_red = LOCATIONS[LOCATIONS['icao'].isin(NAS_30)]
    df_merge = df_hour.merge(df_loc_red, on='icao')
    
    # create and update the figure for the dashboard
    fig = px.scatter_mapbox(
        df_merge,
        lat="lat",
        lon="long",
        hover_data=["icao", "cumulative_delay"],
        size="scaled_size",
        color="scaled_size",
        color_continuous_scale=px.colors.sequential.Cividis,
        zoom=3,
        center={'lat':38, 'lon':-98},
        height=600,
        opacity=0.8,
        labels={"scaled_size": "Scaled Delay"},
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig, graph_info

def update_system_total_delay(df, hour):
    """
    function to create a figure of the total cumulative delay of the network;
    plots two lines. Line 1 is the cumulative delay trajectory before the 
    requested hour, and Line 2 is the cumulative delay trajectory after the 
    requested hour.

    Input(s):
    ----------
    df          the DataFrame of delay data for a given date
    hour        the selected hour

    Output(s):
    ----------
    fig         the figure that is created/updated when the parameters above 
                change
    """
    # sum the delays across the airports (columns)
    df_sum = df.sum(axis=1).to_frame(name="total_delay").reset_index()

    # duplicate the selected hour to make and connect two lines
    df_sum = pd.concat([df_sum, df_sum.iloc[hour].to_frame().T], axis=0)
    df_sum = df_sum.sort_index()

    # add a column in the dataframe to indicate which values are before the 
    # selected hour (line 1) and which values are after the selected hour 
    # (line 2)
    df_sum["legend"] = [*["Before" for _ in range(hour+1)], *["After" for _ in range(24-hour)]]

    # create the plotly graph object
    fig = px.line(
        df_sum,
        x="timestamp",
        y="total_delay",
        title="Total Cumulative Arrival Delay of the NAS",
        color="legend",
        line_dash="legend",
        labels={"timestamp": "Time of Day", "total_delay": "Cumulative Arrival Delay (min)", "legend": "__ Selected Hour"},
        height=500,
    )

    return fig


@app.callback(
    [
        Output('chosen-date', 'children'),
        Output('chosen-hour', 'children'),
        Output('graph-network-by-date-hour', 'figure'),
        Output('graph-total-delay', 'figure'),
     ],
    [
        Input('hist-date', 'date'),
        Input('hour-slider', 'value'),
        Input('carrier', 'value'),
        Input('hourshift', 'value'),
    ]
)
def update_historical_page(date, hour, carrier, hourshift):
    """
    callback function to update the page when parameters are changed
    """
    if date is not None:
        date_obj = dt.date.fromisoformat(date)
        date_str = date_obj.strftime('%b %d, %Y')
        fig, graph_info = update_bubble_map(date, hour, carrier, hourshift)
        fig2 = update_system_total_delay(graph_info, hour)
        return [
            f'You have selected a date of {date_str}.',
            f'You have selected {hour} hour(s) after the cumulative start time.',
            fig,
            fig2,
            ]
    else:
        return dash.no_update