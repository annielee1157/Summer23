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
import plotly.express as px
import base64
from pathlib import Path

import apps.nowcast.query_graph as query_surface
from app import app
import config

'''
TODO: 
modify to create nowcast display of network delays:
we want to see the current network delay graph
something similar to https://flightaware.com/miserymap/
'''


# Set NEC route image
image = 'images/NEC_routes.png'
encoded_image = base64.b64encode(open(image, 'rb').read()).decode('ascii')

# Set the layout
def render() -> html.Div:
    return html.Div([
        dbc.Row(
            [
            dbc.Col([

                dbc.Row([
                    dbc.Col(
                        [
                        html.H4('Surface Summary', className='text-left mr-5'),
                        dcc.Loading(id = 'loading-info-surface'),
                        ], style = {'display': 'flex'})
                ]
                ),

                dbc.Row(
                [
                    dbc.Col(html.H6('Start time (UTC)', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'15px'}), width = '3'),

                    dbc.Col(
                        [
                            dbc.Input(
                                id="run-time-start",
                                type="datetime-local",
                                style={'maxWidth': '300px'},
                                step="1"
                            ),
                            dbc.Button(
                                '✕',
                                id='clear-run-time-start',
                                n_clicks=0,
                                color="primary"
                            )
                        ], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-top':'10px'}, width = 'auto'),
                ]
                ),

                dbc.Row(
                [
                    dbc.Col(html.H6('End time (UTC)', style = {'textAlign': 'left', 'margin-left':'15px', 'margin-top':'25px'}), width = '3'),

                    dbc.Col(
                        [
                            dbc.Input(
                                id="run-time-end",
                                type="datetime-local",
                                style={'maxWidth': '300px'},
                                step="1"
                            ),
                            dbc.Button(
                                '✕',
                                id='clear-run-time-end',
                                n_clicks=0,
                                color="primary"
                            )
                        ], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-top':'20px'}, width = 'auto'),
                ]
                ),

                dbc.Row(
                    html.H6('OR',
                    style = {'textAlign': 'left', 'margin-left':'250px', 'margin-top':'40px'})
                ),

                dbc.Row(
                    html.H6('Input how far to look back at the data from current time in minutes and press Enter.',
                    style = {'textAlign': 'left', 'margin-left':'25px', 'margin-top':'40px'})
                ),

                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Input(
                                id="lookback-surface",
                                type = "number",
                                debounce = True,
                                min = 60,
                                placeholder = "Lookback Minutes"
                            ),
                            dbc.Button(
                                '✕',
                                id='clear-lookback',
                                n_clicks=0,
                                color="primary"
                            )
                        ], style = {'display': 'flex', 'height': '40px', 'alignSelf': 'center', 'margin-left':'10px'}, width = 'auto')
                ),
                
                dbc.Row(
                    [
                        dbc.Col(html.Div(
                            dcc.Dropdown(id = 'airport-dropdown-surface',
                                options = [
                                    {'label': 'JFK', 'value': 'JFK'},
                                    {'label': 'LGA', 'value': 'LGA'},
                                    {'label': 'EWR', 'value': 'EWR'}
                                ],
                                value = 'JFK', multi = True, clearable = False, persistence = True),
                                style = {'margin-top': '60px', 'margin-left': '10px'}), width = 5),

                        dbc.Col(html.Div(
                            dcc.RadioItems(id = 'bin-options-surface',
                                options = [
                                    {'label': '15 min', 'value': 15},
                                    {'label': '30 min', 'value': 30},
                                    {'label': '60 min', 'value': 60}
                                ],
                                value = 15, labelStyle = {'display': 'inline-block'}),
                                style = {'margin-top': '60px'}), width = "auto"),
                    ]
                    ),
                
                dbc.Row(
                    html.Div(id = 'text-surface'),
                    style = {'margin-left': '10px', 'margin-top': '20px'}
                ),
            ]
            ),

            dbc.Col(html.Img(src = 'data:image/png;base64,{}'.format(encoded_image), style = {'width':'105%', 'margin-top': '20px', 'display':'inline-block'})),
            ]
        ),

        dbc.Row(
            dbc.Col(dcc.Graph(id = 'graph-surface',
                figure = {'data': [],
                'layout': go.Layout(
                        xaxis = {'showticklabels':False, 'ticks':'', 'showgrid':False, 'zeroline': False},
                        yaxis= {'showticklabels':False, 'ticks':'', 'showgrid':False, 'zeroline': False}, title = {'text': ''})})
            ), style = {'margin-top': '20px'}
        ),
        
        dcc.Interval(
            id = 'interval-component-surface',
            interval = config.params['refresh_interval_minutes']*60*1000, # in milliseconds
            n_intervals = 0)

    ])

@app.callback(
    Output('run-time-end', 'min'),
    Input('run-time-start', 'value')
)
def get_min_time(start_run):
    if start_run:
        # Work around since input does not recognize 00 for the seconds
        if (len(start_run) == 16):
            start_run = start_run + ':00'
        start_run = dt.datetime.strptime(start_run, '%Y-%m-%dT%H:%M:%S')
        min_runtime = start_run + pd.Timedelta(hours = 1)
        return min_runtime

@app.callback(
    Output('run-time-start', 'value'),
    Input('clear-run-time-start', 'n_clicks')
)
def clear_run_time_start_value(n):
    return None

@app.callback(
    Output('run-time-end', 'value'),
    Input('clear-run-time-end', 'n_clicks')
)
def clear_run_time_value(n):
    return None

@app.callback(
    Output('lookback-surface', 'value'),
    Input('clear-lookback', 'n_clicks')
)
def clear_lookback_value(n):
    return None

# Query function
def run_query(min_lookback, start_run, end_run, n):
    if ((min_lookback) or (start_run and end_run)):
        tic = time.time()

        # Gather data
        df_total, start_time, end_time = query_surface.get_surface_count_actual_taxi(min_lookback, start_run, end_run)

        # Find query time
        toc = time.time()
        duration = np.round((toc - tic),2)
        print('Query Time: ', duration, ' seconds')

        return {
            'df_total':df_total,
            'start_time':start_time, 
            'end_time':end_time
        }

    else:
        return dash.no_update

# Callback function updates display.
@app.callback([Output('graph-surface', 'figure'), Output('text-surface', 'children'), Output('loading-info-surface', 'children')], 
                [Input('lookback-surface', 'value'), Input('run-time-start', 'value'), Input('run-time-end', 'value'),
                Input('interval-component-surface', 'n_intervals'), Input('airport-dropdown-surface', 'value'), Input('bin-options-surface', 'value')])
def update_plot_and_text(min_lookback, start_run, end_run, n, airports, bin_value):

    # Convert airports to list type
    if type(airports) != list:
        temp = []
        temp.append(airports)
        airports = temp

    # Verify one or more airports have been selected
    if len(airports) == 0:
        print('No airport chosen. Graph has not been updated.')
        return dash.no_update

    if ((min_lookback) or (start_run and end_run)):
        # Gather data

        start_run = str(pd.Timestamp(' '.join(start_run.split('T')))) if start_run else None
        end_run = str(pd.Timestamp(' '.join(end_run.split('T')))) if end_run else None

        query_results = run_query(min_lookback, start_run, end_run, n)

        if not query_results:
            print('No data gathered from db.')
            return dash.no_update

        # Convert dictionaries back to list of dataframes and variables
        df_total = query_results['df_total']
        start_time = query_results['start_time']
        end_time = query_results['end_time']

        if (df_total is None) and (str((dt.datetime.strptime(end_run, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours = 1))) < start_time):
            return [
                dash.no_update, html.P('Invalid start and end time combination. The end time must be at least 1 hour later than the start time.',
                style = {'textAlign':'left', 'margin-left':'10px', 'font-size': 18, 'font-family' :'Times', 'color': 'red'}),
                True
            ]

        if (df_total is not None) and len(df_total) == 0:
            print('No data was gathered from db.')
            return [
                dash.no_update,
                html.P('No data to show.', style = {'textAlign':'left', 'margin-left':'10px', 'font-size': 18, 'font-family' :'Times', 'color': 'red'}),
                True
            ]

        if start_run and end_run:
            min_lookback_val = (dt.datetime.strptime(end_run, '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(start_run, '%Y-%m-%d %H:%M:%S')).total_seconds()/60.0
        else:
            min_lookback_val = min_lookback

        # Create date range based on bin value
        time_sequence = pd.date_range(start_time, end_time, freq = str(bin_value) + 'min')

        # Generate subplots
        num_rows = len(airports) *2
        subplot_titles_list = ['Surface Delay By Departure Fix', 'Median Surface Delay By Departure Fix (All Airports)']
        for i in range(len(airports)):
            subplot_titles_list.append(airports[i] + " Average Surface Counts")
            subplot_titles_list.append(airports[i] + " Average Actual Taxi Out Time")
        v_spacing = 0.25 if ((min_lookback_val/bin_value) >= 35) else 0.15

        fig = make_subplots(rows = num_rows + 2, cols = 1, subplot_titles = (subplot_titles_list), 
                    vertical_spacing = v_spacing/((num_rows+2)/2), shared_xaxes = True)

        ### Departure Fix
        # Add departure fix plots if there are flights in PUSHBACK state
        if len(df_total[df_total['surface_flight_state'] == 'PUSHBACK'].reset_index(drop = True)) != 0:
            dep_fix_plots(df_total, time_sequence, fig, min_lookback_val, bin_value)
        else:
            print('There are no flights found with surface state PUSHBACK.')
            fig.add_annotation(x = 0.5, y = 0, text = 'No flights found with surface state PUSHBACK.', showarrow = False, xref = 'x1', yref = 'y1')

        # Create lists with max values for the range on the plots
        upper_surface_count_list = []
        upper_taxi_list = []
        
        ### Surface Counts and Taxi Out Durations
        # Loop through the dataframe for each airport to calculate bin values and generate plots
        for i in range(len(airports)):
            # Filter dataframe by specific airport
            df = df_total[(df_total['departure_aerodrome_iata_name'] == airports[i]) | (df_total['arrival_aerodrome_iata_name'] == airports[i])].reset_index(drop = True)

            if len(df[df['departure_aerodrome_iata_name'] == airports[i]]) > 0:
                # Drop zero and negative taxi time departures
                df = df[~((df['departure_aerodrome_iata_name'] == airports[i]) & ((df['actual_ramp_taxi_minutes']<= 0) | (df['actual_ama_taxi_minutes']<= 0)
                    | (df['actual_full_taxi_minutes']<= 0)))].reset_index(drop = True)

                # Apply function that calculates how much time a flight spends in the ramp and AMA in each bin
                dep_ramp_counts = active_time(df['departure_stand_actual_time'], df['departure_movement_area_actual_time'], time_sequence[:-1], bin_value, 'dep_ramp')
                dep_ama_counts = active_time(df['departure_movement_area_actual_time'], df['departure_runway_actual_time'], time_sequence[:-1], bin_value, 'dep_ama')
                arr_ramp_counts = active_time(df['arrival_movement_area_actual_time'], df['arrival_stand_actual_time'], time_sequence[:-1], bin_value, 'arr_ramp')
                arr_ama_counts = active_time(df['arrival_runway_actual_time'], df['arrival_movement_area_actual_time'], time_sequence[:-1], bin_value, 'arr_ama')

                df = pd.concat([df, dep_ramp_counts, dep_ama_counts, arr_ramp_counts, arr_ama_counts], axis =1)

                # Apply function that indicates if a flight exits the ramp or takes off during a bin
                df = df.apply(dep_location, time_sequence = time_sequence, bin_value = bin_value, axis = 1)

                # Write active time and location data to csv file
                Path('data', 'surface_summary').mkdir(parents = True, exist_ok = True)
                df.to_csv('data/surface_summary/' + airports[i] + '_surface_active_times.csv')
    
                # Create a dataframe where rows indicate each bin
                df_bin = pd.DataFrame(data = {'start_time_bins': time_sequence[:-1]})

                for bin in range(len(time_sequence)-1):
                    df_bin.loc[bin, 'average_ramp_dep_flight_count'] = df['bin_'+ str(bin) + '_dep_ramp_time'].sum()/bin_value
                    df_bin.loc[bin, 'average_ama_dep_flight_count'] = df['bin_'+ str(bin) + '_dep_ama_time'].sum()/bin_value
                    df_bin.loc[bin, 'average_ramp_arr_flight_count'] = df['bin_'+ str(bin) + '_arr_ramp_time'].sum()/bin_value
                    df_bin.loc[bin, 'average_ama_arr_flight_count'] = df['bin_'+ str(bin) + '_arr_ama_time'].sum()/bin_value

                    # Find average ramp taxi for flights that have exited the ramp in each bin
                    df_temp = df[df['exit_ramp_bin'] == bin].reset_index(drop = True)
                    df_bin.loc[bin, 'average_actual_ramp_taxi'] = df_temp['actual_ramp_taxi_minutes'].mean()

                    # Find average AMA and full taxi for flights that have taken off in each bin
                    df_temp = df[df['takeoff_bin'] == bin].reset_index(drop = True)
                    df_bin.loc[bin, 'average_actual_ama_taxi'] = df_temp['actual_ama_taxi_minutes'].mean()
                    df_bin.loc[bin, 'average_actual_full_taxi'] = df_temp['actual_full_taxi_minutes'].mean()
            
                # Write bin groups to csv file
                Path('data', 'surface_summary').mkdir(parents = True, exist_ok = True)
                df_bin.to_csv('data/surface_summary/' + airports[i] + '_surface_binned_data.csv')

                legend = True if i == 0 else False

                # Add stacked bar chart for surface count
                fig.add_trace(go.Bar(x = df_bin['start_time_bins'], y = df_bin['average_ramp_dep_flight_count'].round(2), marker_color = 'red', marker_line_color = 'black',
                    marker_line_width = 2, name = 'Ramp Deps', showlegend = legend, legendgroup = '1'), row = 2*i+3, col = 1)
                fig.add_trace(go.Bar(x = df_bin['start_time_bins'], y = df_bin['average_ama_dep_flight_count'].round(2), marker_color = 'cornflowerblue', marker_line_color = 'black',
                    marker_line_width = 2, name = 'AMA Deps', showlegend = legend, legendgroup = '1'), row = 2*i+3, col = 1)
                fig.add_trace(go.Bar(x = df_bin['start_time_bins'], y = df_bin['average_ramp_arr_flight_count'].round(2), marker_color = 'mediumseagreen', marker_line_color = 'black',
                    marker_line_width = 2, name = 'Ramp Arrs', showlegend = legend, legendgroup = '1'), row = 2*i+3, col = 1)
                fig.add_trace(go.Bar(x = df_bin['start_time_bins'], y = df_bin['average_ama_arr_flight_count'].round(2), marker_color = 'purple', marker_line_color = 'black',
                    marker_line_width = 2, name = 'AMA Arrs', showlegend = legend, legendgroup = '1'), row = 2*i+3, col = 1)

                # Add average actual taxi out times
                fig.add_trace(go.Scatter(mode = 'lines+markers', line = dict(color = "red", width = 1.5), marker = dict(color = "red", size = 10, symbol = 'star-triangle-up'),
                        x = df_bin['start_time_bins'], y = df_bin['average_actual_ramp_taxi'].round(2), name = 'Ramp', showlegend = legend, legendgroup = '2'), row = 2*i+4, col = 1)
                fig.add_trace(go.Scatter(mode = 'lines+markers', line = dict(color = "cornflowerblue", width = 1.5), marker = dict(color = "cornflowerblue", size = 10, symbol = 'star-triangle-down'),
                        x = df_bin['start_time_bins'], y = df_bin['average_actual_ama_taxi'].round(2), name = 'AMA', showlegend = legend, legendgroup = '2'), row = 2*i+4, col = 1)
                fig.add_trace(go.Scatter(mode = 'lines+markers', line = dict(color = "black", width = 1.5), marker = dict(color = "black", size = 10, symbol = 'star-triangle-down'),
                        x = df_bin['start_time_bins'], y = df_bin['average_actual_full_taxi'].round(2), name = 'Full', showlegend = legend, legendgroup = '2'), row = 2*i+4, col = 1)

                # Find max surface count as an integer for plot range
                max_surface_count = math.ceil((df_bin['average_ramp_dep_flight_count'] + df_bin['average_ama_dep_flight_count'] +
                            df_bin['average_ramp_arr_flight_count'] + df_bin['average_ama_arr_flight_count']).max())
                # Add to list of upper surface counts
                upper_surface_count_list.append(max_surface_count)

                # Find max taxi time as an integer for plot range
                max_value = np.nanmax([df_bin['average_actual_ramp_taxi'].max(), df_bin['average_actual_ama_taxi'].max(), df_bin['average_actual_full_taxi'].max()])
                max_taxi = 0 if math.isnan(max_value) else math.ceil(max_value)

                # Add to list of upper taxi values
                upper_taxi_list.append(max_taxi)

            else:
                fig.add_annotation(x = 0, y = 1, text = 'No data found for this airport.', showarrow = False, xref = 'x'+str(2*i + 3), yref = 'y'+str(2*i + 3))
                fig.add_annotation(x = 0, y = 1, text = 'No data found for this airport.', showarrow = False, xref = 'x'+str(2*i + 4), yref = 'y'+str(2*i + 4))

        # Format the figure
        # Create array of bins used for the tick marks - done this way so that bins with no result will still show that time as a tick mark on the plots.
        start_time_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_time_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        bins = [start_time_dt]
        num_bins = math.ceil(((end_time_dt - start_time_dt).total_seconds())/(bin_value*60))
        for i in range(num_bins -1):
            start_time_dt = start_time_dt + dt.timedelta(minutes = bin_value)
            bins.append(start_time_dt)

        # Use the maximum value from all airports for the range in the plots for easier comparison - round to next multiple of 5
        max_upper_count = max(upper_surface_count_list) + (5 - max(upper_surface_count_list)) % 5
        max_upper_taxi = max(upper_taxi_list) + (5 - max(upper_taxi_list)) % 5
        
        for i in range(len(airports)):
            fig.update_yaxes(title = 'Count of Active<br>Flights', title_font_size = 16, range = [0, max_upper_count], row = 2*i +3, col =1)
            fig.update_yaxes(title = 'Time<br>(in Minutes)', title_font_size = 16, range = [0, max_upper_taxi], row = 2*i +4, col =1)

        if len(airports) ==1:
            legend_location = 0.46
        elif len(airports) ==2:
            legend_location = 0.65
        else:
            legend_location = 0.74

        fig.update_xaxes(categoryorder='category ascending', tickfont_size = 12, showticklabels = True, row = 1, col = 1)
        if ((min_lookback_val/bin_value) >= 60):
            fig.update_xaxes(tickfont_size = 12)

        for i in range(2*len(airports)+2):
            fig.update_xaxes(tickvals = bins, ticktext = ["%02d:%02d:%02d" % (x.hour, x.minute, x.second) for x in bins], showticklabels = True, row = i+2, col = 1)
        fig.update_layout(margin = {'t':20}, height = 500 *(len(airports)) + 500, width = 1500, font = dict(size = 15), boxgap = 0.1, bargap = 0.1, barmode = 'stack', hovermode = 'x unified', legend_tracegroupgap= 150, 
                    title = {'text': 'Start of Bin (UTC)', 'y': 0, 'x': 0.46, 'xanchor': 'center', 'yanchor':'bottom', 'font': dict(size= 15, color = 'black')}, 
                    yaxis_title = 'Surface Delay<br>(Minutes)', yaxis_title_font_size = 16, yaxis2_title = 'Surface Delay<br>(Minutes)', yaxis2_title_font_size = 16,
                    legend=dict(yanchor="top", y=legend_location, xanchor="right", x=1.15))

        return [
        fig,
        [html.P('Start Time: ' + start_time, style = {'textAlign': 'left', 'font-size': 18}), 
        html.P('End Time: ' + end_time, style = {'textAlign': 'left', 'font-size': 18})],
        True]

    else:
        return dash.no_update

# Function generates two first subplots for departure fix
def dep_fix_plots(df_total, time_sequence, fig, min_lookback_val, bin_value):
        # Filter by pushback surface state and not null delay
        df_depfix = df_total[(df_total['surface_flight_state'] == 'PUSHBACK') & (df_total['delay'].notnull())].reset_index(drop = True)

        # Find what bin the flight takes off
        df_depfix = df_depfix.apply(dep_location, time_sequence = time_sequence, bin_value = bin_value, axis =1)

        # Filter to only columns interested in
        df_depfix = df_depfix[['gufi', 'timestamp', 'surface_flight_state', 'departure_runway_undelayed_time', 'departure_runway_actual_time', 'departure_fix_actual', 'departure_fix_decision_tree',
                                'departure_aerodrome_iata_name', 'delay', 'takeoff_bin']]
    
        # Filter to only takeoff bins >= 0
        df_depfix = df_depfix[df_depfix['takeoff_bin'] >= 0].reset_index(drop = True)

        if (len(df_depfix) == 0):
            print('There are no flights found with surface state PUSHBACK in any bins.')
            fig.add_annotation(x = 0.5, y = 0, text = 'No flights found with surface state PUSHBACK in any bins.', showarrow = False, xref = 'x1', yref = 'y1')
    
        # Write departure fix information to csv file
        Path('data', 'surface_summary').mkdir(parents = True, exist_ok = True)
        df_depfix.to_csv('data/surface_summary/departure_fix.csv')

        # List all fixes in order of East, North, South, West gates
        all_fixes = ['BAYYS', 'BDR', 'BETTE', 'GREKI', 'HAPIE', 'MERIT', # EAST
                    'BREZY', 'COATE', 'DEEZZ', 'GAYEL', 'HAAYS', 'NEION', # NORTH
                    'DIXIE', 'RBV', 'SHIPP', 'WAVEY', 'WHITE', # SOUTH
                    'BIGGY', 'ELIOT', 'LANNA', 'NEWEL', 'PARKE', 'ZIMMZ'] # WEST

        # Create dictionary
        fix_to_gate_map = {'BAYYS':'E', 'BDR':'E', 'BETTE':'E', 'GREKI':'E', 'HAPIE':'E', 'MERIT':'E',
                    'BREZY':'N', 'COATE':'N', 'DEEZZ':'N', 'GAYEL':'N', 'HAAYS':'N', 'NEION':'N',
                    'DIXIE':'S', 'RBV':'S', 'SHIPP':'S', 'WAVEY':'S', 'WHITE':'S',
                    'BIGGY':'W', 'ELIOT':'W', 'LANNA':'W', 'NEWEL':'W', 'PARKE':'W', 'ZIMMZ':'W'}

        # Set colors for boxplots
        colors_fill = ['blueviolet', 'lightcoral', 'mediumturquoise']
        colors_line = ['indigo', 'firebrick', 'darkcyan']

        for j in range(len(all_fixes)):
            df_init = df_depfix[df_depfix['departure_fix_decision_tree'] == all_fixes[j]].reset_index(drop = True)

            if len(df_init) != 0:
                for i, dep_airport in enumerate(['JFK', 'LGA', 'EWR']):
                    df_temp = df_init[df_init['departure_aerodrome_iata_name'] == dep_airport].reset_index(drop = True)

                    if len(df_temp) !=0:
                        airport_repeated = [dep_airport] * len(df_temp)

                        fig.add_trace(go.Box(x = [df_temp['departure_fix_decision_tree'], airport_repeated], y = df_temp['delay'], line = dict(color = colors_line[i]), fillcolor = colors_fill[i], name = dep_airport,
                            showlegend = False, boxpoints = False,  hoverinfo = 'y', opacity = 0.7), row = 1, col = 1)

                        fig.add_annotation(x = [all_fixes[j], dep_airport], y = df_temp['delay'].max(), text = str(len(df_temp)), yshift = 10, showarrow = False)
                        fig.add_annotation(x = [all_fixes[j], dep_airport], y = df_temp['delay'].max(), text = str(fix_to_gate_map[all_fixes[j]]), yshift = 25, showarrow = False)

        # Create a dataframe where rows indicate each bin
        df_fix_bin = pd.DataFrame(data = {'start_time_bins': time_sequence[:-1]})

        # Find the median surface delay for each departure fix at each time bin
        for bin in range(len(time_sequence)-1):
            for fix in all_fixes:
                df_temp = df_depfix[(df_depfix['takeoff_bin'] == bin) & (df_depfix['departure_fix_decision_tree'] == fix)].reset_index(drop = True)
                df_fix_bin.loc[bin, fix + '_median_delay'] = df_temp['delay'].median()
                df_fix_bin.loc[bin, fix + '_count'] = str(len(df_temp))

        # Save dataframe as csv file
        Path('data', 'surface_summary').mkdir(parents = True, exist_ok = True)
        df_fix_bin.to_csv('data/surface_summary/departure_fix_bins.csv')

        # Set marker size for scatter plot depending on the number of bins
        if ((min_lookback_val/bin_value ) <= 27):
            m_size = 30
        elif (((min_lookback_val/bin_value) <= 56) & ((min_lookback_val/bin_value) > 27)):
            m_size = 15
        elif (((min_lookback_val/bin_value) <=106 ) & ((min_lookback_val/bin_value) > 56)):
            m_size = 8
        else:
            m_size = 4

        # Add scatter plot for median surface delay for each fix at each time bin
        colors = px.colors.qualitative.Dark24
        for i, fix in enumerate(all_fixes[::-1]):
            fig.add_trace(go.Scatter(x = df_fix_bin['start_time_bins'], y = df_fix_bin[fix + '_median_delay'].round(2), mode = 'markers', marker_color = colors[i], marker_line_color = colors[i], 
                marker_line_width = 4, marker_symbol = 'line-ew-open', marker_size = m_size, name = fix, hovertemplate = '%{y}, n = ' + df_fix_bin[fix + '_count'], showlegend = False), row = 2, col = 1)

# Function calculates how much time a flight spends in the ramp or AMA in each bin
def active_time(start_location_times, end_location_times, time_seq, bin_value, calc_type):

    # Set column values to datetype timestamp or np.nan
    start_location_times = [np.nan if pd.isnull(x) else x.timestamp() for x in start_location_times]
    end_location_times = [np.nan if pd.isnull(x) else x.timestamp() for x in end_location_times]

    # Set duration value depending on bin size
    duration = pd.Timedelta(str(bin_value) + "minutes")

    # Initialize a list to append how long a flight is within the ramp or AMA
    area_time_min = []
    for t in time_seq:
        # Set lower and upper bound lists that are the same length as the time columns for comparison
        lower_bound = np.repeat(t.timestamp(), len(start_location_times))
        upper_bound = np.repeat((t+duration).timestamp(), len(start_location_times))

        # Calculate how long a flight is within particular area
        pos = np.minimum(end_location_times, upper_bound)
        neg = np.maximum(start_location_times, lower_bound)
        contrib = (pos - neg)/60.0

        # Append values to list
        area_time_min.append(contrib)

    # Convert list of list to a dataframe
    results = pd.DataFrame(area_time_min).transpose()
    results.columns = ['bin_' + str(i) + '_'+ calc_type + '_time' for i in range(len(time_seq))]

    # Replace negative values with 0
    results[results < 0] = 0

    return results

# Function indicates if a flight exits the ramp or takes off during a bin
def dep_location(df, time_sequence, bin_value):

    # Find what bin the flight exited the ramp area
    df['exit_ramp_bin'] = (df['departure_movement_area_actual_time'] - time_sequence[0]).total_seconds() / (bin_value*60)

    if math.isnan(df['exit_ramp_bin']):
        df['exit_ramp_bin'] = np.nan 
    else:
        df['exit_ramp_bin'] = math.floor(df['exit_ramp_bin'])

    # Find what bin the flight exited the AMA (takeoff)    
    df['takeoff_bin'] = (df['departure_runway_actual_time'] - time_sequence[0]).total_seconds() / (bin_value*60)

    if math.isnan(df['takeoff_bin']):
        df['takeoff_bin'] = np.nan 
    else:
        df['takeoff_bin'] = math.floor(df['takeoff_bin'])

    return df
