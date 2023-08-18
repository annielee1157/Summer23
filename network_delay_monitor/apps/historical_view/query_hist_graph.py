import numpy as np
import pandas.io.sql as psql
import pandas as pd
from pathlib import Path
import psycopg2

DT_COLS = [
    'departure_stand_scheduled_time',
    'departure_stand_actual_time',
    'departure_runway_actual_time',
    'arrival_stand_scheduled_time',
    'arrival_stand_actual_time',
    'arrival_runway_actual_time',
]

def convert_tz(df, tz='US/Eastern', remove_tz=False):
    '''
    function to convert a dateTime (assumed to be in UTC) to a named time zone
    
    Input(s):
    ---------
    df          a pandas Series of dateTimes
    tz          (optional) the timezone to convert to (default: US/Eastern)
    remove_tz   (optional) remove the timezone information after converting to Eastern Time (default: False)

    Output(s):
    ----------
    a converted pandas Series of dateTimes
    '''
    Path('data', 'historical_view').mkdir(parents = True, exist_ok = True)

    # if the dateTimes have no association with a time zone, assume they are in UTC
    if df.dt.tz is None:
        df = df.dt.tz_localize(tz='UTC')
    # convert to eastern time; tz_convert() takes care of daylight savings time
    df = df.dt.tz_convert(tz)
    # remove the time zone info
    if remove_tz:
        df = df.dt.tz_localize(None)
    return df

def raw_query(airports, start_time, end_time, port=8520, carrier='DAL', tab='historical_view'):
    '''
    function to query raw data from database

    Input(s):
    ---------
    airports            a tuple of airports to query from
    start_time          datetime representation of the start time to query
    end_time            datetime representation of the end time to query
    port                integer port number to use to query the data
    carrier             string of the carrier to query

    Output(s):
    ----------
    df                  DataFrame created from querying data
    '''
    with psycopg2.connect(
        f"dbname='fusernas' user='fuser' password='fuser' host='localhost'  "
    ) as conn:
        conn.set_session(autocommit=True)
        columns_to_query = ['gufi',
                            'departure_aerodrome_icao_name',
                            'arrival_aerodrome_icao_name',
                            'arrival_stand_actual_time',
                            'arrival_stand_scheduled_time',
                            ]
        conditions = [f"departure_runway_actual_time between '{start_time}' AND '{end_time}'",
                      "arrival_stand_actual_time IS NOT NULL",
                      "arrival_stand_scheduled_time IS NOT NULL",
                      f"departure_aerodrome_icao_name IN {airports}",
                      f"arrival_aerodrome_icao_name IN {airports}",
                      ]
        if carrier is not None:
            conditions.append(f"carrier = '{carrier}'")
        
        condition_join = '\nAND '
        q = f'''SELECT {', '.join(columns_to_query)} 
        FROM matm_flight_summary 
        WHERE {condition_join.join(conditions)}
        '''
        print("\nQUERY:\n", q)
        df = psql.read_sql(q, conn)
        if carrier is not None:
            df.to_csv(f'data/{tab}/{carrier}_{start_time}_{end_time}_raw_queried_data.csv')
        else:
            df.to_csv(f'data/{tab}/{start_time}_{end_time}_raw_queried_data.csv')
    return df

def compute_delay_from_raw_data(df, carrier=None, tz='US/Eastern', tab='historical_view'):
    """
    """
    # convert datetime listings to desired timezone for interpretability
    for item in df.columns:
        if item in DT_COLS:
            df[f'{item}_tz'] = convert_tz(df[item], tz=tz, remove_tz=True)

    # compute the arrival delay time defined as the time that the flight actually arrived at the gate minus the scheduled arrival time
    df['arrival_delay_amount'] = (df['arrival_stand_actual_time'] - df['arrival_stand_scheduled_time']).dt.total_seconds() / 60.0

    # set the early and on-time arrivals to 0 (i.e., no delay) because we only care about the delay, FAA defines anything >15 min as a delay
    delay_mask = df['arrival_delay_amount'] <= 15
    df.loc[delay_mask,'arrival_delay_amount'] = 0

    # extract dates and times
    df['arrival_date_tz'] = df['arrival_stand_actual_time_tz'].dt.normalize()
    df['arrival_hour_tz'] = df['arrival_stand_actual_time_tz'].dt.hour

    # create a reduced dataframe
    red_columns = [
        'arrival_date_tz',
        'arrival_hour_tz',
        'arrival_aerodrome_icao_name',
        'arrival_delay_amount'
    ]
    df_red = df[red_columns]

    # grouped flights by airport, date, and hour to sum the delay
    dfg_sum = df_red.groupby(['arrival_date_tz', 'arrival_hour_tz', 'arrival_aerodrome_icao_name']).sum()
    dfg_sum = dfg_sum.sort_values(by=['arrival_date_tz', 'arrival_hour_tz', 'arrival_aerodrome_icao_name'])
    dfg_sum = dfg_sum.reset_index(drop=False)
    # resulting dataframe is the total arrival delay at an airport for a given date for each hour of the date
    if carrier is not None:
        dfg_sum.to_csv(f'data/{tab}/{carrier}_processed_data.csv')
    else:
        dfg_sum.to_csv(f'data/{tab}/processed_data.csv')
    return dfg_sum

def create_node_data_for_date(df, date, hr_shift):
    """
    function to compute the node feature matrix for a specfied date

    Inputs:
    -------
    df          the Pandas DataFrame containing the flight information after 
                computing the cumulative arrival delay
    date        the date object representing the selected date
    hr_shift    the number of hours after midnight the accumulation of delay 
                was computed

    Outputs:
    --------
    graph       the node feature matrix consisting of cumulative arrival delay 
                per day per airport
    node_data   raw dataframe of the node features
    """
    # get the hour index for cumulative delay computation with the hour shift
    df['arrival_hour_shifted'] = df['arrival_hour_tz'] - hr_shift 
    mask = df['arrival_hour_shifted'] < 0
    df.loc[mask, 'arrival_hour_shifted'] = 24 + df.loc[mask,'arrival_hour_shifted']

    date = np.datetime64(date)
    arr_airport = np.sort(df['arrival_aerodrome_icao_name'].unique())
    node_data = []
    for a in arr_airport:
        total_delay = np.zeros((24,))
        mask = (df['arrival_date_tz'] == date) & (df['arrival_aerodrome_icao_name']==a)
        idx = df.loc[mask, 'arrival_hour_shifted']
        total_delay[idx] = np.array(df.loc[mask, 'arrival_delay_amount'])
        d = {}
        d['timestamp'] = [date + np.timedelta64(t, 'h') for t in np.arange(hr_shift, hr_shift+24)]
        d['airport'] = [a for _ in range(24)]
        d['delay'] = np.add.accumulate(total_delay)
        node_data.append(pd.DataFrame(d))
    node_data = pd.concat(node_data)
    graph = node_data.pivot_table(index=['timestamp'], columns=['airport'], values='delay')
    graph.to_csv(f'data/historical_view/{date}_graph_data.csv')
    return graph, node_data
