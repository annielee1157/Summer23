import numpy as np
import pandas as pd


def create_node_data_around_date(df, date, hr_shift):
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
    date_vec = [date-np.timedelta64(1,'D'), date, date+np.timedelta64(1,'D')]#np.sort(df['arrival_date_tz'].unique())
    arr_airport = np.sort(df['arrival_aerodrome_icao_name'].unique())
    node_data = []
    for date in date_vec:
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
    graph.to_csv(f'data/prediction/{date}_graph_data.csv')
    return graph, node_data