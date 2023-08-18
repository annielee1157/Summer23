import psycopg2
import pandas.io.sql as psql
import datetime as dt
from datetime import datetime
import pandas as pd
from pathlib import Path

# Run query given the airport, the port connection, and times
def run_airport_query(airport, port_val, start_time_buffer, timestamp_start, end_time):
	with psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost' port = %s" % (port_val)) as conn:
		conn.set_session(autocommit = True)

		q = '''WITH mfs as
		(
			SELECT *
			FROM matm_flight_summary
			WHERE departure_stand_actual_time between '%s' and '%s'
			and departure_aerodrome_iata_name = '%s'
		), 
		mfa as (
			SELECT 
				gufi,
				timestamp,
				surface_flight_state,
				departure_runway_undelayed_time,
				departure_runway_decision_tree,
				departure_fix_decision_tree
			FROM matm_flight_all
			WHERE timestamp between '%s' and '%s'
			and departure_aerodrome_iata_name = '%s'
		)
		SELECT DISTINCT ON (gufi)
			mfa.gufi, 
			mfa.timestamp,
			mfa.surface_flight_state, 
			mfs.departure_stand_actual_time,
			mfs.departure_movement_area_actual_time,
			mfs.departure_runway_actual_time,
			mfa.departure_runway_undelayed_time,
   			mfs.departure_runway_actual,
			mfa.departure_runway_decision_tree,
   			mfs.departure_fix_actual,
			mfa.departure_fix_decision_tree,
			mfs.departure_aerodrome_iata_name,
    		EXTRACT( epoch from(mfs.departure_movement_area_actual_time - mfs.departure_stand_actual_time)/60) as actual_ramp_taxi_minutes,
    		EXTRACT( epoch from(mfs.departure_runway_actual_time - mfs.departure_movement_area_actual_time)/60) as actual_ama_taxi_minutes,
    		EXTRACT( epoch from(mfs.departure_runway_actual_time - mfs.departure_stand_actual_time)/60) as actual_full_taxi_minutes, 
			EXTRACT( epoch from(mfs.departure_runway_actual_time - mfa.departure_runway_undelayed_time)/60) as delay
		FROM mfs
		INNER JOIN mfa
		ON mfs.gufi = mfa.gufi AND (mfa.timestamp - mfs.departure_stand_actual_time) < INTERVAL '4 MINUTE'
		AND (mfs.departure_stand_actual_time < mfa.timestamp)
		ORDER BY mfa.gufi, mfa.timestamp DESC
		'''%(start_time_buffer, end_time, airport, timestamp_start, end_time, airport)
		df = psql.read_sql(q,conn)

		q2 = '''
		SELECT
			gufi,
			surface_flight_state,
			arrival_stand_actual_time,
			arrival_movement_area_actual_time,
			arrival_runway_actual_time, 
    		arrival_aerodrome_iata_name
		FROM matm_flight_summary
		WHERE arrival_runway_actual_time between '%s' and '%s'
		and arrival_aerodrome_iata_name = '%s'
		ORDER BY gufi
		'''%(start_time_buffer, end_time, airport)
		df2 = psql.read_sql(q2,conn)
	
		df = df.append(df2, ignore_index = True)
		df.to_csv('data/surface_summary/' + airport + '_surface_data_query.csv')

	return df


def get_surface_count_actual_taxi(min_lookback, start_run, end_run):

	Path('data', 'surface_summary').mkdir(parents = True, exist_ok = True)

	if min_lookback:
		# Departure stand actual time range (UTC)
		start_time = str(datetime.now().utcnow().replace(microsecond = 0) - dt.timedelta(minutes = min_lookback))
		end_time = str(datetime.now().utcnow().replace(microsecond = 0))

	if start_run and end_run:
		if str((dt.datetime.strptime(end_run, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours = 1))) < start_run:
			print('Invalid start and end time combination.')
			return None, start_run, end_run
		# Departure stand actual time range (UTC)
		start_time = str(dt.datetime.strptime(start_run, '%Y-%m-%d %H:%M:%S'))
		end_time = str(dt.datetime.strptime(end_run, '%Y-%m-%d %H:%M:%S'))

	# Look back an extra hour for the query so that the surface counts can be determined for the first few bins
	start_time_buffer = str(datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') - dt.timedelta(minutes = 60))

	# Timestamp range
	timestamp_start = str(datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') - dt.timedelta(minutes = 65))

	# Run queries for each airport
	df_lga = run_airport_query('LGA', 5447, start_time_buffer, timestamp_start, end_time)
	df_ewr = run_airport_query('EWR', 5448, start_time_buffer, timestamp_start, end_time)
	df_jfk = run_airport_query('JFK', 5449, start_time_buffer, timestamp_start, end_time)

	# Combine dataframes from queries
	df_total = pd.concat([df_lga, df_ewr, df_jfk])
	df_total = df_total.reset_index(drop = True)

	# Save csv file with queries from all airports
	df_total.to_csv('data/surface_summary/all_airports_surface_data_query.csv')

	return df_total, start_time, end_time