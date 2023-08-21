import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')

SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output/clustering/refineries'
OUT_SUBMISSION_PATH = '002_Optimization/output/clustering/submissions'
FORECAST_FILE = 'Biomass_History_Forecast.csv' # Forecast File (Synthetic at the beginning) Path
SAMPLE_SUBMISSION_FILE = 'sample_submission.csv'
SOLUTION_FILE = 'solution_20_08_2023_21_04_19.csv'
SUBMISSION_FILE = f'submission_{SOLUTION_FILE}.csv'

df_submission = pd.read_csv(os.path.join(SYNTH_DATA_PATH, SAMPLE_SUBMISSION_FILE))
df_sol = pd.read_csv(os.path.join(OUT_SYNTH_DATA_PATH, SOLUTION_FILE))
df_sol.columns = ["data_type", "solution"]
df_fc = pd.read_csv(os.path.join(SYNTH_DATA_PATH, FORECAST_FILE))

# Format Forecasted Data for submission
df_fc_sol = df_fc.copy()
df_fc_sol = df_fc_sol.melt(value_vars=['2018', '2019'], var_name='year', value_name='biomass', 
                           id_vars=['Index'])
df_fc_sol['data_type'] = 'biomass_forecast'
df_fc_sol['destination_index'] = None
df_fc_sol = df_fc_sol[['data_type', 'year', 'Index', 'destination_index', 'biomass']]
df_fc_sol.columns = ['data_type', 'year', 'source_index', 'destination_index', 'value']

# Format Solution Data for submission
df_sol_proc = df_sol.copy()
df_sol_proc['data_type'] = df_sol_proc['data_type'].\
                            str.replace('x_', 'x_20182019_').\
                            str.replace('r_', 'r_20182019_')

df_sol_proc = df_sol_proc['data_type'].str.split("_", expand=True)
df_sol_proc.columns = ['data_type', 'year', 'source_index', 'destination_index']

df_sol_proc['value'] = df_sol['solution']
df_sol_proc['data_type'] = df_sol_proc['data_type'].map({'b': 'biomass_demand_supply', 
                                                         'p': 'pellet_demand_supply', 
                                                         'x': 'depot_location', 
                                                         'r': 'refinery_location'})
df_sol_proc = df_sol_proc[df_sol_proc['value'] != 0]

# Concatenate forecast and solution data
df_sol_proc = pd.concat([df_sol_proc, df_fc_sol])


# Check if the submission is correct

## Constraint 2: The processed biomass from each HS <= FC Biomass
cond_bio_sent_18 = (df_sol_proc['year'] == '2018')&(df_sol_proc['data_type'] == 'biomass_demand_supply')
cond_bio_sent_19 = (df_sol_proc['year'] == '2019')&(df_sol_proc['data_type'] == 'biomass_demand_supply')

provided_bio_18 = df_sol_proc[cond_bio_sent_18].groupby('source_index')[['value']].sum()
provided_bio_19 = df_sol_proc[cond_bio_sent_19].groupby('source_index')[['value']].sum()

provided_bio_18.index = provided_bio_18.index.astype(int)
provided_bio_19.index = provided_bio_19.index.astype(int)

cond_bio_fc_18 = (df_sol_proc['year'] == '2018')&(df_sol_proc['data_type'] == 'biomass_forecast')
cond_bio_fc_19 = (df_sol_proc['year'] == '2019')&(df_sol_proc['data_type'] == 'biomass_forecast')

forecasted_bio_18 = df_sol_proc[cond_bio_fc_18].groupby('source_index')[['value']].sum().reset_index(drop=True)
forecasted_bio_19 = df_sol_proc[cond_bio_fc_19].groupby('source_index')[['value']].sum().reset_index(drop=True)

error_c2_18 = (provided_bio_18 - forecasted_bio_18.iloc[provided_bio_18.index.astype(int), :]).values.sum()
error_c2_19 = (provided_bio_19 - forecasted_bio_19.iloc[provided_bio_19.index.astype(int), :]).values.sum()

assert np.all(provided_bio_18.values <= forecasted_bio_18.iloc[provided_bio_18.index.astype(int), :].values),\
            f"RESTRICTION 2 NOT SATISFIED 2018: {error_c2_18:.4f}"
assert np.all(provided_bio_18.values <= forecasted_bio_18.iloc[provided_bio_18.index.astype(int), :].values),\
            f"RESTRICTION 2 NOT SATISFIED 2019: {error_c2_19:.4f}"

## Constraint 3: The stocked biomass in each HS <= 20000
# Its important to notice that we should check that the ARRIVING and EXITING biomass is <= 20000
# for each DEPOT. This condition is closely related to the 8th Constraint.

### INTO DEPOT < 20k
stocked_bio_18 = df_sol_proc[cond_bio_sent_18].groupby('destination_index')[['value']].sum()
stocked_bio_19 = df_sol_proc[cond_bio_sent_19].groupby('destination_index')[['value']].sum()

max_stocked_18 = stocked_bio_18.max()[0]
max_stocked_19 = stocked_bio_19.max()[0]

print(f"Max stocked biomass in 2018: {max_stocked_18}")
print(f"Max stocked biomass in 2019: {max_stocked_19}")

assert (stocked_bio_18 <= 20000.).values.all(), f"Constraint 3 violated for 2018 (INTO DEPOT): {max_stocked_18}"
assert (stocked_bio_19 <= 20000.).values.all(), f"Constraint 3 violated for 2019 (INTO DEPOT): {max_stocked_19}"

### OUT OF DEPOT < 20k
cond_pellet_sent_18 = (df_sol_proc['year'] == '2018')&(df_sol_proc['data_type'] == 'pellet_demand_supply')
cond_pellet_sent_19 = (df_sol_proc['year'] == '2019')&(df_sol_proc['data_type'] == 'pellet_demand_supply')

stocked_bio_sent_18 = df_sol_proc[cond_pellet_sent_18].groupby('source_index')[['value']].sum()
stocked_bio_sent_19 = df_sol_proc[cond_pellet_sent_19].groupby('source_index')[['value']].sum()

max_stocked_sent_18 = stocked_bio_sent_18.max()[0]
max_stocked_sent_19 = stocked_bio_sent_19.max()[0]

print(f"Max stocked SENT biomass in 2018: {max_stocked_sent_18}")
print(f"Max stocked SENT biomass in 2019: {max_stocked_sent_19}")

assert max_stocked_sent_18 <= 20000. , f"Constraint 3 violated for 2018 (OUT OF DEPOT): {max_stocked_sent_18}"
assert max_stocked_sent_19 <= 20000. , f"Constraint 3 violated for 2019 (OUT OF DEPOT): {max_stocked_sent_19}"

## Constraint 4: The processed biomass in each refinery <= 100000

processed_bio_18 = df_sol_proc[cond_pellet_sent_18].groupby('destination_index')[['value']].sum()
processed_bio_19 = df_sol_proc[cond_pellet_sent_19].groupby('destination_index')[['value']].sum()

max_processed_18 = processed_bio_18.max()[0]
max_processed_19 = processed_bio_19.max()[0]

print(f"Max processed biomass in 2018: {max_processed_18}")
print(f"Max processed biomass in 2019: {max_processed_19}")

assert (processed_bio_18 <= 100000.).values.all(), f"Constraint 3 violated for 2018: {max_processed_18}"
assert (processed_bio_19 <= 100000.).values.all(), f"Constraint 3 violated for 2019: {max_processed_19}"

## Constraint 5: Number of depots should be less than or equal to 25
depot_sol = len(df_sol_proc[df_sol_proc['data_type'] == 'depot_location'])
print("N Depots: ", depot_sol)
assert depot_sol <= 25, f"Constraint 5 violated: {depot_sol}"

## Constraint 6: Number of refineries should be less than or equal to 5
refinery_sol = len(df_sol_proc[df_sol_proc['data_type'] == 'refinery_location'])
print("N Refineries: ", refinery_sol)
assert refinery_sol <= 5, f"Constraint 6 violated: {refinery_sol}"

## Constraint 7: The 80% of the forecast biomass should be processed
total_processed_18 = df_sol_proc[cond_pellet_sent_18]['value'].sum()
total_forecasted_18 = df_sol_proc[cond_bio_fc_18]['value'].sum()
total_processed_19 = df_sol_proc[cond_pellet_sent_19]['value'].sum()
total_forecasted_19 = df_sol_proc[cond_bio_fc_19]['value'].sum()

print('Processed 2018: ', total_processed_18)
print('Forecast 2018: ', total_forecasted_18)
print("Ratio: ", total_processed_18/total_processed_18)
print('Processed 2018: ', total_processed_19)
print('Forecast 2018: ', total_forecasted_19)
print("Ratio: ", total_processed_19/total_processed_19)


## Constraint 8: FLUX CONDITION
pellets_in_18 = stocked_bio_18.copy()
pellets_out_18 = stocked_bio_sent_18.copy()
pellets_in_19 = stocked_bio_19.copy()
pellets_out_19 = stocked_bio_sent_19.copy()

max_dif_18 = (pellets_out_18 - pellets_in_18).abs().max().values[0]
max_dif_19 = (pellets_out_19 - pellets_in_19).abs().max().values[0]

print(f'Pellets in and out of 2018 match: {max_dif_18}')
print(f'Pellets in and out of 2019 match: {max_dif_19}')

assert max_dif_18 <= .001, f'Pellets in and out of 2018 do not match: {max_dif_18}'
assert max_dif_19 <= .001, f'Pellets in and out of 2018 do not match: {max_dif_19}'

# Write the data
df_sol_proc.to_csv(os.path.join(OUT_SUBMISSION_PATH, SUBMISSION_FILE), index=False)
