import pandas as pd
import sys
import os
sys.path.append('.')

INPUT_PATH = 'data/forecasts/biomass_forecasted.csv'
OUTPUT_PATH = '002_Optimization/data/Biomass_History_Forecast.csv'

print("\n\n")
print("="*50)
print("READ DATA FROM: ", INPUT_PATH)
print("="*50)

# Read the data
df_fc_orig = pd.read_csv(INPUT_PATH)

print("\n\n")
print("="*50)
print("PROCESS DATA", INPUT_PATH)
print("="*50)

# Process the data
# 1. Extract the year from Date
df_fc_orig['Date'] = pd.to_datetime(df_fc_orig['Date'])
df_fc_orig['year'] = df_fc_orig['Date'].dt.year
df_fc_orig = df_fc_orig[["Latitude", "Longitude", "Forecast", "year"]]

# 2. Pivot the table so that there are a 2018 and 2019 column
df_fc_proc = df_fc_orig.pivot_table(index=["Latitude", "Longitude"], columns="year", values="Forecast").reset_index()
df_fc_proc.columns = ["Latitude", "Longitude", "2018", "2019"]
df_fc_proc = df_fc_proc.reset_index(names="Index")

print("\n\n")
print("SIZE OF DATAFRAME: ", df_fc_proc.shape)
print("\n")
print(df_fc_proc.head())
print("\n")

# Write the data
df_fc_proc.to_csv(OUTPUT_PATH, index=False)

print("\n\n")
print("DATAFRAME SAVED TO: ", OUTPUT_PATH)

print("\n\n")
print("="*50)
print("END OF THE SCRIPT")
print("="*50)