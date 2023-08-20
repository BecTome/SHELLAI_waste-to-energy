SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output/clustering'
FORECAST_FILE = 'Biomass_History_Forecast.csv' # Forecast File (Synthetic at the beginning) Path
DISTANCE_FILE = 'Distance_Matrix_Synthetic.csv' # Distance Matrix File Path
N_CLUSTERS = 25
DEPOT_SOL_EXT = 'DEPOTS.csv'

TRANSPORT_FACTOR_A = .001
DEPOT_LOWER_THRESHOLD = 8000
DEPOT_UPPER_THRESHOLD = 15000

cap_b_j = 20000 - 1e-2 # Maximum depot capacity
cap_p_k = 100000 - 1e-2# Maximum production capacity

