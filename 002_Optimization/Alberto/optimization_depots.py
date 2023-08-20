##################################################################

# Script that solves the optimization problem of the depots
# Structure of the problem:
# 1. The problem is divided in clusters
# 2. For each cluster, the problem is solved
# 3. The solution of each cluster is saved in a dataframe
# 4. The solution of all the clusters is saved in a csv file
# 5. This csv file feeds the script that generates the refineries

##################################################################

# Import libraries
import pandas as pd
import logging
import os 
from mip import Model, xsum, minimize, OptimizationStatus, BINARY
from datetime import datetime
import numpy as np
import sys
sys.path.append('.')
from sklearn.cluster import AgglomerativeClustering


print("\n\n")
print("="*50)
print("OPTIMIZATION MODEL: Division In Clusters")
print("="*50)
print("\n\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

print("\n\n")
print("="*50)
print("CONSTANTS DEFINITION")
print("="*50)
print("\n\n")

# Define constants
np.random.seed(42)

SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output/clustering'
FORECAST_FILE = 'Biomass_History_Forecast.csv' # Forecast File (Synthetic at the beginning) Path
DISTANCE_FILE = 'Distance_Matrix_Synthetic.csv' # Distance Matrix File Path
DEPOT_SOL_EXT = 'DEPOTS.csv'

N_CLUSTERS = 25
CLUSTERING_METHOD = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='average', 
                                            metric="precomputed")

N_DEPOTS = 25

TRANSPORT_FACTOR_A = .001
DEPOT_LOWER_THRESHOLD = 8000
DEPOT_UPPER_THRESHOLD = 15000

cap_b_j = 20000 - 1e-2 # Maximum depot capacity
cap_p_k = 100000 - 1e-2# Maximum production capacity

print("SYNTHETIC DATA PATH: {}".format(SYNTH_DATA_PATH))
print("OUTPUT SYNTHETIC DATA PATH: {}".format(OUT_SYNTH_DATA_PATH))
print("FORECAST FILE: {}".format(FORECAST_FILE))
print("DISTANCE FILE: {}".format(DISTANCE_FILE))
print("TRANSPORT FACTOR A: {}".format(TRANSPORT_FACTOR_A))
print("DEPOT LOWER THRESHOLD: {}".format(DEPOT_LOWER_THRESHOLD))
print("DEPOT UPPER THRESHOLD: {}".format(DEPOT_UPPER_THRESHOLD))
print("CAPACITY DEPOT: {}".format(cap_b_j))
print("CAPACITY PRODUCTION: {}".format(cap_p_k))
print("NUMBER OF DEPOTS: {}".format(N_DEPOTS))
print("NUMBER OF CLUSTERS: {}".format(N_CLUSTERS))


print("\n\n")
print("="*50)
print("DATA LOADING")
print("="*50)

# Load data
# Forecast matrix
df_fc = pd.read_csv(os.path.join(SYNTH_DATA_PATH, FORECAST_FILE))

# Distance matrix
df_matrix_orig = pd.read_csv(os.path.join(SYNTH_DATA_PATH, DISTANCE_FILE), 
                            index_col=0)

d_matrix_orig = df_matrix_orig.values # Distance matrix as numpy array

print("Forecast Matrix Shape: {}".format(df_fc.shape))
print("Forecast Matrix Columns: {}".format(df_fc.columns))
print("Distance Matrix Shape: {}".format(df_matrix_orig.shape))

print("\n\n")
print("="*50)
print("DATA PREPROCESSING")
print("="*50)
print("\n\n")

# CLUSTERING
print("AGGLOMERATIVE CLUSTERING WITH {} CLUSTERS".format(N_CLUSTERS))
print("Clustering variables: Latitude, Longitude")

# Fit the model using precomputed distance matrix
agg_clus = CLUSTERING_METHOD.fit(d_matrix_orig, df_fc[['Latitude', 'Longitude']])

# Use the average forecast of the two years in order to make decissions about the number
# of depot candidates            
df_clus = df_fc.copy()
df_clus['Cluster'] = agg_clus.labels_
df_clus['avg_fc'] = df_clus.loc[:, ['2018', '2019']].mean(axis=1)
total_avg = df_clus['avg_fc'].sum()
d_clus = df_clus.groupby('Cluster')['avg_fc'].sum().to_dict()

# Dictionaries with the total biomass of each cluster for each year
d_clus_18 = df_clus.groupby('Cluster')['2018'].sum().to_dict()
d_clus_19 = df_clus.groupby('Cluster')['2019'].sum().to_dict()

# Dictionaries with the number of depots for each cluster
# The criterion is that there are depots enough to cover the demand of the cluster
# To add an extra one there should be at least DEPOT_LOWER_THRESHOLD biomass remaining
d_depots = {k: ((v - DEPOT_LOWER_THRESHOLD) // DEPOT_UPPER_THRESHOLD) + 1 for k, v in d_clus.items()}

# Indexes of the clusters in a dictionary and the internal max distance of each cluster
d_idxs_clus = {k: list(ls_idxs) for k, ls_idxs in df_clus.groupby('Cluster').groups.items()}
d_intradist = {k: d_matrix_orig[d_idxs_clus[k], :][:, d_idxs_clus[k]].max() for k in d_idxs_clus.keys()}

# Info about the clusters
for k, v in d_idxs_clus.items():
    print(f"GROUP: {k} - DEPOTS: {d_depots[k]} - N POINTS: {len(v)}"+\
                    f" - BIOMASS 2018: {d_clus_18[k]:.2f} - BIOMASS 2019: {d_clus_19[k]:.2f} - MAX INTRADISTANCE: {d_intradist[k]:.2f}")
print()
logging.info(f"TOTAL DEPOTS: {sum(d_depots.values())}")

print("\n\n")
print("="*50)
print("OPTIMIZATION MODEL")
print("="*50)
print("\n\n")

# Optimization model for each cluster
df_sol = pd.DataFrame([], columns=['biomass'])
for CLUSTER in list(d_depots.keys()):
    print('\n')
    print('*'*50)
    print('\nCluster: {}'.format(CLUSTER))
    print('\n')

    IDX_CLUS = d_idxs_clus[CLUSTER] # Indexes of the cluster
    n_depots = d_depots[CLUSTER] # Number of depots for the cluster

    print('Number of depots: {}'.format(n_depots))
    if n_depots == 0:
        continue
    
    # Get the forecasted biomass for years 2018 and 2019 of all the positions 
    # within the cluster and the distances between them
    df_fc_clus = df_fc.loc[IDX_CLUS, :].copy()
    total_fc_18 = df_fc_clus.loc[:, '2018'].sum()
    total_fc_19 = df_fc_clus.loc[:, '2019'].sum()

    df_matrix = df_matrix_orig.iloc[IDX_CLUS, IDX_CLUS]
    df_matrix_obj = TRANSPORT_FACTOR_A * df_matrix.copy() # For Obj Function

    print('Total forecast 2018: {:.2f}'.format(total_fc_18))
    print('Total forecast 2019: {:.2f}'.format(total_fc_19))
    print('Distance matrix shape: {}'.format(df_matrix.shape))

    ## REDUCE CANDIDATES 3: DELETE ARCS LONGER THAN MAXIMUM DISTANCE
    # It's supposed that the distance matrix is symmetric ()
    MAX_DISTANCE = 10000.
    st_df_matrix = df_matrix.stack().copy()
    st_df_matrix = st_df_matrix[st_df_matrix <= MAX_DISTANCE]
    df_arcs_ij = pd.DataFrame(st_df_matrix.index.get_level_values(1), 
                            index=st_df_matrix.index.get_level_values(0)).reset_index()
    df_arcs_ij.columns = ['i', 'j']
    df_arcs_ij['j'] = df_arcs_ij['j'].astype(int)
    ls_arcs_ij = df_arcs_ij.values
    d_arcs_ij = {}
    d_arcs_ji = {}
    for key, value in ls_arcs_ij:
        d_arcs_ij.setdefault(key, []).append(value)
        d_arcs_ji.setdefault(value, []).append(key)

    # Get the list of the indexes of the biomass sources and the depots
    # From the reduced list of candidates
    ls_biosource = list(d_arcs_ij.keys()) # range(N) # ls_dep_red
    ls_depots = list(d_arcs_ji.keys()) # range(N) # ls_dep_red

    # Get the forecasted biomass for year 2018 of all the positions
    d_bio_18 = df_fc_clus.loc[ls_biosource, '2018']
    d_bio_19 = df_fc_clus.loc[ls_biosource, '2019']

    d_bio_18 = d_bio_18.to_dict()
    d_bio_19 = d_bio_19.to_dict()

    logging.info(f"Forecasted biomass for year 2018: {total_fc_18}")
    logging.info(f"Forecasted biomass for year 2019: {total_fc_19}")

    # Get the solution for the optimization problem
    logging.info("\nDefine model\n")
    m = Model(sense=minimize)
    m.threads = -1 # Parallel threads

    # Variables: biomass b_{i, 0}
    # 1. All values (forecasted biomass, biomass demand-supply, pellet demand-supply) must be
    # greater than or equal to zero.

    logging.info("SET VARIABLES")
    b_18 = [m.add_var(name=f'b_2018_{i}_{j}', lb=0) for i, j in ls_arcs_ij]
    logging.info(f"Variables b_2018: {len(b_18)}")

    b_19 = [m.add_var(name=f'b_2019_{i}_{j}', lb=0) for i, j in ls_arcs_ij]
    logging.info(f"Variables b_2019: {len(b_19)}")

    x = [m.add_var(name=f'x_{j}', var_type=BINARY) for j in ls_depots]
    logging.info(f"Variables x: {len(x)}")

    # Constraints:
    # 2. The amount of biomass procured for processing from each harvesting site ′𝑖𝑖′ must be less than
    # or equal to that site’s forecasted biomass.
    
    print("\n\n")
    print("SET CONSTRAINTS")
    print("\n\n")

    logging.info("Constraint 2: The processed biomass from each HS <= FC Biomass")
    for i in ls_biosource:
        m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for j in d_arcs_ij[i]) - d_bio_18[i] <= 1e-8
        m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for j in d_arcs_ij[i]) - d_bio_19[i] <= 1e-8

    logging.info("Constraint 3-4: Can't transport more than storage limit")
    for j in ls_depots:
        # 3-4. Can't transport more than storage limit
        m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in d_arcs_ji[j]) <= cap_b_j * m.var_by_name(f'x_{j}')
        m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in d_arcs_ji[j]) <= cap_b_j * m.var_by_name(f'x_{j}')
    
    logging.info(f"Constraint 5: Number of depots should be less than or equal to {n_depots}")
    # 5. Number of depots should be less than or equal to 25.
    m += xsum(m.var_by_name(f'x_{j}') for j in ls_depots) <= n_depots

    logging.info(r"Constraint 7: At least 80% of the total forecasted biomass must be processed by depots each year")
    # 7. At least 80% of the total forecasted biomass must be processed by refineries each year
    m += total_fc_18 - 1e-2>= xsum(m.var_by_name(f'b_2018_{i}_{j}') for i, j in ls_arcs_ij)\
        >= 0.8 * total_fc_18 + 1e-2
    m += total_fc_19 - 1e-2 >= xsum(m.var_by_name(f'b_2019_{i}_{j}') for i, j in ls_arcs_ij)\
        >= 0.8 * total_fc_19 + 1e-2

    logging.info(f'Number of constraints: {m.num_rows}')                  # number of rows (constraints) in the model
    logging.info(f'Number of variables: {m.num_cols}')                    # number of columns (variables) in the model
    logging.info(f'Number of integer variables: {m.num_int}')             # number of integer variables in the model
    logging.info(f'Number of non-zeros in constraint matrix: {m.num_nz}') # number of non-zeros in the constraint matrix

    print("\n")
    print("SET OBJECTIVE FUNCTION")
    print("\n")
    # Objective function:
    # Transport cost + empty depot cost
    m.objective = minimize(
                        xsum(df_matrix_obj.loc[i, str(j)] * (m.var_by_name(f'b_2018_{i}_{j}') + m.var_by_name(f'b_2019_{i}_{j}')) +\
                                - m.var_by_name(f'b_2018_{i}_{j}') - m.var_by_name(f'b_2019_{i}_{j}')\
                                for i, j in ls_arcs_ij) + \
                        xsum(2 * cap_b_j*m.var_by_name(f'x_{j}') for j in ls_depots)
                        )
    
    print("\n")
    print("SOLVE MODEL")
    print("\n")

    logging.info("Start optimization")
    print("\n\n")
    # Solve the problem
    status = m.optimize(max_seconds=100) # Max seconds for the optimization

    print("\n\n")
    print("MODEL STATUS AND SOLUTION")
    print("\n\n")

    logging.info(status)
    # Check the status and show the solutions
    if status == OptimizationStatus.OPTIMAL:
        logging.info('optimal solution cost {} found'.format(m.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        logging.info('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
    elif status in [OptimizationStatus.NO_SOLUTION_FOUND, OptimizationStatus.INFEASIBLE]:
        logging.info('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        # Save the solution in a dictionary
        d_sol = {}
        for v in m.vars:
            d_sol.update({v.name: v.x})

        df_sol_clus = pd.DataFrame.from_dict(d_sol, orient='index', columns=['biomass'])

        # Save the solution in a dataframe and drop the vars with zero biomass
        df_sol_clus = df_sol_clus[df_sol_clus['biomass'] > 0]
        df_sol = pd.concat([df_sol, df_sol_clus])

        # Get the indexes of the depots and show them
        idxs_depots_sol = df_sol_clus.filter(regex='x_', axis=0).copy()
        idxs_depots_sol = idxs_depots_sol[idxs_depots_sol['biomass'] >0]
        idxs_depots_sol = idxs_depots_sol.index.str.split('_', expand=True).get_level_values(1).astype(int).unique()

        print("\n\n")
        print(f"COORDINATES OF THE DEPOTS: {idxs_depots_sol}")    
        print("\n\n")

print("\n\n")
print("="*50)
print("WRITE SOLUTION")
print("="*50)

# Write the solution in a csv file with the current timestamp
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
MODEL_NAME = f'model_{dt_string}.lp'
OUTPUT_MODEL_PATH = os.path.join(OUT_SYNTH_DATA_PATH, "models", MODEL_NAME)
OUTPUT_CSV_PATH = os.path.join(OUT_SYNTH_DATA_PATH, "solutions", "depots", DEPOT_SOL_EXT)

logging.info(f"Write solution (MODEL) in: {OUTPUT_MODEL_PATH}")
logging.info(f"Write solution (CSV FILE) in: {OUTPUT_CSV_PATH}")

df_sol.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n\n")
print("="*50)
print("END OF THE SCRIPT")
print("="*50)