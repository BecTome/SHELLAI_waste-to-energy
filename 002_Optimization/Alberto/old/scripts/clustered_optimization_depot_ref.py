import pandas as pd
import logging
import os 
from itertools import product
from mip import Model, xsum, minimize, OptimizationStatus, BINARY
from datetime import datetime
import numpy as np
import sys
sys.path.append('.')
from utils.config import LS_INDUSTRY_EXT
from utils.help import get_n_closer
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

np.random.seed(42)

SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output/clustering'
FORECAST_FILE = 'Biomass_History_Synthetic.csv' # Forecast File (Synthetic at the beginning) Path
DISTANCE_FILE = 'Distance_Matrix_Synthetic.csv' # Forecast File (Synthetic at the beginning) Path
N_CLUSTERS = 5

TRANSPORT_FACTOR_A = .001
DEPOT_LOWER_THRESHOLD = 5000
DEPOT_UPPER_THRESHOLD = 17000
REF_LOWER_THRESHOLD = 20000
REF_UPPER_THRESHOLD = 87000

cap_b_j = 20000 # Maximum depot capacity
cap_p_k = 100000 # Maximum production capacity
n_refineries = 5 # Number of refineries
n_depots = 25 # Number of depots

print("SYNTHETIC DATA PATH: {}".format(SYNTH_DATA_PATH))
print("OUTPUT SYNTHETIC DATA PATH: {}".format(OUT_SYNTH_DATA_PATH))
print("FORECAST FILE: {}".format(FORECAST_FILE))
print("DISTANCE FILE: {}".format(DISTANCE_FILE))
print("TRANSPORT FACTOR A: {}".format(TRANSPORT_FACTOR_A))
print("DEPOT LOWER THRESHOLD: {}".format(DEPOT_LOWER_THRESHOLD))
print("DEPOT UPPER THRESHOLD: {}".format(DEPOT_UPPER_THRESHOLD))
print("REFINERY LOWER THRESHOLD: {}".format(REF_LOWER_THRESHOLD))
print("REFINERY UPPER THRESHOLD: {}".format(REF_UPPER_THRESHOLD))
print("CAPACITY DEPOT: {}".format(cap_b_j))
print("CAPACITY PRODUCTION: {}".format(cap_p_k))
print("NUMBER OF DEPOTS: {}".format(n_depots))
print("NUMBER OF CLUSTERS: {}".format(N_CLUSTERS))


print("\n\n")
print("="*50)
print("DATA LOADING")
print("="*50)

df_fc = pd.read_csv(os.path.join(SYNTH_DATA_PATH, FORECAST_FILE))
df_matrix_orig = pd.read_csv(os.path.join(SYNTH_DATA_PATH, DISTANCE_FILE), 
                            index_col=0)

d_matrix_orig = df_matrix_orig.values

print("Forecast Matrix Shape: {}".format(df_fc.shape))
print("Forecast Matrix Columns: {}".format(df_fc.columns))
print("Distance Matrix Shape: {}".format(df_matrix_orig.shape))

print("\n\n")
print("="*50)
print("DATA PREPROCESSING")
print("="*50)
print("\n\n")

print("AGGLOMERATIVE CLUSTERING WITH {} CLUSTERS".format(N_CLUSTERS))
print("Clustering variables: Latitude, Longitude")
agg_clus = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='average', metric="precomputed").\
            fit(d_matrix_orig, df_fc[['Latitude', 'Longitude']])
            
df_clus = df_fc.copy()
df_clus['Cluster'] = agg_clus.labels_
df_clus['avg_fc'] = df_clus.loc[:, ['2018', '2019']].mean(axis=1)
total_avg = df_clus['avg_fc'].sum()
d_clus = df_clus.groupby('Cluster')['avg_fc'].sum().to_dict()

d_clus_18 = df_clus.groupby('Cluster')['2018'].sum().to_dict()
d_clus_19 = df_clus.groupby('Cluster')['2019'].sum().to_dict()

d_depots = {k: (v // DEPOT_UPPER_THRESHOLD + 1) if v > DEPOT_LOWER_THRESHOLD else 0 for k, v in d_clus.items()}
d_refs = {k: (v // REF_UPPER_THRESHOLD + 1) if v > REF_LOWER_THRESHOLD else 0 for k, v in d_clus.items()}

d_depots_18 = {k: int(v // DEPOT_UPPER_THRESHOLD + 1) if v > DEPOT_LOWER_THRESHOLD else 0 for k, v in d_clus_18.items()}
d_depots_19 = {k: int(v // DEPOT_UPPER_THRESHOLD + 1) if v > DEPOT_LOWER_THRESHOLD else 0 for k, v in d_clus_19.items()}

# d_depots = {k: for k, v in d_clus.items()}
d_depots_min = {k: np.ceil(np.max([d_clus_18[k], d_clus_19[k]]) * 0.8 / DEPOT_UPPER_THRESHOLD) for k, v in d_clus.items()}
d_depots_max = {k: np.floor(np.min([d_clus_18[k], d_clus_19[k]]) / DEPOT_UPPER_THRESHOLD) for k, v in d_clus.items()}

d_idxs_clus = {k: list(ls_idxs) for k, ls_idxs in df_clus.groupby('Cluster').groups.items()}
d_intradist = {k: d_matrix_orig[d_idxs_clus[k], :][:, d_idxs_clus[k]].max() for k in d_idxs_clus.keys()}

for k, v in d_idxs_clus.items():
    print(f"GROUP: {k} - DEPOTS: {d_depots[k]} - REFINERIES: {d_refs[k]} - N POINTS: {len(v)}")
          
print()
logging.info(f"TOTAL DEPOTS: {sum(d_depots.values())}")
logging.info(f"TOTAL REFINERIES: {sum(d_refs.values())}")
logging.info(f"TOTAL DEPOTS 18: {sum(d_depots_18.values())}")
logging.info(f"TOTAL DEPOTS 19: {sum(d_depots_19.values())}")
logging.info(f"PCT COVERED (CONSTRAINED): {sum([v for k, v in d_clus.items() if  (d_depots[k] != 0)&(d_depots_min[k]<=d_depots_max[k])]) / sum(d_clus.values()):.2%}")
logging.info(f"PCT COVERED (UNCONSTRAINED): {sum([v for k, v in d_clus.items() if  (d_depots[k] != 0)]) / sum(d_clus.values()):.2%}")

print("\n\n")
print("="*50)
print("OPTIMIZATION MODEL")
print("="*50)
print("\n\n")

df_sol = pd.DataFrame([], columns=['biomass'])
for CLUSTER in d_depots.keys():
    print('\n')
    print('*'*50)
    print('\nCluster: {}'.format(CLUSTER))
    print('\n')

    IDX_CLUS = d_idxs_clus[CLUSTER] # Indexes of the cluster
    n_depots = d_depots[CLUSTER] # Number of depots
    n_refineries = d_refs[CLUSTER] # Number of refineries

    print('Number of depots: {}'.format(n_depots))
    if n_depots == 0:
        continue

    print('Number of refineries: {}'.format(n_refineries))

    df_fc_clus = df_fc.loc[IDX_CLUS, :].copy()
    total_fc_18 = df_fc_clus.loc[:, '2018'].sum()
    total_fc_19 = df_fc_clus.loc[:, '2019'].sum()

    df_matrix = df_matrix_orig.iloc[IDX_CLUS, IDX_CLUS]

    df_matrix_obj = TRANSPORT_FACTOR_A * df_matrix.copy()

    print('Total forecast 2018: {:.2f}'.format(total_fc_18))
    print('Total forecast 2019: {:.2f}'.format(total_fc_19))
    print('Distance matrix shape: {}'.format(df_matrix.shape))

    ## REDUCE CANDIDATES 3: DELETE ARCS LONGER THAN MAXIMUM DISTANCE
    MAX_DISTANCE = 600.
    # idxs = df_matrix.where(df_matrix <= MAX_DISTANCE)
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

    ls_arcs_jk = ls_arcs_ij.copy()
    d_arcs_jk = d_arcs_ij.copy()
    d_arcs_kj = d_arcs_ji.copy()

    del df_arcs_ij

    ls_biosource = list(d_arcs_ij.keys()) # range(N) # ls_dep_red
    ls_depots = list(d_arcs_ji.keys()) # range(N) # ls_dep_red
    ls_refineries = list(d_arcs_jk.keys()) # range(N) # ls_dep_red

    # Get the forecasted biomass for year 2018 of all the positions
    d_bio_18 = df_fc_clus.loc[ls_biosource, '2018']
    d_bio_18 = d_bio_18.to_dict()
    logging.info(f"Forecasted biomass for year 2018: {total_fc_18}")


    d_bio_19 = df_fc_clus.loc[ls_biosource, '2019']
    d_bio_19 = d_bio_19.to_dict()
    logging.info(f"Forecasted biomass for year 2019: {total_fc_19}")

    # Get the solution for the optimization problem
    logging.info("\nDefine model\n")
    m = Model(sense=minimize)
    m.threads = -1

    # Variables: biomass b_{i, 0}
    # 1. All values (forecasted biomass, biomass demand-supply, pellet demand-supply) must be
    # greater than or equal to zero.

    logging.info("SET VARIABLES")

    b_18 = [m.add_var(name=f'b_2018_{i}_{j}', lb=0) for i, j in ls_arcs_ij]
    logging.info(f"Variables b_2018: {len(b_18)}")

    b_19 = [m.add_var(name=f'b_2019_{i}_{j}', lb=0) for i, j in ls_arcs_ij]
    logging.info(f"Variables b_2019: {len(b_19)}")

    p_18 = [m.add_var(name=f'p_2018_{j}_{k}', lb=0) for j, k in ls_arcs_jk]
    logging.info(f"Variables p_2018: {len(p_18)}")

    p_19 = [m.add_var(name=f'p_2019_{j}_{k}', lb=0) for j, k in ls_arcs_jk]
    logging.info(f"Variables p_2019: {len(p_19)}")

    x = [m.add_var(name=f'x_{j}', var_type=BINARY) for j in ls_depots]
    logging.info(f"Variables x: {len(x)}")

    r = [m.add_var(name=f'r_{k}', var_type=BINARY) for k in ls_refineries]
    logging.info(f"Variables r: {len(r)}")

    # Constraints:
    # 2. The amount of biomass procured for processing from each harvesting site ′𝑖𝑖′ must be less than
    # or equal to that site’s forecasted biomass.
    
    print("\n\n")
    print("SET CONSTRAINTS")
    print("\n\n")

    logging.info("Constraint 2: The processed biomass from each HS <= FC Biomass")
    for i in ls_biosource:
        m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for j in d_arcs_ij[i]) <= d_bio_18[i]
        m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for j in d_arcs_ij[i]) <= d_bio_19[i]

    logging.info("Constraint 3-4: Can't transport more than storage limit")
    for j in ls_depots:
        # 3-4. Can't transport more than storage limit
        m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in d_arcs_ji[j]) <= cap_b_j * m.var_by_name(f'x_{j}')
        m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in d_arcs_ji[j]) <= cap_b_j * m.var_by_name(f'x_{j}')
    
    for k in ls_refineries:
        m += xsum(m.var_by_name(f'p_2018_{j}_{k}') for j in d_arcs_kj[k]) <= cap_p_k * m.var_by_name(f'r_{k}')
        m += xsum(m.var_by_name(f'p_2019_{j}_{k}') for j in d_arcs_kj[k]) <= cap_p_k * m.var_by_name(f'r_{k}')

    logging.info("Constraint 8: Pellets in = Pellets out")
    for j in ls_depots:
        # 8. Total amount of biomass entering each preprocessing depot is equal to the total amount of
        # pellets exiting that depot (within tolerance limit of 1e-03
        
        m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in d_arcs_ji[j])\
            - xsum(m.var_by_name(f'p_2018_{j}_{k}') for k in d_arcs_jk[j])\
            <= .001 # * m.var_by_name(f'x_{j}')
        
        m += xsum(m.var_by_name(f'p_2018_{j}_{k}') for k in d_arcs_jk[j])\
            - xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in d_arcs_ji[j])\
            <= .001 # * m.var_by_name(f'x_{j}')

        m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in d_arcs_ji[j])\
            - xsum(m.var_by_name(f'p_2019_{j}_{k}') for k in d_arcs_jk[j])\
            <= .001 # * m.var_by_name(f'x_{j}')
        
        m += xsum(m.var_by_name(f'p_2019_{j}_{k}') for k in d_arcs_jk[j])\
            - xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in d_arcs_ji[j])\
            <= .001 #* m.var_by_name(f'x_{j}')
    
    logging.info("Constraint 6: Number of refineries should be less than or equal to 5")
    # 6. Number of refineries should be less than or equal to 5.
    m += xsum(m.var_by_name(f'r_{k}') for k in ls_refineries) <= n_refineries

    logging.info(r"Constraint 7: At least 80% of the total forecasted biomass must be processed by refineries each year")
    # 7. At least 80% of the total forecasted biomass must be processed by refineries each year
    m += xsum(m.var_by_name(f'p_2018_{j}_{k}') for j, k in ls_arcs_jk)\
        >= 0.8 * total_fc_18
    m += xsum(m.var_by_name(f'p_2019_{j}_{k}') for j, k in ls_arcs_jk)\
        >= 0.8 * total_fc_19

    logging.info(f"Constraint 5: Number of depots should be less than or equal to {n_depots}")
    # 5. Number of depots should be less than or equal to 25.
    m += xsum(m.var_by_name(f'x_{j}') for j in ls_depots) <= n_depots


    logging.info(r"Constraint 7: At least 80% of the total forecasted biomass must be processed by depots each year")
    # 7. At least 80% of the total forecasted biomass must be processed by refineries each year
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i, j in ls_arcs_ij)\
        >= 0.8 * total_fc_18
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i, j in ls_arcs_ij)\
        >= 0.8 * total_fc_19

    logging.info(f'Number of constraints: {m.num_rows}')                  # number of rows (constraints) in the model
    logging.info(f'Number of variables: {m.num_cols}')                    # number of columns (variables) in the model
    logging.info(f'Number of integer variables: {m.num_int}')             # number of integer variables in the model
    logging.info(f'Number of non-zeros in constraint matrix: {m.num_nz}') # number of non-zeros in the constraint matrix

    print("\n")
    print("SET OBJECTIVE FUNCTION")
    print("\n")
    # Objective function:
    m.objective = minimize(
                        xsum(df_matrix_obj.loc[i, str(j)] * (m.var_by_name(f'b_2018_{i}_{j}') + m.var_by_name(f'b_2019_{i}_{j}')) +\
                                - m.var_by_name(f'b_2018_{i}_{j}') - m.var_by_name(f'b_2019_{i}_{j}')\
                                for i, j in ls_arcs_ij) + \
                        xsum(df_matrix_obj.loc[j, str(k)] * (m.var_by_name(f'p_2018_{j}_{k}') + m.var_by_name(f'p_2019_{j}_{k}')) +\
                                - m.var_by_name(f'p_2018_{j}_{k}') - m.var_by_name(f'p_2019_{j}_{k}')
                                for j, k in ls_arcs_jk) + \
                        xsum(2 * cap_b_j*m.var_by_name(f'x_{j}') for j in ls_depots) + \
                        xsum(2 * cap_p_k*m.var_by_name(f'r_{k}') for k in ls_refineries)\
                        )
    
    print("\n")
    print("SOLVE MODEL")
    print("\n")

    logging.info("Start optimization")
    print("\n\n")
    # Solve the problem
    status = m.optimize(max_seconds=100) #m.optimize(max_seconds=100) 

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    MODEL_NAME = f'model_{dt_string}.lp'
    OUTPUT_MODEL_PATH = os.path.join(OUT_SYNTH_DATA_PATH, "models", MODEL_NAME)

    # logging.info(f"Write model in: {OUTPUT_MODEL_PATH}")
    # m.write(OUTPUT_MODEL_PATH)

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
        # logging.info('solution:')
        d_sol = {}
        for v in m.vars:
            d_sol.update({v.name: v.x})

        # logging.info("Solution: ", d_sol)
        df_sol_clus = pd.DataFrame.from_dict(d_sol, orient='index', columns=['biomass'])
        # df_sol.to_csv(os.path.join(OUT_SYNTH_DATA_PATH, f'solution_{dt_string}.csv'))

        df_sol_clus = df_sol_clus[df_sol_clus['biomass'] > 0]
        df_sol = pd.concat([df_sol, df_sol_clus])

        idxs_depots_sol = df_sol_clus.filter(regex='x_', axis=0).copy()#[df_sol['biomass'] != 0].index
        idxs_depots_sol = idxs_depots_sol[idxs_depots_sol['biomass'] == 1]
        idxs_depots_sol = idxs_depots_sol.index.str.split('_', expand=True).get_level_values(1).astype(int).unique()
        print("\n\n")
        print(f"COORDINATES OF THE DEPOTS: {idxs_depots_sol}")    
        print("\n\n")

        idxs_refs_sol = df_sol_clus.filter(regex='r_', axis=0).copy()#[df_sol['biomass'] != 0].index
        idxs_refs_sol = idxs_refs_sol[idxs_refs_sol['biomass'] == 1]
        idxs_refs_sol = idxs_refs_sol.index.str.split('_', expand=True).get_level_values(1).astype(int).unique()
        print("\n\n")
        print(f"COORDINATES OF THE REFINERIES: {idxs_refs_sol}")    
        print("\n\n")

print("\n\n")
print("="*50)
print("WRITE SOLUTION")
print("="*50)

logging.info(f"Write solution in: {OUTPUT_MODEL_PATH}")
df_sol.to_csv(os.path.join(OUT_SYNTH_DATA_PATH, f'solution_{dt_string}.csv'))

print("\n\n")
print("="*50)
print("END OF THE SCRIPT")
print("="*50)