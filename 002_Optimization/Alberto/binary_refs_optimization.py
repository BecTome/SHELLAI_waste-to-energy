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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output/binary_refs'
FORECAST_FILE = 'Biomass_History_Synthetic.csv' # Forecast File (Synthetic at the beginning) Path
DISTANCE_FILE = 'Distance_Matrix_Synthetic.csv' # Forecast File (Synthetic at the beginning) Path

N = 20
TRANSPORT_FACTOR_A = .001

df_matrix = pd.read_csv(os.path.join(SYNTH_DATA_PATH, DISTANCE_FILE), 
                       index_col=0)

d_matrix = TRANSPORT_FACTOR_A * df_matrix.values #[:N, :N]

df_fc = pd.read_csv(os.path.join(SYNTH_DATA_PATH, FORECAST_FILE))
df_fc = df_fc#.iloc[:N, :]

df_get_idx = df_fc.loc[:, ['2018', '2019']].mean(axis=1)

## REDUCE CANDIDATES 1: SELECT ONLY THE CANDIDATES THAT ARE CLOSER TO THE INDUSTRIAL AREA AND
## THAT HAVE A BIOMASS GREATER THAN 15,000 TONS PER YEAR
# d = 70

# ls_depots_dist = []
# for center in LS_INDUSTRY_EXT:
#     df_center = pd.DataFrame(d_matrix[:, center], columns=['distance'])
#     df_center["biomass"] = df_get_idx.values
#     bmu50 = df_center[df_center['distance'] <= d].biomass.sum()
#     if bmu50 >= 15000:
#         ls_depots_dist.append(center)

# ls_refineries = df_matrix.iloc[ls_depots, ls_depots].mean(axis=1).sort_values(ascending=True)[:100].index.tolist()
# ls_biosource = df_matrix.iloc[ls_depots, ls_depots].mean(axis=1).sort_values(ascending=False)[:500].index.tolist()

## REDUCE CANDIDATES 2: GET N SURROUNDING CANDIDATES TO INDUSTRIAL AREAS
# arr_n_closer_ind = get_n_closer(df_matrix.iloc[LS_INDUSTRY_EXT, :].values, n=5, uniques=True)
# arr_n_closer_ind = np.concatenate([arr_n_closer_ind, np.array(LS_INDUSTRY_EXT)])
# arr_n_closer_ind = np.unique(arr_n_closer_ind)
# ls_red = list(arr_n_closer_ind)
# logging.info(f"Number of reduced candidates [HS]: {len(ls_red)}")

# arr_n_closer_dep_ind = get_n_closer(df_matrix.iloc[LS_INDUSTRY_EXT, :].values, n=5, uniques=True)
# arr_n_closer_dep_ind = np.concatenate([arr_n_closer_dep_ind, np.array(LS_INDUSTRY_EXT)])
# arr_n_closer_dep_ind = np.unique(arr_n_closer_dep_ind)
# ls_dep_red = list(arr_n_closer_dep_ind)
# logging.info(f"Number of reduced candidates [DEPOT]: {len(ls_dep_red)}")
# logging.info(f"Number of reduced candidates [REF]: {len(ls_dep_red)}")

## REDUCE CANDIDATES 3: DELETE ARCS LONGER THAN MAXIMUM DISTANCE
MAX_DISTANCE = 100.
idxs = np.where(df_matrix.values <= MAX_DISTANCE)
df_arcs_ij = pd.DataFrame(idxs[1].reshape(-1,1), index=idxs[0]).reset_index()
df_arcs_ij.columns = ['i', 'j']
ls_arcs = df_arcs_ij.values
d_arcs_ij = {}
d_arcs_ji = {}
for key, value in ls_arcs:
    d_arcs_ij.setdefault(key, []).append(value)
    d_arcs_ji.setdefault(value, []).append(key)

ls_biosource = np.unique(ls_arcs[:, 0].reshape(-1)) # range(N) # ls_dep_red
ls_refineries =  np.unique(ls_arcs[:, 1].reshape(-1)) # ls_red 

# ls_arcs_ijk = list(product(ls_biosource, ls_depots, ls_refineries))
logging.info(f"Number of combinations: {len(ls_arcs)}")
# ls_refineries = ls_depots.copy()
# logging.info("Number of depot candidates: ", len(ls_depots))

cap_p_k = 5000 # Maximum production capacity
n_refineries = 5 # Number of refineries

# Get the forecasted biomass for year 2018 of all the positions
d_bio_18 = df_fc.loc[ls_biosource, '2018']
total_fc_18 = d_bio_18.sum()
d_bio_18 = d_bio_18.to_dict()
logging.info(f"Forecasted biomass for year 2018: {total_fc_18}")


d_bio_19 = df_fc.loc[ls_biosource, '2019']
total_fc_19 = d_bio_19.sum()
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
b_18 = [m.add_var(name=f'b_2018_{i}_{j}', lb=0) for i, j in ls_arcs]
logging.info(f"Variables b_2018: {len(b_18)}")

b_19 = [m.add_var(name=f'b_2019_{i}_{j}', lb=0) for i, j in ls_arcs]
logging.info(f"Variables b_2019: {len(b_19)}")

r = [m.add_var(name=f'r_{k}', var_type=BINARY) for k in ls_refineries]
logging.info(f"Variables r: {len(r)}")

# Constraints:
# 2. The amount of biomass procured for processing from each harvesting site ′𝑖𝑖′ must be less than
# or equal to that site’s forecasted biomass.
logging.info("\nSET CONSTRAINTS")
logging.info("Constraint 2: The processed biomass from each HS <= FC Biomass")
for i in ls_biosource:
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for j in d_arcs_ij[i]) <= d_bio_18[i]
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for j in d_arcs_ij[i]) <= d_bio_19[i]

logging.info("Constraint 3-4: Can't transport more than storage limit")
for j in ls_refineries:
    # 3-4. Can't transport more than storage limit
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in d_arcs_ji[j]) <= cap_p_k * m.var_by_name(f'r_{j}')
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in d_arcs_ji[j]) <= cap_p_k * m.var_by_name(f'r_{j}')


logging.info("Constraint 6: Number of refineries should be less than or equal to 5")
# 6. Number of refineries should be less than or equal to 5.
m += xsum(m.var_by_name(f'r_{k}') for k in ls_refineries) <= n_refineries

logging.info(r"Constraint 7: At least 80% of the total forecasted biomass must be processed by refineries each year")
# 7. At least 80% of the total forecasted biomass must be processed by refineries each year
m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i, j in ls_arcs)\
    >= 0.8 * total_fc_18
m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i, j in ls_arcs)\
    >= 0.8 * total_fc_19

logging.info(f'Number of constraints: {m.num_rows}')                  # number of rows (constraints) in the model
logging.info(f'Number of variables: {m.num_cols}')                    # number of columns (variables) in the model
logging.info(f'Number of integer variables: {m.num_int}')             # number of integer variables in the model
logging.info(f'Number of non-zeros in constraint matrix: {m.num_nz}') # number of non-zeros in the constraint matrix

logging.info("\nSET OBJECTIVE FUNCTION")
# Objective function:
m.objective = minimize(
                       xsum(d_matrix[i, j] * (m.var_by_name(f'b_2018_{i}_{j}') + m.var_by_name(f'b_2019_{i}_{j}')) +\
                            - m.var_by_name(f'b_2018_{i}_{j}') - m.var_by_name(f'b_2019_{i}_{j}')\
                            for i, j in ls_arcs) + \
                       xsum(2 * cap_p_k*m.var_by_name(f'r_{k}') for k in ls_refineries)\
                       )

logging.info("Solve")
# Solve the problem
# m.max_gap = 0.1
# m.threads = -1

status = m.optimize() #m.optimize(max_seconds=100)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
MODEL_NAME = f'model_{dt_string}.lp'
OUTPUT_MODEL_PATH = os.path.join(OUT_SYNTH_DATA_PATH, "models", MODEL_NAME)

logging.info(f"Write model in: {OUTPUT_MODEL_PATH}")
m.write(OUTPUT_MODEL_PATH)

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
    df_sol = pd.DataFrame.from_dict(d_sol, orient='index', columns=['biomass'])
    df_sol.to_csv(os.path.join(OUT_SYNTH_DATA_PATH, f'solution_{dt_string}.csv'))
