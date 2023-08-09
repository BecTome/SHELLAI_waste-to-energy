import pandas as pd
import os 
from mip import Model, xsum, minimize, OptimizationStatus, BINARY
from datetime import datetime
import numpy as np
import sys
sys.path.append('.')
from utils.config import LS_INDUSTRY_EXT
from utils.help import get_n_closer

SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output'
FORECAST_FILE = 'Biomass_History_Synthetic.csv' # Forecast File (Synthetic at the beginning) Path
DISTANCE_FILE = 'Distance_Matrix_Synthetic.csv' # Forecast File (Synthetic at the beginning) Path

N = 2418
TRANSPORT_FACTOR_A = .001

df_matrix = pd.read_csv(os.path.join(SYNTH_DATA_PATH, DISTANCE_FILE), 
                       index_col=0)

d_matrix = TRANSPORT_FACTOR_A * df_matrix.values #[:N, :N]

df_fc = pd.read_csv(os.path.join(SYNTH_DATA_PATH, FORECAST_FILE))
df_fc = df_fc#.iloc[:N, :]

df_get_idx = df_fc.loc[:, ['2018', '2019']].mean(axis=1)

d = 70

# ls_depots_dist = []
# for center in LS_INDUSTRY_EXT:
#     df_center = pd.DataFrame(d_matrix[:, center], columns=['distance'])
#     df_center["biomass"] = df_get_idx.values
#     bmu50 = df_center[df_center['distance'] <= d].biomass.sum()
#     if bmu50 >= 15000:
#         ls_depots_dist.append(center)

# ls_refineries = df_matrix.iloc[ls_depots, ls_depots].mean(axis=1).sort_values(ascending=True)[:100].index.tolist()
# ls_biosource = df_matrix.iloc[ls_depots, ls_depots].mean(axis=1).sort_values(ascending=False)[:500].index.tolist()
# ls_depots_dist

arr_n_closer_ind = get_n_closer(df_matrix.iloc[LS_INDUSTRY_EXT, :].values, n=50, uniques=True)
arr_n_closer_ind = np.concatenate([arr_n_closer_ind, np.array(LS_INDUSTRY_EXT)])
arr_n_closer_ind = np.unique(arr_n_closer_ind)
ls_red = list(arr_n_closer_ind)
print("Number of reduced candidates [HS]: ", len(ls_red))

arr_n_closer_dep_ind = get_n_closer(df_matrix.iloc[LS_INDUSTRY_EXT, :].values, n=10, uniques=True)
arr_n_closer_dep_ind = np.concatenate([arr_n_closer_dep_ind, np.array(LS_INDUSTRY_EXT)])
arr_n_closer_dep_ind = np.unique(arr_n_closer_dep_ind)
ls_dep_red = list(arr_n_closer_dep_ind)
print("Number of reduced candidates [DEPOT]: ", len(ls_dep_red))
print("Number of reduced candidates [REF]: ", len(ls_dep_red))

ls_depots = ls_dep_red # range(N)
ls_refineries = ls_dep_red # range(N)
ls_biosource =  ls_red # range(N)
# ls_refineries = ls_depots.copy()
print("Number of depot candidates: ", len(ls_depots))

cap_b_j = 20000 # Maximum depot capacity
cap_p_k = 100000 # Maximum production capacity
n_refineries = 5 # Number of refineries
n_depots = 25 # Number of depots

# Get the forecasted biomass for year 2018 of all the positions
d_bio_18 = df_fc.loc[ls_biosource, '2018']
total_fc_18 = d_bio_18.sum()
d_bio_18 = d_bio_18.to_dict()
print("Forecasted biomass for year 2018: ", total_fc_18)


d_bio_19 = df_fc.loc[ls_biosource, '2019']
total_fc_19 = d_bio_19.sum()
d_bio_19 = d_bio_19.to_dict()
print("Forecasted biomass for year 2019: ", total_fc_19)

# Get the solution for the optimization problem
m = Model(sense=minimize)
m.threads = -1

# Variables: biomass b_{i, 0}
# 1. All values (forecasted biomass, biomass demand-supply, pellet demand-supply) must be
# greater than or equal to zero.
b_18 = [m.add_var(name=f'b_2018_{i}_{j}', lb=0) for i in ls_biosource for j in ls_depots]
print(f"Variables b_2018: {len(b_18)}")

b_19 = [m.add_var(name=f'b_2019_{i}_{j}', lb=0) for i in ls_biosource for j in ls_depots]
print(f"Variables b_2019: {len(b_19)}")

p_18 = [m.add_var(name=f'p_2018_{j}_{k}', lb=0) for k in ls_refineries for j in ls_depots]
print(f"Variables p_2018: {len(p_18)}")

p_19 = [m.add_var(name=f'p_2019_{j}_{k}', lb=0) for k in ls_refineries for j in ls_depots]
print(f"Variables p_2019: {len(p_19)}")

x = [m.add_var(name=f'x_{j}', var_type=BINARY) for j in ls_depots]
print(f"Variables x: {len(x)}")

r = [m.add_var(name=f'r_{k}', var_type=BINARY) for k in ls_refineries]
print(f"Variables r: {len(r)}")

# Constraints:
# 2. The amount of biomass procured for processing from each harvesting site ′𝑖𝑖′ must be less than
# or equal to that site’s forecasted biomass.
for i in ls_biosource:
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for j in ls_depots) <= max(d_bio_18[i], d_bio_19[i])
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for j in ls_depots) <= max(d_bio_18[i], d_bio_19[i])

for j in ls_depots:
    # 3-4. Can't transport more than storage limit
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in ls_biosource) <= cap_b_j * m.var_by_name(f'x_{j}')
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in ls_biosource) <= cap_b_j * m.var_by_name(f'x_{j}')

    # 8. Total amount of biomass entering each preprocessing depot is equal to the total amount of
    # pellets exiting that depot (within tolerance limit of 1e-03
    
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}')  - m.var_by_name(f'p_2018_{j}_{k}') for i in ls_biosource for k in ls_refineries) <=\
          .001 * m.var_by_name(f'x_{j}')
    m += xsum(m.var_by_name(f'p_2018_{j}_{k}') - m.var_by_name(f'b_2018_{i}_{j}') for i in ls_biosource for k in ls_refineries) <=\
          .001 * m.var_by_name(f'x_{j}')
    
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}')  - m.var_by_name(f'p_2019_{j}_{k}') for i in ls_biosource for k in ls_refineries) <=\
          .001 * m.var_by_name(f'x_{j}')
    m += xsum(m.var_by_name(f'p_2019_{j}_{k}') - m.var_by_name(f'b_2019_{i}_{j}') for i in ls_biosource for k in ls_refineries) <=\
          .001 * m.var_by_name(f'x_{j}')


for k in ls_refineries:
    m += xsum(m.var_by_name(f'p_2018_{j}_{k}') for j in ls_depots) <= cap_p_k * m.var_by_name(f'r_{k}')
    m += xsum(m.var_by_name(f'p_2019_{j}_{k}') for j in ls_depots) <= cap_p_k * m.var_by_name(f'r_{k}')
      
# 5. Number of depots should be less than or equal to 25.
m += xsum(m.var_by_name(f'x_{j}') for j in ls_depots) <= n_depots

# 6. Number of refineries should be less than or equal to 5.
m += xsum(m.var_by_name(f'r_{k}') for k in ls_refineries) <= n_refineries

# 7. At least 80% of the total forecasted biomass must be processed by refineries each year
m += xsum(m.var_by_name(f'p_2018_{j}_{k}') for k in ls_refineries for j in ls_depots)\
    >= 0.8 * total_fc_18
m += xsum(m.var_by_name(f'p_2019_{j}_{k}') for k in ls_refineries for j in ls_depots)\
    >= 0.8 * total_fc_19

m.objective = minimize(
                       xsum(d_matrix[i, j] * (m.var_by_name(f'b_2018_{i}_{j}') + m.var_by_name(f'b_2019_{i}_{j}')) +\
                            - m.var_by_name(f'b_2018_{i}_{j}') - m.var_by_name(f'b_2019_{i}_{j}')\
                            for i in ls_biosource for j in ls_depots) + \
                       xsum(d_matrix[j, k] * (m.var_by_name(f'p_2018_{j}_{k}') + m.var_by_name(f'p_2019_{j}_{k}')) +\
                            - m.var_by_name(f'p_2018_{j}_{k}') - m.var_by_name(f'p_2019_{j}_{k}')
                            for j in ls_depots for k in ls_refineries) + \
                       xsum(2 * cap_b_j*m.var_by_name(f'x_{j}') for j in ls_depots) + \
                       xsum(2 * cap_p_k*m.var_by_name(f'r_{k}') for k in ls_refineries)\
                       )

print("Solve")
# Solve the problem
# m.max_gap = 0.1
# m.threads = -1

status = m.optimize(max_seconds=100)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
MODEL_NAME = f'model_{dt_string}.lp'
OUTPUT_MODEL_PATH = os.path.join(OUT_SYNTH_DATA_PATH, "models", MODEL_NAME)

print("Write model in: ", OUTPUT_MODEL_PATH)
m.write(OUTPUT_MODEL_PATH)

print(status)
# Check the status and show the solutions
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(m.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status in [OptimizationStatus.NO_SOLUTION_FOUND, OptimizationStatus.INFEASIBLE]:
    print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    # print('solution:')
    d_sol = {}
    for v in m.vars:
        d_sol.update({v.name: v.x})

    # print("Solution: ", d_sol)
    df_sol = pd.DataFrame.from_dict(d_sol, orient='index', columns=['biomass'])
    df_sol.to_csv(os.path.join(OUT_SYNTH_DATA_PATH, f'solution_{dt_string}.csv'))
