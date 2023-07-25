import pandas as pd
import os 
from mip import Model, xsum, minimize, OptimizationStatus, BINARY
import matplotlib.pyplot as plt

SYNTH_DATA_PATH = '002_Optimization/data'
OUT_SYNTH_DATA_PATH = '002_Optimization/output'

d_matrix = pd.read_csv(os.path.join(SYNTH_DATA_PATH, 'Distance_Matrix_Synthetic.csv'), 
                       index_col=0)

df_fc = pd.read_csv(os.path.join(SYNTH_DATA_PATH, 'Biomass_History_Synthetic.csv'))

ls_j = range(len(d_matrix))
year = 2018
cap_b_j = 20000 # Maximum depot capacity
cap_p_j = 100000 # Maximum production capacity
n_refineries = 5 # Number of refineries
n_depots = 25 # Number of depots

# Get the forecasted biomass for year 2018 of all the positions
d_bio_18 = df_fc.loc[:, '2018']
total_fc_18 = d_bio_18.sum()
d_bio_18 = d_bio_18.to_dict()
print("Forecasted biomass for year 2018: ", total_fc_18)


d_bio_19 = df_fc.loc[:, '2019']
total_fc_19 = d_bio_19.sum()
d_bio_19 = d_bio_19.to_dict()
print("Forecasted biomass for year 2019: ", total_fc_19)

# Get the solution for the optimization problem
m = Model(sense=minimize)

# Variables: biomass b_{i, 0}
b_18 = [m.add_var(name=f'b_2018_{i}_{j}', lb=0) for i in range(len(d_matrix)) for j in ls_j]
b_19 = [m.add_var(name=f'b_2019_{i}_{j}', lb=0) for i in range(len(d_matrix)) for j in ls_j]

p_18 = [m.add_var(name=f'p_2018_{i}_{j}', lb=0) for i in range(len(d_matrix)) for j in ls_j]
p_19 = [m.add_var(name=f'p_2019_{i}_{j}', lb=0) for i in range(len(d_matrix)) for j in ls_j]

x = [m.add_var(name=f'x_{j}', var_type=BINARY) for j in ls_j]
r = [m.add_var(name=f'r_{j}', var_type=BINARY) for j in ls_j]

print(f"Variables b_2018 go from {b_18[0].name} to {b_18[-1].name}")
print(f"Variables b_2019 go from {b_19[0].name} to {b_19[-1].name}")
print(f"Variables p_2018 go from {p_18[0].name} to {p_18[-1].name}")
print(f"Variables p_2019 go from {p_19[0].name} to {p_19[-1].name}")

print(f"Variables x go from {x[0].name} to {x[-1].name}")
print(f"Variables r go from {r[0].name} to {r[-1].name}")

# Constraints:
# 1. Can't transport more than generated
for i in range(len(d_matrix)):
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for j in ls_j) <= d_bio_18[i]
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for j in ls_j) <= d_bio_19[i]

m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in range(len(d_matrix)) for j in ls_j)\
    == xsum(m.var_by_name(f'p_2018_{i}_{j}') for i in range(len(d_matrix)) for j in ls_j)

m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in range(len(d_matrix)) for j in ls_j)\
    == xsum(m.var_by_name(f'p_2019_{i}_{j}') for i in range(len(d_matrix)) for j in ls_j)

# 2. Can't transport more than storage limit
for j in ls_j:
    m += xsum(m.var_by_name(f'b_2018_{i}_{j}') for i in range(len(d_matrix))) <= cap_b_j * x[j]
    m += xsum(m.var_by_name(f'b_2019_{i}_{j}') for i in range(len(d_matrix))) <= cap_b_j * x[j]
    m += xsum(m.var_by_name(f'p_2018_{i}_{j}') for i in range(len(d_matrix))) <= cap_p_j * r[j]
    m += xsum(m.var_by_name(f'p_2019_{i}_{j}') for i in range(len(d_matrix))) <= cap_p_j * r[j]

# 3. Where does the biomass go
m += xsum(x[j] for j in ls_j) <= n_depots

# 4. Where does the biomass go
m += xsum(r[j] for j in ls_j) <= n_refineries

# 5. Flux conservation

m += xsum(m.var_by_name(f'p_2018_{i}_{j}') for i in range(len(d_matrix)) for j in ls_j)\
    >= 0.8 * total_fc_18
m += xsum(m.var_by_name(f'p_2019_{i}_{j}') for i in range(len(d_matrix)) for j in ls_j)\
    >= 0.8 * total_fc_19

m.objective = minimize(xsum(d_matrix.iloc[i, j] * (m.var_by_name(f'b_2018_{i}_{j}') + m.var_by_name(f'b_2019_{i}_{j}')) + \
                            d_matrix.iloc[i, j] * (m.var_by_name(f'p_2018_{i}_{j}') + m.var_by_name(f'p_2019_{i}_{j}')) + \
                            (cap_b_j - m.var_by_name(f'b_2018_{i}_{j}')) + (cap_b_j - m.var_by_name(f'b_2019_{i}_{j}')) + \
                            (cap_p_j - m.var_by_name(f'p_2018_{i}_{j}')) + (cap_p_j - m.var_by_name(f'p_2019_{i}_{j}')) \
                                for i in range(len(d_matrix)) for j in ls_j))

print("Solve")
# Solve the problem
# m.max_gap = 0.05
status = m.optimize(max_seconds=300)
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
df_sol.to_csv(os.path.join(OUT_SYNTH_DATA_PATH, 'solution.csv'))
