import numpy as np
def get_n_closer(d_matrix, n=2, uniques=True):
    '''
    Given a distance matrix, return the n closer elements for each row (indices).
    '''
    idx_matrix = d_matrix.argsort(axis=1)[:, 1:n+1]

    if uniques:
        return np.unique(idx_matrix.reshape(-1))
    else:
        return idx_matrix

if __name__ == "__main__":
    import os
    import sys
    sys.path.append('.')
    import numpy as np
    import pandas as pd
    from utils.config import LS_INDUSTRY_EXT
    d_matrix = pd.read_csv('002_Optimization/data/Distance_Matrix_Synthetic.csv', index_col=0)
    d_matrix = d_matrix.iloc[LS_INDUSTRY_EXT, :].values
    print(d_matrix.shape)
    print(len(LS_INDUSTRY_EXT))
    print(get_n_closer(d_matrix, n=30, uniques=True).shape)