DESC='''
Tests for ILPs.

AA
'''

import pandas as pd
from pdb import set_trace

import farms_to_cells

def eg1():
    fdf = pd.DataFrame({
            'size_min': [1, 10],
            'size_max': [9, 19],
            'all': [1, 1],
            's1': [1, 0],
            's2': [0, 1]
            })

    hdf = pd.DataFrame({
            'size_min': [1, 10],
            'size_max': [9, 19],
            'all': [6, 15],
            's1': [6, 0],
            's2': [0, 15]
            })

    stats = farms_to_cells.Stats()
    farms = farms_to_cells.generate_farms(hdf, fdf, stats)
    cells = pd.DataFrame({
        'cell': [1, 2, 3, 4, 5, 6],
        'val': [14, 0, 1, 0, 7, 1]})
    assg = farms_to_cells.farms_to_cells(farms, cells, stats)
    print(farms)
    print(assg)
    return

if __name__ == '__main__':
    eg1()


