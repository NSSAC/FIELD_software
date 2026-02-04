DESC = '''
Find assignment of farms (from AgCensus) to cells (GLW) for each county.
This is done by using an ILP which comes up with an assignment that tries
to reduce the gap of head counts of AgCensus and GLW for each cell.

By: SSR, ChatGPT, and AA

To run this on interactive, set up the following environment:
    interactive.sh <xx>
    module load gurobi
    export PYTHONPATH=$EBROOTGUROBI/lib/python3.11_utf32

'''

import argparse
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from os import environ
import pandas as pd
from pdb import set_trace
from scipy.stats import pearsonr
import sys
from time import time

TIME_LIMIT = 3600
INFINITY = 1e10

class Stats():
    def __init__(self):
        pass

def generate_farms(heads, farms):
    set_trace()

def farms_to_cells(glw, farms, heads, stats):

    state = heads.head(1).state.values[0]
    county = heads.head(1).county.values[0]
    livestock = heads.head(1).livestock.values[0]

    print(state, county, livestock)

    # cells
    qvec = glw.val.to_numpy()
    m = qvec.shape[0]
    qsum = round(qvec.sum())

    # subtypes
    subtypes = farms.subtype.drop_duplicates().tolist()
    subtypes.remove('all')
    ng = len(subtypes)

    # farms
    farms = farms.sort_values(by='size_min')
    fdf = farms[farms.category=='county_by_farmsize'][
            ['size_min', 'size_max', 'subtype', 'value']].pivot(
                    index=['size_min', 'size_max'], columns='subtype',
                    values='value').reset_index().fillna(0)

    # heads
    heads = heads.sort_values(by='size_min')
    hdf = heads[heads.category=='county_by_farmsize'][
            ['size_min', 'size_max', 'subtype', 'value']].pivot(
                    index=['size_min', 'size_max'], columns='subtype',
                    values='value').reset_index().fillna(0)

    ### extract farms/heads of livestock
    Ni = fdf['all'].values
    Hi = hdf['all'].values
    Ng = {}
    Hg = {}
    for subtype in subtypes:
        Ng[subtype] = fdf[subtype].values
        Hg[subtype] = hdf[subtype].values
    Wmin = fdf.size_min.to_list()
    Wmax = fdf.size_max.to_list()
    ell = fdf.shape[0]
    minheads = np.dot(Wmin, Ni)
    maxheads = np.dot(Wmax, Ni)

    H = heads[heads.category=='county_total'].value.values[0]

    stats.num_farms = sum(Ni)
    stats.glw_cells = m
    stats.num_categories = ell
    stats.glw_sum = qsum

    ## # Decide scenario and normalize if necessary
    ## if Hvec.shape[0] and not (Hvec==-1).any():
    ##     stats.scenario = 1
    ## else:
    ##     print('Condition not satisfied')
    ##     return
    ## elif H >= minheads and H <= maxheads:
    ##     stats.scenario = 2
    ## elif qsum >= minheads and qsum <= maxheads:
    ##     stats.scenario = 3.1
    ##     H = qsum
    ## elif qsum < minheads:
    ##     stats.scenario = 3.2
    ##     H = minheads
    ## elif qsum > maxheads:
    ##     stats.scenario = 3.3
    ##     H = maxheads

    stats.normalize = H/qvec.sum()
    qvec = qvec * stats.normalize

    # MILP starts here
    # Create a new model
    model = gp.Model("Optimal Livestock Distribution")
    
    # Create variables
    y = model.addVars(ell, m, vtype=GRB.INTEGER, name="y")
    x = model.addVars(subtypes, ell, m, vtype=GRB.INTEGER, name="x")
    h = model.addVars(subtypes, ell, m, vtype=GRB.INTEGER, name="h")
    lamb = model.addVar(vtype=GRB.INTEGER, name="lambda")
    
    # Set objective
    model.setObjective(lamb, GRB.MINIMIZE)

    # Add constraints
    # Constraint 1: Each farm should be assigned to one of the cells
    for i in range(ell):
        model.addConstr(
                gp.quicksum(y[i, j] for j in range(m)) == Ni[i], \
                        name=f"farm_assignment_{i}")

    # Constraint 2: Head count constraints for each class and general farm category
    for i in range(ell):
        for j in range(m):
            model.addConstr(gp.quicksum(
                h[g, i, j] for g in subtypes) >= y[i, j] * Wmin[i])
            model.addConstr(gp.quicksum(
                h[g, i, j] for g in subtypes) <= y[i, j] * Wmax[i])

    # Constraint 3: Constraints for a cell
    for g in subtypes:
        for i in range(ell):
            for j in range(m):
                model.addConstr(h[g, i, j] >= x[g, i, j] * Wmin[i])
                model.addConstr(h[g, i, j] <= x[g, i, j] * Wmax[i])

    # Constraint 4: Total head count over all farms of a category
    for g in subtypes:
        for i in range(ell):
            model.addConstr(gp.quicksum(x[g, i, j] for j in range(m)) == 
                    Ng[g][i])
            model.addConstr(gp.quicksum(h[g, i, j] for j in range(m)) == 
                    Hg[g][i])

    # Constraint 5: Gap constraints
    for j in range(m):
        model.addConstr(qvec[j] - gp.quicksum(h[g, i, j] 
            for g in subtypes for i in range(ell)) <= lamb, 
            name=f"gap_upper_{j}")
        model.addConstr(qvec[j] - gp.quicksum(h[g, i, j] 
            for g in subtypes for i in range(ell)) >= -lamb, 
            name=f"gap_lower_{j}")

    ## if stats.scenario == 1:
    ##     for i in range(ell):
    ##         model.addConstr(gp.quicksum(h[i, j] for j in range(m)) == Hvec[i], name=f"total_head_count_{i}")
    ## else:
    ##     model.addConstr(gp.quicksum(h[i, j] for i in range(ell) for j in range(m)) == H, name=f"total_head_count")
    
    # Optimize model
    model.setParam('TimeLimit', TIME_LIMIT) 
    try:
        model.Params.Threads=int(environ['SLURM_NTASKS'])
    except:
        pass

    model.optimize()
    
    stats.optimal_sol = True
    stats.feasible_sol = True
    stats.lamb = -1

    if model.SolCount == 0:
        print('Infeasible/Timed out. Did not find a solution.')
        stats.optimal_sol = False
        stats.feasible_sol = False
        return pd.DataFrame()
    elif model.SolCount > 0:
        stats.feasible_sol = True
        if model.status != GRB.OPTIMAL:
            print('Timed out. No optimal solution found. Reporting best solution.')
            stats.optimal_sol = False
        else:
            stats.optimal_sol = True
            print('Optimal solution found.')
    else:
        raise ValueError('Number of solutions is negative.')

    sol = []
    for i in range(ell):
        for j in range(m):
            h_all = 0
            for subtype in subtypes:
                sol.append((subtype, i, j, h[subtype, i, j].X, 
                    x[subtype, i, j].X))
                h_all += h[subtype, i, j].X
            sol.append(('all', i, j, h_all, y[i, j].X))

    sol_df = pd.DataFrame(sol, 
            columns=['subtype', 'farm_cat', 'cell', 'heads', 'farms'])
    farmsize = pd.DataFrame(data={'size_min': Wmin, 'size_max': Wmax}, 
            index=np.arange(len(Wmin)))
    sol_df['size_min'] = sol_df.farm_cat.map(farmsize.size_min)
    sol_df['size_max'] = sol_df.farm_cat.map(farmsize.size_max)

    stats.lamb = lamb.x
    model.dispose()

    dfl = []
    for subtype,size_min,size_max in sol_df[
            ['subtype', 'size_min', 'size_max']].drop_duplicates().values:
        tdf = sol_df[(sol_df.subtype==subtype) &
                (sol_df.size_min==size_min) &
                (sol_df.size_max==size_max)]

        glw['size_min'] = tdf.size_min.values
        glw['size_max'] = tdf.size_max.values
        glw['livestock'] = livestock
        glw['subtype'] = subtype
        glw['heads'] = tdf.heads.values
        glw['farms'] = tdf.farms.values

        dfl.append(glw.copy())
    df = pd.concat(dfl, ignore_index=True)
    df.loc[df.size_max==INFINITY, 'size_max'] = -1

    df = df[df.farms>0]
    corr_vec = df[df.subtype=='all'][['x', 'y', 'heads']].copy()
    corr_vec = corr_vec.groupby(['x', 'y']).sum()
    corr_vec = corr_vec.merge(glw[['x', 'y', 'val']], on=['x', 'y'])

    if corr_vec.shape[0] == 1:
        stats.corr = 1
        stats.pval = -1
    else:
        pr = pearsonr(corr_vec.val, corr_vec.heads)
        stats.corr = pr.statistic
        stats.pval = pr.pvalue

    df = df.drop('val', axis=1)
    df = df.astype({'farms': 'int', 'heads': 'int'})

    return df

if __name__ == '__main__':
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--state', required=True, type=int)
    parser.add_argument('-c', '--county', required=True, type=int)
    parser.add_argument('-t', '--type', required=True)
    args = parser.parse_args()

    start = time()

    # Load datasets
    glw = pd.read_csv('../../data/glw/glw_livestock.csv.zip')
    df = pd.read_csv('../results/agcensus_processed2_filled_counts.csv.zip')
    df = df.astype({'value': 'int'})

    glw = glw.astype({'state_code': 'int', 'county_code': 'int'})
    df = df.astype({'state_code': 'int', 'county_code': 'int'})

    # Assign per county,livestock
    stats = Stats()
    stats.livestock = args.type
    stats.state = args.state
    stats.county = args.county
    loc_glw = glw[(glw.state_code==args.state) &
            (glw.county_code==args.county) & 
            (glw.livestock==args.type)].copy()
    loc_df = df[(df.county_code==args.county) & 
            (df.state_code==args.state) &
            (df.livestock==args.type)]

    heads = loc_df[loc_df.unit=='heads']
    farms = loc_df[loc_df.unit=='operations']

    # No fields to be assigned, no problem
    if not farms.shape[0]: 
        print(f'No farms found.')
        print(f'Total time: {state},{county},{livestock},{(time()-start)//60}')
        sys.exit()

    # No cells to assign to, assign uniformly
    stats.uniform = False
    if not loc_glw.val.sum() and farms.shape[0]: 
        stats.uniform = True
        loc_glw = glw[(glw.state_code==state) & (glw.county_code==county)][
                ['x', 'y', 'state_code', 'county_code']].drop_duplicates()
        loc_glw['livestock'] = args.type
        loc_glw['val'] = 1

    proc_start = time()
    farm_assignments = generate_farms(heads, farms)
    assg = farms_to_cells(loc_glw, farms, heads, stats)
    stats.time = time() - proc_start

    sdict = {x: [getattr(stats,x)] for x in dir(stats) if x[0:2] != '__'}
    pd.DataFrame.from_records(sdict).to_csv(
            f'stats_farms_to_cells_{state}_{county}_{livestock}.csv.zip', 
            index=False)

    try:
        assg.to_csv(
                f'farms_to_cells_{state}_{county}_{livestock}.csv.zip', 
                index=False)
    except ValueError:
        raise ValueError('No farms to be assigned.')

    print(f'Total time: {state},{county},{livestock},{(time()-start)//60}')



    print('Done')
