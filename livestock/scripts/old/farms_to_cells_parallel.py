DESC = '''
Find assignment of farms (from AgCensus) to cells (GLW) for each county.
This is done by using an ILP which comes up with an assignment that tries
to reduce the gap of head counts of AgCensus and GLW for each cell.

By: SSR, ChatGPT, and AA

To run this, set up the following environment:
    module load anaconda/2023.07-py3.11
    module load gurobi

Timing:
-------
interactive.sh 30
procs,max_workers_per_node,time (mins) 
5,40,2
30,40,8
30,10,8
30,1,7
'''

import argparse
import numpy as np
import pandas as pd
from pdb import set_trace
from time import time

PARSL = True

if PARSL:
    import parsl
    #from parsl.monitoring.monitoring import MonitoringHub
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.app.app import python_app, join_app

    config = Config(
       executors=[
           HighThroughputExecutor(
               label="local_htex",
               cores_per_worker=1,
               max_workers_per_node=2,
               address='127.0.0.1',
           )
       ],
       ## monitoring=MonitoringHub(
       ##     #hub_address=address_by_hostname(),
       ##     hub_address='127.0.0.1',
       ##     hub_port=55055,
       ##     monitoring_debug=False,
       ##     resource_monitoring_interval=5,
       ## ),
       strategy='none',
       run_dir='farms_to_cells'
    )
    parsl.load(config)

def wrapper(farms, cells):
    print('In:', farms, cells)
    assgn, lambda_val, corr, pval = ilp(farms, cells)
    print('Out:', assgn, lambda_val, corr, pval, '\n')

def examples():
    wrapper([1,2,3,4],[1,2,3,4])
    wrapper([.1,.2,.3,.4],[1,2,3,4])
    wrapper([1,2,3,4],[1,2])

@python_app
def process_county(glw, agcensus):
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    statefp = glw.statefp.head(1).values[0]
    countyfp = glw.countyfp.head(1).values[0]
    livestock = glw.livestock.head(1).values[0]
    print(statefp, countyfp, livestock)

    res = {}

    if glw.shape[0] == 0 and agcensus.shape[0] == 0:
        res['cell'] = [-1]
        res['farm_weight'] = [-1]
        stats = {}
        print('Not enough data. Skipping ...')
    elif agcensus.shape[0] == 0:
        res['farm_weight'] = [-1]
        res['cell'] = [-1]
        stats = {}
        print('Not enough data. Skipping ...')
    elif glw.shape[0] == 0:
        farms = np.repeat(
                agcensus.avg_size.to_numpy(), agcensus.num_farm.to_numpy())
        res['farm_weight'] = farms
        res['cell'] = [-1] * len(farms)
        stats = {}
        print('Not enough data. Skipping ...')
    else:
        farms = np.repeat(
                agcensus.avg_size.to_numpy(), agcensus.num_farm.to_numpy())
        cells = glw.val.to_list()
        print('Computing assignment ...')

        ##########################################
        # ILP starts here
        ##########################################
        # Normalize
        farms = np.array(farms)
        farms = farms/np.sum(farms)
        cells = np.array(cells)
        cells = cells/np.sum(cells)

        # Initialize the model
        model = gp.Model('farm_allocation')
        
        # Create variables
        # x[i, j] is 1 if farm i is assigned to cell j, 0 otherwise
        x = model.addVars(len(farms), len(cells), vtype=GRB.BINARY, name="x")
        
        # Variable for the maximum allowed gap λ
        lambda_val = model.addVar(vtype=GRB.CONTINUOUS, name="lambda", lb=0)
        
        # Objective: Minimize λ
        model.setObjective(lambda_val, GRB.MINIMIZE)
        
        # Constraint: Each farm must be assigned to exactly one cell
        for i in range(len(farms)):
            model.addConstr(gp.quicksum(x[i, j] for j in range(len(cells))) == 1, f"farm_{i}_assignment")
        
        # Constraint: The gap in each cell should be at most λ
        for j in range(len(cells)):
            total_weight_in_cell = gp.quicksum(farms[i] * x[i, j] for i in range(len(farms)))
            model.addConstr(total_weight_in_cell - cells[j] <= lambda_val, f"pos_gap_{j}")
            model.addConstr(cells[j] - total_weight_in_cell <= lambda_val, f"neg_gap_{j}")
        
        # Optimize the model
        model.setParam('OutputFlag', 0)
        model.optimize()
        
        # Print the results
        assignment = [None for i in range(len(farms))]
        if model.status == GRB.OPTIMAL:
            for j in range(len(cells)):
                for i in range(len(farms)):
                    if x[i, j].x > 0.5:
                        assignment[i] = j
            lamb = lambda_val.x
            model.dispose()
            farm_weight_per_cell = np.zeros(len(cells))
            for i,w in enumerate(farms):
                farm_weight_per_cell[assignment[i]] += farms[i]

            pr = pearsonr(farm_weight_per_cell, cells)
            corr = pr.statistic
            pval = pr.pvalue
        else:
            print('No optimal solution found')
            model.dispose()
            assignment = -1
            lamb = -1
            corr = -1
            pval = -1

        res['farm_weight'] = farms
        cell_coords = glw[['x','y']].values
        res['cell'] = [cell_coords[assignment[i]] for i in range(len(farms))]
        stats = {'lambda': lamb, 'corr': corr, 'pval': pval}

    df = pd.DataFrame(res)
    coords = df.cell.apply(pd.Series)
    try:
        df['x'] = coords[0]
        df['y'] = coords[1]
    except KeyError:
        df['x'] = -1
        df['y'] = -1
    df['livestock'] = livestock
    df['countyfp'] = countyfp
    df['statefp'] = statefp
    df = df.drop('cell', axis=1)

    stats['livestock'] = livestock
    stats['countyfp'] = countyfp
    stats['statefp'] = statefp

    return statefp, countyfp, livestock, df, stats

def farms_to_cells():

    start = time()
    # Load datasets
    glw = pd.read_csv('../data/glw_sans_geom_processed.csv.zip')
    agcensus = pd.read_csv('../data/agcensus_processed.csv.zip')

    # Generate farm entities
    res = []
    i=0
    thresh=30
    for livestock in agcensus.commodity.drop_duplicates().to_list():
        for state in glw.statefp.drop_duplicates().to_list():
            for county in glw.countyfp.drop_duplicates().to_list():
                res.append(process_county(
                    glw[(glw.statefp==state) &
                        (glw.countyfp==county) & 
                        (glw.livestock==livestock)], 
                    agcensus[(agcensus.county_fips==county) & 
                        (agcensus.state_fips==state) &
                        (agcensus.commodity==livestock)])
                    )
                if i>thresh:
                    break
                i+=1
            if i>thresh:
                break
        if i>thresh:
            break

    if PARSL:
        res_parsl = []
        for rr in res:
            res_parsl.append(rr.result())
        res = res_parsl

    dfl = []
    stats = []
    for rc in res:
        dfl.append(rc[3])
        stats.append(rc[4])

    pd.concat(dfl).to_csv('farm_to_cell_assignment.csv.zip', index=False)
    pd.DataFrame.from_records(stats).to_csv(
            'stats_farm_to_cell_assignment.csv.zip', index=False)
    print(f'Total time: {(time()-start)//60}')

if __name__ == '__main__':
    # examples()
    farms_to_cells()
