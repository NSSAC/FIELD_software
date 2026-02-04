DESC = '''
Compute risk for milk poultry and score w.r.t. incidence reports.

AA
'''

import argparse
import click
from datetime import datetime
from dateutil.relativedelta import relativedelta
from hmmlearn import hmm
from itertools import product
import numpy as np
import pandas as pd
from pdb import set_trace
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, ParameterGrid
from skopt import gp_minimize
from skopt.space import Real, Integer

# from parlist import POULTRY_PARLIST as PARLIST
from kbdata import loader
import utils

COMMERCIAL = True
COMMERCIAL_THRESHOLD = 100 
BIRDS_NEIGHBOR_COUNTY_SCALE = 1

SPILLOVER_MODEL = '../intermediate_data/baseline_spillover_model.csv'
CONDITIONAL_RISK_MODEL = '../intermediate_data/baseline_conditional_risk.pkl'
BIRD_PREVALENCE = '../../data/birds_prevalence/bird_h5_prevalence.parquet'
COUNTY_NEIGHBORS = 'county_neighbors.parquet'

MAX_FORWARD_PRED = 6
NUMSIMS = 1000

# Custom callback to display progress every 50 iterations
class CustomVerboseCallback:
    def __init__(self, display_interval=50):
        self.display_interval = display_interval

    def __call__(self, res):
        iteration = len(res.x_iters)
        if iteration % self.display_interval == 0:
            print(f"Iteration {iteration}: Best Value: {res.fun}, Best Parameters: {res.x}")

def load_features():
    features = pd.read_parquet('livestock_birds_features.parquet')

    pdf = utils.h5n1_poultry()
    ddf = utils.h5n1_dairy()
    ddf['type'] = 'milk'

    events = pd.concat([
        pdf[['start_date', 'county_code', 'year', 'type']], 
        ddf[['start_date', 'county_code', 'year', 'type']]
        ])

    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    preps.loc[preps.type=='all', 'type'] = 'all poultry'

    dreps = pd.read_csv('../results/dairy_outbreaks.csv').fillna(-1)
    dreps['type'] = 'milk'
    dreps = utils.fit_central_valley(dreps)

    outbreaks = pd.concat([preps, dreps])
    outbreaks.year = outbreaks.start_date.str[0:4].astype('int')
    outbreaks.month = outbreaks.start_date.str[5:7].astype('int')

    h5_prevalence = pd.read_parquet(BIRD_PREVALENCE)

    features = features.reset_index(drop=True)
    return features, events, outbreaks, h5_prevalence

def process_features(farms):
    dist = 2
    # Choosing the weighted sum of population based on distance
    retain_cols = []
    for col in farms.columns:
        if '_W' not in col:
            retain_cols.append(col)
        elif f'_W{dist}' in col:
            retain_cols.append(col)

    rename_cols = {x: x[:-1] for x in retain_cols if '_W' in x} 
    farms = farms[retain_cols].rename(columns=rename_cols)

    return farms

def baseline_spillover_model():
    # try loading model
    outfile = 'baseline_spillover_model.csv'
    try:
        model = pd.read_csv(outfile).to_dict(orient='list')
    except:
        model = {'subtype': [], 'birds_W_lb': [], 'birds_W_ub': [], 
                 'fun_lb': [], 'fun_ub': []}

    # load data
    farms_all, events_all, outbreaks_all, h5_prevalence = load_features()

    outbreaks_all = outbreaks_all[(outbreaks_all.delta==30)]
    outbreaks_all = outbreaks_all.rename(columns={'event0': 'outbreak'})

    for type in ['milk', 'turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 
                 'ckn-pullets']:
        print('--------------------------------------------------')
        print(type)
        print('--------------------------------------------------')

        if type in model['subtype']:
            print('Skipping as it is already present')
            continue

        farms = farms_all[farms_all.subtype==type]
        outbreaks = outbreaks_all[outbreaks_all.type==type]

        if type == 'milk':
            # There was one spillover event in 2023. After that, it was Texas B313.
            num_spillovers = 2
        else:
            outbreaks = outbreaks[(outbreaks.year==2022)]
            num_spillovers = outbreaks.outbreak.max()

        # process features
        farms = process_features(farms)

        # optimizer for risk computation
        # Use the custom callback
        callback = CustomVerboseCallback(display_interval=25)
        
        # Define the search space
        ## if type == 'milk':
        ##     search_space = [
        ##             Real(0.0000001, 100, prior='log-uniform', name="birds_W")
        ##             ]
        ## else:
        search_space = [
                Real(0.000001, 1, prior='log-uniform', name="birds_W")
                ]
        bounds = [(0,1)]

        #parameter ranges
        # Run Bayesian Optimization
        lb = gp_minimize(
                func=lambda params: wrapper_evaluation(*params, farms, 
                                                       h5_prevalence, num_spillovers, 1),
                dimensions=search_space,
                n_calls=100,
                n_jobs=-1,
                callback=[callback],
                random_state=123
                )
        if type == 'milk':
            ub_fraction = 5
        else:
            ub_fraction = 1.1
        ub = gp_minimize(
                func=lambda params: wrapper_evaluation(*params, farms, 
                                                       h5_prevalence, num_spillovers, 
                                                       ub_fraction),
                dimensions=search_space,
                n_calls=100,
                n_jobs=-1,
                callback=[callback],
                random_state=123
                )
        model['subtype'].append(type)
        model['birds_W_lb'].append(lb.x[0])
        model['fun_lb'].append(lb.fun)
        model['birds_W_ub'].append(ub.x[0])
        model['fun_ub'].append(ub.fun)

    pd.DataFrame(model).to_csv(outfile, index=False)
    return

def spillover_risk(farms, birds_W, month, bird_h5_prevalence=pd.DataFrame()):
    
    #subtype = farms.subtype.head(1).values[0]
    # risk
    if bird_h5_prevalence.shape[0]:
        farms = farms.merge(bird_h5_prevalence[bird_h5_prevalence.month==month], 
                            on=['x', 'y'], how='left').fillna(0)
        return 1-np.exp(-(birds_W * farms[f'birds{month}_W'] * farms['value']))
    else:
        return 1-np.exp(-(birds_W * farms[f'birds{month}_W'])) # * farms[f'binf{q}']))
    # Note that this risk probability is conditioned upon the event that the farm
    # is not infected in the previous time steps

def wrapper_evaluation(birds_W, farms, h5_prevalence, num_spillovers, spillover_scale):
    for m in range(1,13):
        tdf = spillover_risk(farms, birds_W, m, bird_h5_prevalence=h5_prevalence)
        farms[f'risk_prob{m}'] = tdf.values
    # find probability that each farm gets infected
    farms['inf'] = (1-farms.risk_prob1)
    for t in range(2,13):
        farms.inf = farms.inf * (1-farms[f'risk_prob{t}'])
    farms.inf = 1 - farms.inf

    gt_farms = num_spillovers * spillover_scale

    return abs(gt_farms-farms.inf.sum())

def baseline_conditional_risk_model():
    pdf = utils.h5n1_poultry()

    ddf = utils.h5n1_dairy()
    ddf['type'] = 'milk'
    ddf = utils.fit_central_valley(ddf)

    pdf = pdf[['county_code', 'year', 'month', 'type']]
    ddf = ddf[['county_code', 'year', 'month', 'type']]

    reps = pd.concat([pdf, ddf]).drop_duplicates().reset_index(drop=True)
    reps['time'] = reps.year.astype('str') + '-' + reps.month.astype('str').str.zfill(2)

    risk_model = {}
    score = {}
    params = {}
    for t in reps.type.drop_duplicates():
        if t in ['waterfowl', 'ckn-pullets']:
            continue
        if t == 'milk':
            months = ['2023-01', '2024-12']
        else:
            months = ['2022-01', '2022-12']

        risk_model[t], score[t], params[t] = _type_baseline_conditional_risk(
                reps[reps.type==t].copy(), months)
    with open(CONDITIONAL_RISK_MODEL, 'wb') as f:
        pickle.dump((risk_model, score, params), f)

def _type_baseline_conditional_risk(df, dates):
    type = df.type.head(1).values[0]
    df = df[(df.time >= dates[0]) & (df.time <= dates[1])]
    print('##########\n', type)
    df["present"] = 1  # Assign 1 for presence

    # Generate all months between dates[0] and dates[1], excluding the first month
    months = pd.date_range(start=dates[0], end=dates[1], freq="MS").strftime('%Y-%m').tolist()

    sequences = df.pivot_table(index="county_code", columns="time", 
                              values="present", aggfunc="max").fillna(0)
    # Reindex to ensure all months are present as columns, fill missing with 0
    sequences = sequences.reindex(columns=months, fill_value=0).astype(int)
    print(f'Samples: {sequences.shape}')

    # Ensure all months included
    months = pd.date_range(start=dates[0], end=dates[1], freq="MS"
                           ).strftime('%Y-%m').tolist()
    sequences = sequences.reindex(columns=months, 
                                          fill_value=0).astype(int).values
    train_seqs, val_seqs = train_test_split(sequences, test_size=0.2, random_state=42)

    # Convert sequences into a suitable format for HMM
    def prepare_data(seqs):
        lengths = [len(seq) for seq in seqs]
        X = np.concatenate(seqs).reshape(-1, 1)  # Flatten and reshape
        return X, lengths

    X_train, lengths_train = prepare_data(train_seqs)
    X_val, lengths_val = prepare_data(val_seqs)

    # Define hyperparameter grid
    param_grid = {
        "n_components": [2, 3, 4, 5],  # Number of hidden states
        "n_iter": [500],
        "random_state": [1, 2, 3, 4, 5]
    }

    best_score = np.inf
    best_params = None
    best_model = None

    # Perform grid search
    for params in ParameterGrid(param_grid):
        model = hmm.CategoricalHMM(**params, n_features=2)  # 4 categorical states
        model.fit(X_train, lengths_train)  # Train only on training sequences

        # Validate on unseen sequences
        # earlier score = model.score(X_val, lengths_val)
        # Average log-likelihood per observation with penalty for model complexity
        score = -2*model.score(X_val, lengths_val) / len(lengths_val) + \
            2*params['n_components']    
        # score = -model.score(X_val, lengths_val) / len(lengths_val)    

        if score < best_score:
            best_score = score
            print(best_score)
            best_params = params
            best_model = model
    print("Best Parameters:", best_params)
    print("Best AIC Score on Validation Set:", best_score)
    print("Num. of validation points:", len(lengths_val))

    return best_model, best_params, best_score

def risk_score(period, bird_h5_prevalence=False,
             spillover_risk_model=False,
             conditional_risk_model=False,
             adaptive=False, evaluate=False):

    period_list_str = pd.date_range(start=period[0], end=period[1], freq='MS'
                                ).strftime('%Y-%m').tolist()

    # load data
    farms, events, outbreaks, h5_prevalence = load_features()
    farms = process_features(farms)
    ##subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'ckn-pullets', 'milk']
    subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'milk']
    events = events[events.type.isin(subtypes)]
    events['ym'] = events.start_date.astype('str').str[:7]

    # spillover risk
    county_risk_list = []
    if spillover_risk_model:
        print('Spillover risk probabilities')
        spillover_model = pd.read_csv(SPILLOVER_MODEL)
        farms = farms[farms.subtype.isin(subtypes)]

        fsdf_list = []
        for subtype in subtypes:
            fsdf = farms[farms.subtype==subtype].copy()
            birds_W = spillover_model[spillover_model.subtype==subtype
                                      ].birds_W_lb.values[0]
            # since we are doing forward predictions, we will add more time steps
            # here.
            date_obj = datetime.strptime(period_list_str[-1], '%Y-%m')
            next_months = [(date_obj + relativedelta(months=i)).strftime("%Y-%m") 
                           for i in range(1, MAX_FORWARD_PRED+1)]
            for per in period_list_str + next_months:
                month = int(per[-2:])
                xx = spillover_risk(fsdf, birds_W, month, 
                                           bird_h5_prevalence=h5_prevalence)
                fsdf[per] = xx.values
            fsdf_list.append(fsdf)
                
        farms_risk = pd.concat(fsdf_list)

        # county spillover risk by aggregating over all farms
        tdf = farms_risk[['county_code', 'subtype'] + 
                         [x for x in farms_risk.columns if x[0:2]=='20']]
        farms_risk.to_parquet('farms_spillover_risk.parquet', index=False)
        spcdf = utils.combine_probs(tdf, id_vars=['county_code', 'subtype'])

    print('Total risk including conditional risk ...')

    # load model
    with open(CONDITIONAL_RISK_MODEL, 'rb') as f:
        crm = pickle.load(f)[0]
    for subtype in subtypes: #['milk']:# + subtypes:
        if subtype == 'ckn-pullets':
            continue
        tdf = _type_total_risk(events[events.type==subtype].copy(), 
                               period_list_str, 
                               spcdf[spcdf.subtype==subtype], crm[subtype],
                               cr_flag=conditional_risk_model)
        county_risk_list.append(tdf)

    # combine different risks
    risk = pd.concat(county_risk_list)

    return risk
    
def _type_total_risk(events, dates, spm, crm, numsims=NUMSIMS, cr_flag=True):

    subtype = spm.subtype.head(1).values[0]
    print(f'Total risk for {subtype} ...')

    # get all counties
    counties = loader.load('usa_counties', contiguous_us=True)
    spdf = spm.merge(counties.county_code, on='county_code', how='right')
    spdf.loc[spdf.subtype.isnull(), 'subtype'] = subtype
    spdf = spdf.fillna(0)

    # prepare data for conditional risk
    events["present"] = 1  # Assign 1 for presence
    edf = events.pivot_table(index="county_code", columns="ym", 
                             values="present", aggfunc="max").fillna(0)

    tp = pd.date_range(start='2022-01', end=dates[-1], freq='MS'
                                 ).strftime('%Y-%m').tolist()
    edf = edf.reindex(columns=tp, fill_value=0).astype(int)
    edf = edf.reset_index().merge(counties.county_code, 
                                  on='county_code',
                                  how='right').fillna(0)

    # adjust for central valley
    if subtype == 'milk':
        edf = utils.fit_central_valley(edf)
        edf = edf.groupby('county_code').max().reset_index()
        spdf = utils.fit_central_valley(spdf)
        spdf = utils.combine_probs(spdf, id_vars=
                              ['county_code', 'subtype'])

    # get past period for hmm
    pp = pd.date_range(start='2022-01', end=dates[0], freq='MS'
                       ).strftime('%Y-%m').tolist()
    pp = pp[:-1]

    # 1-3 months forecast for each time step
    risk_per_date = []
    for t in dates:
        print('\t', t)
        # get next MAX_FORWARD_PRED time steps
        date_obj = datetime.strptime(t, "%Y-%m")
        next_months = [(date_obj + relativedelta(months=i)).strftime("%Y-%m") 
                       for i in range(MAX_FORWARD_PRED)]

        # generate sim sequences for spillover risk 

        if cr_flag:
            spp = spdf[next_months].values
            sp_sims = np.random.binomial(n=1, p=spp[:, None, :], 
                                         size=(spp.shape[0], numsims, spp.shape[1]))

            # get hmm state probabilities for the current time period
            pseqs = edf[pp].values.astype(int)
            tseqs = edf[pp+[t]].values.astype(int)
            
            # generate future sequences based on past sequences
            crisk_list = []
            for i in range(pseqs.shape[0]):
                pseq = pseqs[i,:]
                ## tseq = tseqs[i,:]
                flag = False
                ## if tseq[-1] == 1 and tseq[-2] == 1:
                ##     flag = True
                ##     print(i)
                last_posterior = crm.predict_proba(pseq.reshape(-1,1))[-1]
                if not pseq.sum():  # no need to do hmm
                    crisk_list.append(np.zeros((numsims, MAX_FORWARD_PRED), dtype=int))
                    continue

                fseqs_list = []
                for s in range(numsims):
                    initial_state = np.random.choice(crm.n_components, p=last_posterior)
                    __states, observations = crm.sample(n_samples=MAX_FORWARD_PRED, currstate=initial_state)
                    fseqs_list.append(observations)
                    
                crisk_list.append(np.array(fseqs_list))
            cr_sims = np.stack(crisk_list, axis=0)

            # combine sp and cr
            rmat = np.mean(np.logical_or(sp_sims, cr_sims), axis=1)
            rdf = pd.DataFrame(rmat, columns=range(MAX_FORWARD_PRED))
            rdf[['time','subtype']] = (t, subtype)
            rdf['county_code'] = edf.county_code
            risk_per_date.append(rdf)

            # append date to past period
            pp.append(t)
        else:
            tdf = spdf[['county_code', 'subtype'] + next_months].copy()
            tdf.columns = ['county_code', 'subtype'] + list(range(MAX_FORWARD_PRED))
            tdf['time'] = t
            risk_per_date.append(tdf)
        risk = pd.concat(risk_per_date)
    return risk
    
if __name__ == '__main__':
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--mode', 
                        help='spillover_model/conditional_risk_model/risk/recall_eval')
    parser.add_argument('--period_start', 
                        help='Only for "risk_map" mode, e.g., 2024-05')
    parser.add_argument('--period_end', 
                        help='Only for "risk_map" mode, e.g., 2024-05')
    parser.add_argument('--spillover_risk_model', action='store_true')
    parser.add_argument('--conditional_risk_model', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--bird_h5_prevalence', action='store_true')
    args = parser.parse_args()

    if args.mode == 'spillover_model':
        baseline_spillover_model()
    elif args.mode == 'conditional_risk_model':
        baseline_conditional_risk_model()
    elif args.mode == 'risk_score':
        risk_scores = risk_score([args.period_start, args.period_end], 
                            spillover_risk_model=args.spillover_risk_model,
                            conditional_risk_model=args.conditional_risk_model,
                            adaptive=args.adaptive,
                            bird_h5_prevalence=args.bird_h5_prevalence)
        risk_scores.to_csv(
                f'risk_scores_sp{int(args.spillover_risk_model)}_cr{int(args.conditional_risk_model)}.csv', 
                index=False)

