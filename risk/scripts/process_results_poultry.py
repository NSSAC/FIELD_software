DESC='''
Analyzing the model space of parameters based on score.

By: AA
'''

import pandas as pd
from pdb import set_trace

PARLIST = ['cattle', 'cattle_n', 'poultry', 'birds', 'a_ksst']
INSTANCE_LIST = PARLIST + ['state_code', 'quarter']
SCORE_THRESHOLD = 0.75

def state_score(df):
    odf = pd.DataFrame({'kst': [df.kst.sum()], 'ksst': [df.ksst.sum()]})
    odf.loc[:, 'score'] = (odf.kst / odf.ksst).values
    return odf

df = pd.read_parquet('../results/dairy_risk.parquet')

# compute score for each state
score = df.groupby(PARLIST + ['state_code']).apply(
        state_score).rename(columns={None: 'score'}).reset_index()


# filter instances that have low scores for all states
filtered_scores = score.groupby(PARLIST, as_index=False)['score'].max()
filtered_scores = filtered_scores[filtered_scores.score>=SCORE_THRESHOLD]
score = score.merge(filtered_scores[PARLIST], how='left', indicator=True, 
        suffixes=('','_y'))
score = score[score._merge=='both']
score = score.drop(['_merge', 'level_6'], axis=1)

# prepare vectors for clustering (using the state scores
# fv(\underline(\alpha))
odf = score.pivot(index=PARLIST, columns='state_code', 
                  values=['score', 'kst'])
odf.columns = [x+'_'+str(y) for x,y in odf.columns]
ksst = pd.DataFrame(score.groupby(PARLIST)['ksst'].sum())
ksst.columns = ['ksst']
odf = odf.join(ksst)
odf.to_parquet('dairy_risk_processed.parquet')

