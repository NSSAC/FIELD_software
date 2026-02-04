DESC='''
Check for correlation between agriculture-linked fields. Note that we are 
only analyzing values restricted to agriculture.
'''

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import seaborn as sns
from scipy.stats import chi2_contingency

import plot
import utils_pums

FORMAT="%(levelname)s:%(funcName)s:%(message)s"
pd.options.display.float_format = '{:.10g}'.format

def corr(df, col1, col2):
    if col1 == col2:
        return 0, 1
    contingency_table = pd.crosstab(
            df[col1], 
            df[col2], 
            margins = True)
    observed = contingency_table.to_numpy()[0:-1,0:-1]
    chi2, p, dof, __ = chi2_contingency(observed, correction=False)

    # Calculate Cramer's V
    N = np.sum(observed)
    minimum_dimension = min(observed.shape)-1

    cramer = np.sqrt((chi2/N) / minimum_dimension)

    return p, cramer 

def main():

    logging.basicConfig(level=logging.INFO,format=FORMAT)

    person = utils_pums.select_ag(columns = 'all')


    fields = ['indp', 'naicsp', 'occp', 'socp', 'fod1p', 'fod2p']
    pval = np.zeros([len(fields),len(fields)])
    cramer = np.zeros([len(fields),len(fields)])
    for i,v1 in enumerate(fields): 
        for j,v2 in enumerate(fields):
            p, c = corr(person, v1, v2)
            pval[i,j] = p
            cramer[i,j] = c
    pval_df = pd.DataFrame(p,
            index = fields,
            columns = fields)
    cramer_df = pd.DataFrame(cramer,
            index = fields,
            columns = fields)
    ## print(pval_df.style.to_latex())
    ## print(cramer_df.style.to_latex())

    sns.heatmap(cramer_df, annot=True)
    plt.savefig('pums_cramer.pdf', bbox_inches='tight')

if __name__== "__main__":
    main()
