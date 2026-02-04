DESC = '''
Many livestock county populations by farm category are redacted. We apply
the IPF-based method from Burdett et al. to fill these gaps.

by: AA
'''

from ipfn import ipfn
import numpy as np
import pandas as pd
from pdb import set_trace

from stats import Stats, generate_stats

def fill_gap(df, stats_list):
    state = df.name[0]
    livestock = df.name[1]
    subtype = df.name[2]

    stats = Stats()
    stats_list.append(stats)

    stats.subtype = subtype
    stats.state = state
    stats.livestock = livestock

    print('--------------------------------------------------')
    print(f'{state}, {livestock}, {subtype}')

    county_by_farmsize_ind = (df.category=='county_by_farmsize')

    stats.county_by_farmsize = True
    if not county_by_farmsize_ind.any():
        stats.county_by_farmsize = False
        print('No data on county counts by farmsize. Creating a dummy row.')
        name = df.name
        tdf = df[df.category=='county_total'].copy()
        tdf.loc[:, 'category'] = 'county_by_farmsize'
        tdf.size_min = 0
        df = pd.concat([df, tdf], ignore_index=True)
        df.name = name
        return df

    county_by_farmsize_heads_ind = (df.unit=='heads') & \
            (df.category=='county_by_farmsize')
    stats.county_by_farmsize_heads = True
    if not county_by_farmsize_heads_ind.any():
        stats.county_by_farmsize_heads = False
        print('No data on county head counts by farmsize. Creating a dummy row.')
        name = df.name
        tdf = df[(df.unit=='heads') & (df.category=='county_total')].copy()
        tdf.loc[:, 'category'] = 'county_by_farmsize'
        tdf.size_min = 0
        df = pd.concat([df, tdf], ignore_index=True)
        df.name = name
        return df

    if subtype=='all':
        print('Ignored')
        stats.status = 'ignored'
        return df

    heads = df[df.unit=='heads']
    farms = df[df.unit=='operations'].copy()

    # Compute seeds
    redacted = (heads.category=='county_by_farmsize') & (heads.value==-1)
    fixed = (heads.category=='county_by_farmsize') & (heads.value!=-1)

    # Checks
    num_redacted = redacted.sum()

    if num_redacted:
        print(f'{state},{livestock},{subtype}: #redacted: {num_redacted}')
    else:
        print(f'{state},{livestock},{subtype}: No gaps. Skipping IPF ...')
        stats.status = 'no gaps'
        return df

    if (heads[heads.category=='state_by_farmsize'].value==-1).sum():
        raise ValueError(f'{state},{livestock}: Total head count missing for some farm sizes for the state')
    if heads[(heads.category=='county_totals') & (heads.value==-1)].shape[0]:
        raise ValueError(f'{state},{livestock}: missing county totals for the state')

    size_min_state = heads[heads.category=='state_by_farmsize'
            ].size_min.drop_duplicates().tolist()
    size_min_county = heads[heads.category=='county_by_farmsize'
            ].size_min.drop_duplicates().tolist()

    if size_min_state != size_min_county:
        raise ValueError('Size mismatch between state and county farm categories.')

    # Keep track of known values that need to be fixed.
    # Create matrix of fixed values
    fdf = heads[heads.category=='county_by_farmsize']

    mat_fdf = fdf.pivot(index='county_code', columns='size_min', 
            values='value').fillna(0)

    # Note that those values that should be zero will be unaffected.
    mat_fdf.replace(-1, 0, inplace=True)

    row_county_fixed_totals = mat_fdf.sum(axis=1).to_numpy()
    col_state_fixed_totals = mat_fdf.sum().to_numpy()

    row_county_totals = heads[heads.category=='county_total'].value.to_numpy()
    col_state_totals = heads[heads.category=='state_by_farmsize'].value.to_numpy()

    adjusted_row_county_totals = row_county_totals - row_county_fixed_totals
    adjusted_col_state_totals = col_state_totals - col_state_fixed_totals

    if (adjusted_row_county_totals<0).sum() or (adjusted_col_state_totals<0).sum():
        raise ValueError('Some elements in the adjusted totals are negative.')

    # Now fill unknown values with seeds
    ## Seed = Middle of frequency bin * #farms
    farms.loc[farms.size_max==-1, 'size_max'] = farms[
            farms.size_max==-1].size_min * 2
    farms['weight'] = (farms.size_min + farms.size_max)/2 * farms.value


    tdf = heads[redacted].merge(farms[['county_code', 'size_min', 'weight']],
                               on=['county_code', 'size_min'])
    heads.loc[redacted, 'value'] = tdf.weight.values
    
    mat_df = heads[heads.category=='county_by_farmsize'].pivot(
            index='county_code', columns='size_min', 
            values='value').fillna(0)
    farms = farms.drop('weight', axis=1)

    # Subtract fixed values from the matrix
    mat_df = mat_df - mat_fdf

    county_list_from_farmsize = mat_df.index.tolist()
    county_list_from_totals = heads[heads.category=='county_total'
            ].county_code.tolist()
    if county_list_from_farmsize != county_list_from_totals:
        raise ValueErro('County lists extracted from farmsize and totals do not match')

    mat = mat_df.values
    mat_ = mat.copy()   # This is just for debugging

    ## county_cats = mat_df.columns.to_numpy()
    ## state_cats = heads[heads.category=='state_by_farmsize'
    ##                    ].size_min.drop_duplicates().to_numpy()

    ipf = ipfn.ipfn(mat, 
            [adjusted_row_county_totals, adjusted_col_state_totals], [[0],[1]],
                    convergence_rate=1e-8, verbose=2)

    mat, conflag, con = ipf.iteration()

    # Put back fixed values
    mat_df = mat_df + mat_fdf

    new_heads = pd.melt(mat_df.reset_index(), id_vars=['county_code'], 
                        var_name='size_min', value_name='value')

    # Clip
    farms['lb'] = farms.size_min * farms.value
    farms['ub'] = farms.size_max * farms.value
    new_heads = new_heads.merge(farms[['county_code', 'size_min', 'lb', 'ub']],
            on=['county_code', 'size_min'], how='left')
    new_heads.value = new_heads[['value', 'ub']].min(axis=1)
    new_heads.value = new_heads[['value', 'lb']].max(axis=1)
    new_heads = new_heads.drop(['lb', 'ub'], axis=1)

    if new_heads.value.min() < 0:
        raise ValueError('Negative!')

    new_heads[new_heads.value<0] = 0
    new_heads = new_heads.sort_values(['county_code', 'size_min'])
    new_heads['category'] = 'county_by_farmsize'

    county_totals = new_heads[['county_code', 'value']
            ].groupby('county_code').sum().reset_index()
    county_totals['category'] = 'county_total'
    county_totals['size_min'] = -1

    new_heads = pd.concat([new_heads, county_totals])
    heads = heads.merge(new_heads, how='left', 
            on=['county_code', 'size_min', 'category'])
    heads = heads.drop('value_x', axis=1)
    heads = heads.rename(columns={'value_y': 'value'})
    heads.loc[heads.value.isnull(), 'value'] = heads.loc[
            heads.value.isnull()].value_original
    heads.value = heads.value.astype('int')

    # Verify if any fixed value has changed
    heads['test'] = (heads.value!=heads.value_original)
    if heads[(heads.value_original!=-1) &
            (heads.category=='county_by_farmsize')].test.sum():
        raise ValueError('Some fixed values have changed')
    heads = heads.drop('test', axis=1)

    df = pd.concat([heads, farms])

    stats.status = 'ipf'
    stats.iterations = con.shape[0]

    return df
    
def main():
    # Load datasets
    df = pd.read_csv('../results/agcensus_processed1_filled_totals.csv.zip')
    df[df.county.isnull()] = ''

    df = df.sort_values(
            by=['size_min', 'size_max', 'state_code', 'county_code'])

    new_heads_list = []
    stats_list = []
    # Process by state, livestock, subtype
    df = df.groupby(['state_code', 'livestock', 'subtype'],
            ).apply(fill_gap, stats_list, 
                    include_groups=False).reset_index().drop('level_3', axis=1)
    generate_stats(stats_list, out='stats_gaps_county_by_farmsize.csv')

    df_ = df.copy() # just for debugging

    # df = adjust_heads_by_farmsize(df)

    df = df.astype({'value': 'int', 'value_original': 'int'})
    df.to_csv('agcensus_processed2_filled_counts.csv.zip', index=False)

    df = df[df.subtype!='dummy']

    print('-----\nSummary\n-----')
    df1 = df[(df.category=='state_total') & (df.unit=='heads')][
            ['livestock', 'subtype', 'value_original']].groupby(
                    ['livestock', 'subtype']).sum()
    df2 = df[(df.category=='county_by_farmsize') & (df.unit=='heads')][
            ['livestock', 'subtype', 'value_original', 'value']].groupby(
                    ['livestock', 'subtype']).sum()
    df1 = df1.join(df2, lsuffix='-state', rsuffix='-county')
    df3 = df[(df.category=='state_total') & (df.unit=='operations')][
            ['livestock', 'subtype', 'value_original']].groupby(
                    ['livestock', 'subtype']).sum()
    df4 = df[(df.category=='county_by_farmsize') & (df.unit=='operations')][
            ['livestock', 'subtype', 'value']].groupby(
                    ['livestock', 'subtype']).sum()

    if (df4-df3).sum().values[0]:
        raise ValueError('Farm counts should have been the same for state total and county total.')

    df1 = df1.join(df3, how='right').fillna(-1)
    df1 = df1.join(df4, how='right', rsuffix='-ipf').fillna(-1)
    df1 = df1.astype({'value_original-state': 'int', 
                      'value_original-county': 'int', 
                      'value_original': 'int',
                      'value': 'int'})
    df1 = df1.rename(columns={ 'value_original-state': ('heads', 'state'), 'value_original-county': ('heads', 'county'), 'value': ('heads', 'processed'), 'value_original': ('farms', 'state'), 'value-ipf': ('farms', 'processed')})
    df1.columns = pd.MultiIndex.from_tuples(df1.columns)
    df1 = df1.replace(-1, '--')
    print(df1)
    df1.to_latex('table_heads_farms.tex')


if __name__ == '__main__':
    main()
