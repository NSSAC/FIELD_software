import pandas as pd
from pdb import set_trace

FARM_SIZE = [1, 99, 999, 99999, 1000000000] 
FARM_SIZE_NAMES = ['s', 'm', 'l', 'vl']

def main():

    # load features
    print('Loading data ...')
    features = pd.read_parquet('../intermediate_data/risk_features.parquet')
    features = features[['x', 'y', 'county_code', 'state_code',
                         'poultry', 'poultry_W1', 'poultry_W2',
                         'birds1', 'birds2', 'birds3', 'birds4',
                         'birds1_W1', 'birds2_W1', 'birds3_W1', 'birds4_W1',
                         'birds1_W2', 'birds2_W2', 'birds3_W2', 'birds4_W2']]
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    poultry = farms[(farms.livestock=='poultry') & (farms.subtype=='all')]
    df = poultry.merge(features, on=['x', 'y'], suffixes=('_f', ''), how='left')

    print('Processing features ...')
    # Adding within grid cell and around grid cell values
    for q in [1,2,3,4]:
        df[f'birds{q}_W1'] = df[f'birds{q}_W1'] + df[f'birds{q}']
        df[f'birds{q}_W2'] = df[f'birds{q}_W2'] + df[f'birds{q}']

    # This is poultry-specific farm-level risk computation. Hence, this step.
    df = df.rename(columns={'heads': 'poultry_f'})
    df.poultry_W1 = df.poultry_W1 + df.poultry - df.poultry_f
    df.poultry_W2 = df.poultry_W2 + df.poultry - df.poultry_f

    # Farm sizes
    df['size_category'] = pd.cut(df.poultry_f, bins=FARM_SIZE, 
            labels=FARM_SIZE_NAMES, right=False)
    df.size_category = df.size_category.astype(str)

    # Remove unwanted columns
    df = df.drop(['state_code_f', 'county_code_f', 'livestock', 'subtype',
                  'poultry', 'birds1', 'birds2', 'birds3', 'birds4'], 
                 axis=1)
    
    # normalize independent variables
    birds_W1_cols = [f'birds{x}_W1' for x in range(1,5)]
    max_birds_W1 = df[birds_W1_cols].max().max()
    birds_W2_cols = [f'birds{x}_W2' for x in range(1,5)]
    max_birds_W2 = df[birds_W2_cols].max().max()

    for col in ['poultry_W1', 'poultry_W2',
                'birds1_W1', 'birds2_W1', 'birds3_W1', 'birds4_W1',
                'birds1_W2', 'birds2_W2', 'birds3_W2', 'birds4_W2']:
        if 'birds' not in col:
            df[col] = df[col]/df[col].max()
        elif '_W1' in col:
            df[col] = df[col]/max_birds_W1
        elif '_W2' in col:
            df[col] = df[col]/max_birds_W2
    df.to_parquet('poultry_features.parquet')

if __name__ == '__main__':
    main()

