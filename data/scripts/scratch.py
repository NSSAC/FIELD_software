        ## # trigger event: text should be a state
        ## for par in inc:
        ##     if par.get_text().lower() in states.state.to_list():
        ##         target_class = par.get('class')

        ## j = 0
        ## for par in inc:
        ##     if par.get('class') == target_class:
        ##         if j == 1:
        ##             rd['start_date'] = par.get_text()
        ##         elif j == 2:
        ##             rd['end_date'] = par.get_text()
        ##         elif j == 4:
        ##             rd['state'] = par.get_text().lower()
        ##         elif j == 5:
        ##             rd['county'] = par.get_text().lower()
        ##         j += 1

def verify(df):
    print('No states:', df[df.state.isnull()].shape[0])
    print('No counties:', df[df.county.isnull()].shape[0])
    print('No start date:', df[df.start_date.isnull()].shape[0])

    print(df[(df.state.isnull()) | (df.county.isnull()) | 
             (df.start_date.isnull())])

def report_number(rep):
    return re.search(r">ob_(\d+)", rep).group(1)

def outbreak_reference(rep):
    try:
        return re.search(r'>ob_\d+\s*-\s*(.*?)<', rep).group(1)
    except:
        raise ValueError('Error in outbreak reference extraction:')

def dates(rep):
    match = re.findall(r'(\d\d\d\d/\d\d/\d\d)', rep)
    match.sort()
    if len(match) == 1:
        return match[0], np.nan
    elif len(match) == 2:
        return match[0], match[1]
    else:
        return np.nan, np.nan
        raise ValueError('Error in date extraction:', match)

def state_county(rep, states_list, counties_list):
    # state
    states = [x for x in states_list if re.search(r'>' + x + r'<', rep)] 
    ## if '145720' in rep:
    ##     with open('tp', 'w') as f: f.write(rep)
    ##     set_trace()

    if len(states) == 1:
        state = re.sub(r'\*', '', states[0])
    else:
        state = np.nan
        return np.nan, np.nan

    # county
    if ' *' in state:
        state = re.sub(r'\*', '', state)
    counties = [x for x in counties_list[state] if re.search(r'>' + x + r'<', rep)]
    if len(counties) == 1:
        county = re.sub(r'\*', '', counties[0])
        return state, county
    elif len(counties) > 1:
        counties.remove(state)   # state name == county name scenario
        if len(counties) == 1:
            county = re.sub(r'\*', '', counties[0])
            return state, county
        else:
            return state, np.nan
    else:
        return state, np.nan

def location(rep):
    match = re.search(r'(-?\d+\.\d+)[\s,]*(-?\d+\.\d+)', rep, 
                      re.MULTILINE | re.DOTALL)
    try:
        return match.group(1), match.group(2)
    except:
        return np.nan, np.nan
        raise ValueError('Error in location extraction:', match)
# dairy
def old_dairy():
    df = pd.read_excel('../h5n1/dairy.xlsx')
    df = df.fillna(method='ffill')
    df.Confirmed = pd.to_datetime(df.Confirmed)
    df.loc[:, 'quarter'] = df.Confirmed.dt.quarter
    df.State = df.State.str.lower()
    
    states = loader.load('usa_states')
    df = df.merge(states[['fips', 'state']], left_on='State', right_on='state')
    df = df.drop('state', axis=1)
    
    df.to_csv('dairy.csv', index=False)

    ## # goats
    ## print('goats')
    ## dfs = df[df.short_desc=='GOATS, MEAT & OTHER - SALES, MEASURED IN HEAD']

    ## dfs.loc[dfs.domain_desc=='TOTAL', 'category'] = 'county_total'
    ## dfs = dfs[dfs.category!='unassigned']
    # It is not clear if inventory and sales can be compared.
    ## dfs = dfs[(df.domaincat_desc.str.contains(r'AREA OPERATED: \('))]
    ## dfs.loc[(dfs.county_code==-1) &
    ##         (dfs.domaincat_desc.str.contains(
    ##             r'AREA OPERATED: \(')), 'category'] = 'state_by_farmsize'

    ## heads.append(dfs)
    ## print(dfs.category.value_counts())

    # Currently deleting all rows that are in the large farm category not 
    # specified at the county level
    ## dfs = dfs[~(dfs.domaincat_desc.str.contains(
    ##     '500 TO 999 HEAD|1,000 TO 2,499 HEAD|2,500 OR MORE HEAD')) &
    ##           (dfs.category=='state_by_farmsize')]
    # combining farm sizes
    ## tdf = dfs[(dfs.domaincat_desc.str.contains(
    ##     '500 TO 999 HEAD|1,000 TO 2,499 HEAD|2,500 OR MORE HEAD')) &
    ##           (dfs.category=='state_by_farmsize')]
    ## cols = list(set(tdf.columns.tolist()) - set(['value', 'domaincat_desc']))
    ## tdf = tdf[cols+['value']].groupby(cols).sum().reset_index()
    ## tdf['domaincat_desc'] = 'INVENTORY OF COWS: (500 OR MORE HEAD)'
    ## 
    ## dfs = pd.concat([dfs, tdf])


def subtype_sums(df, col):
    if not df.head(1).category.values == 'state_by_farmsize':
        return df[col]
    sum = df[df.subtype!='all'].value.sum()
    if sum < -1:
        sum = -1
    if df[df.subtype=='all'].shape[0] == 1:
        df.loc[df.subtype=='all', 'value'] = sum
    elif df[df.subtype=='all'].shape[0] == 0:
        new_row = df.head(1)
        new_row.subtype = 'all'
        new_row.value = sum
        df = pd.concat([df, new_row])
    return df[col]

