        stats.cells_generated = False
        if not loc_glw.shape[0]:    # Load shape and generate cells
            stats.cells_generated = True
            counties = loader.load('usa_county_shapes')
            cg = counties[(counties.statefp==args.state) &
                    (counties.countyfp==args.county)]
            
            geometry.shape_to_glw_cells(cg.geometry)
            set_trace()

            cells = gen_glw_cells(stats.state,stats.county)
def adjust_heads_by_farmsize(df):
    print('Adjusting new heads to be consistent with farm sizes.')
    
    df = df.reset_index(drop=True)
    ind = (df.unit=='heads') & (df.category=='county_by_farmsize')
    heads = df[ind].copy()

    farms = df[(df.unit=='operations') & (df.category=='county_by_farmsize')].copy()
    farms['lb'] = farms.size_min * farms.value
    farms['ub'] = farms.size_max * farms.value

    heads = heads.merge(farms[['state', 'county', 'subtype', 'size_min', 'size_max', 'lb', 'ub']], on=['state', 'county', 'subtype', 'size_min', 'size_max'], how='left')
    heads.value = heads[['value', 'ub']].min(axis=1)
    heads.value = heads[['value', 'lb']].max(axis=1)

    df['new_value'] = df.value
    try:
        df.loc[ind, 'new_value'] = heads.value.values
    except:
        set_trace()

    # Verify: only the values that were -1 originally should be affected.
    changed = df.value!=df.new_value
    if df[changed].value_original.drop_duplicates().values!=[-1]:
        raise ValueError('Some fixed value was changed in this process.')
    else:
        print('Number of changed values:', changed.sum())

    # Assign new values
    df.value = df.new_value
    df = df.drop('new_value', axis=1)

    return df

        # Final lower bound
        sbf_lb = sbf_heads.groupby('state')['value'].sum().values
        tdf.lb = np.maximum(tdf.lb, sbf_lb)
    fdf['lb'] = fdf.size_min * fdf.value
    fdf['ub'] = fdf.size_max * fdf.value

    # Are heads consistent with farm sizes?
    min_criterion = ((tdf.value!=-1) & 
                     (tdf.value < tdf.lb)).sum() 
    max_criterion = ((tdf.value!=-1) & 
                     (tdf.value > tdf.ub)).sum() 

    if min_criterion or max_criterion:
        print('Minimum or Maximum criterion violated.')
        set_trace()

    # Are heads consistent with farm sizes?
    min_criterion = ((tdf.value!=-1) & 
                     (tdf.value < tdf.lb)).sum() 
    max_criterion = ((tdf.value!=-1) & 
                     (tdf.value > tdf.ub)).sum() 

    if min_criterion or max_criterion:
        print('Minimum or Maximum criterion violated.')
        set_trace()

def milp_livestock(statefp, countyfp, livestock, glw, farms, heads, stats):

    print(statefp, countyfp, livestock)

    # cells
    qvec = glw.val.to_numpy()
    m = qvec.shape[0]
    qsum = round(qvec.sum())

    # farms
    farms = farms.sort_values(by='size_min')
    farms.loc[farms.size_min==-1, 'size_min'] = 1
    farms.loc[farms.size_max==-1, 'size_max'] = INFINITY
    stats.farm_cat_absent = farms[
            (farms.size_min==-1) & (farms.size_max==-1)].sum()

    Nvec = farms.value.values
    Wmin = farms.size_min.to_list()
    Wmax = farms.size_max.to_list()
    ell = farms.shape[0]
    minheads = np.dot(Wmin, Nvec)
    maxheads = np.dot(Wmax, Nvec)

    # heads
    heads = heads.sort_values(by='size_min')
    Hvec = heads[heads.category=='county_by_farmsize'].value.values
    H = heads[heads.category=='county_total'].value.values[0]

    stats.num_farms = sum(Nvec)
    stats.glw_cells = m
    stats.num_categories = ell
    stats.glw_sum = qsum

    # Decide scenario and normalize if necessary
    if Hvec.shape[0] and not (Hvec==-1).any():
        stats.scenario = 1
    elif H >= minheads and H <= maxheads:
        stats.scenario = 2
    elif qsum >= minheads and qsum <= maxheads:
        stats.scenario = 3.1
        H = qsum
    elif qsum < minheads:
        stats.scenario = 3.2
        H = minheads
    elif qsum > maxheads:
        stats.scenario = 3.3
        H = maxheads

    stats.normalize = H/qvec.sum()
    qvec = qvec * stats.normalize


    # MILP starts here
    # Create a new model
    model = gp.Model("Optimal Livestock Distribution")
    
    # Create variables
    x = model.addVars(ell, m, vtype=GRB.INTEGER, name="x")
    h = model.addVars(ell, m, vtype=GRB.CONTINUOUS, name="h")
    lamb = model.addVar(vtype=GRB.CONTINUOUS, name="lambda")
    
    # Set objective
    model.setObjective(lamb, GRB.MINIMIZE)

    # Add constraints
    # Constraint 1: Each farm should be assigned to one of the cells
    for i in range(ell):
        model.addConstr(
                gp.quicksum(x[i, j] for j in range(m)) == Nvec[i], \
                        name=f"farm_assignment_{i}")

    # Constraint 2: Head count constraints for each farm category
    for i in range(ell):
        for j in range(m):
            model.addConstr(h[i, j] >= x[i, j] * Wmin[i], name=f"head_count_min_{i}_{j}")
            model.addConstr(h[i, j] <= x[i, j] * Wmax[i], name=f"head_count_max_{i}_{j}")

    # Constraint 3: Total head count over all farms of a category
    if stats.scenario == 1:
        for i in range(ell):
            model.addConstr(gp.quicksum(h[i, j] for j in range(m)) == Hvec[i], name=f"total_head_count_{i}")
    else:
        model.addConstr(gp.quicksum(h[i, j] for i in range(ell) for j in range(m)) == H, name=f"total_head_count")
    
    # Constraint 5: Gap constraints
    for j in range(m):
        model.addConstr(qvec[j] - gp.quicksum(h[i, j] for i in range(ell)) <= lamb, name=f"gap_upper_{j}")
        model.addConstr(qvec[j] - gp.quicksum(h[i, j] for i in range(ell)) >= -lamb, name=f"gap_lower_{j}")
    
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

    h_array = np.zeros((ell, m))
    x_array = np.zeros((ell, m))
    for i in range(ell):
        for j in range(m):
            h_array[i, j] = h[i, j].X
            x_array[i, j] = x[i, j].X

    stats.lamb = lamb.x
    model.dispose()

    dfl = []
    for i,cat in enumerate(Wmin):
        glw['size_min'] = Wmin[i]
        glw['size_max'] = Wmax[i]
        glw['farms'] = x_array[i,:]
        glw['heads'] = h_array[i,:]
        dfl.append(glw.copy())
    df = pd.concat(dfl, ignore_index=True)
    df.loc[df.size_max==INFINITY, 'size_max'] = -1

    df = df[df.farms>0]
    corr_vec = df[['x', 'y']].copy()
    corr_vec['heads'] = df.heads
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
    ## admin2_layer = pdk.Layer(
    ##     "PolygonLayer",
    ##     admin2,
    ##     stroked=True,
    ##     get_line_width=50,
    ##     get_polygon="polygon",
    ##     get_fill_color=[255, 255, 255],  
    ##     get_line_color=[0, 0, 0],         # Black outline
    ##     pickable=True,
    ##     extruded=True,
    ##     wireframe=True,
    ##     filled=True
    ## )

    ## # Define the GridCellLayer with calculated cell size
    ## grid_cell_layer = pdk.Layer(
    ##     "ColumnLayer",
    ##     gdf,
    ##     pickable=True,
    ##     filled=True,
    ##     get_position="pydeck_geometry",
    ##     get_elevation="farms+100",
    ##     elevation_scale=10,
    ##     radius=50,
    ##     # cell_size=cell_size_lon,  # Set width based on longitude
    ##     get_fill_color='color',
    ##     extruded=True,
    ##     #elevation_range=[0, 1000],
    ## )
## # Create a GeoJSON with color information
## geojson_dict = json.loads(geojson_data)
## for feature in geojson_dict['features']:
##     value = feature['properties']['value']  # Replace with your column name
##     feature['properties']['color'] = get_viridis_color(value, min_value, max_value)
## 
## # Save the modified GeoJSON
## with open('choropleth_geojson.json', 'w') as f:
##     json.dump(geojson_dict, f)
## 
## 
## # Create the map deck
## deck = pdk.Deck(
##     initial_view_state=view,
##     layers=[
##         pdk.Layer(
##             'GeoJsonLayer',
##             data=choropleth_data,
##             get_fill_color='[color[0], color[1], color[2], 200]',  # Use the color from the GeoJSON
##             get_line_color=[0, 0, 0, 255],  # Black boundaries
##             line_width_min_pixels=1,
##             opacity=0.6
##         )
##     ],
##     map_style='mapbox://styles/mapbox/light-v9'  # Base map style
## )
## 
## # Save the map as an HTML file
## deck.to_html('viridis_choropleth_map.html')


   ##  # Create the map deck
   ##  deck = pdk.Deck(
   ##      initial_view_state=view,
   ##      layers=[
   ##          pdk.Layer(
   ##              'GeoJsonLayer',
   ##              data=geojson_data,
   ##              get_fill_color=[200, 0, 0, 100],  # Set color for features
   ##              get_line_color=[0, 0, 0, 255],    # Set color for boundaries
   ##              line_width_min_pixels=1,
   ##              opacity=0.6
   ##          )
   ##      ],
   ##      map_style='mapbox://styles/mapbox/light-v9'  # Base map style
   ##  )
    



    ## norm = LogNorm(vmin=1, vmax=gdf.heads.max())

    ## ## ax = plot.subplot(fig=fig, ax=ax, func='gpd.plot',
    ## ##                   data=gdf,
    ## ##                   pf_column='heads',
    ## ##                   pf_legend=False, 
    ## ##                   pf_legend_kwds={'shrink': 0.5, 
    ## ##                                   'label': 'Heads',
    ## ##                                   'orientation': 'horizontal'},
    ## ##                   pf_norm=norm,
    ## ##                   pf_cmap=args.palette,
    ## ##                   la_xlabel='', la_ylabel=''
    ## ##                   )
    ## plot.savefig('temp.png')

    ## from PIL import Image
    ## from PIL import ImageDraw

    ## # Open the saved map image
    ## image = Image.open('temp.png')

    ## # Define the perspective transformation
    ## width, height = image.size
    ## coeffs = (1, -0.5, 0,  # coefficients for the x-axis
    ##           0, 1, -0.3,  # coefficients for the y-axis
    ##           0.0008, 0.0008)  # perspective distortion coefficients

    ## # Apply the perspective transformation
    ## image = image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    ## # Save and display the transformed image
    ## image.save('us_map_perspective.png')
    ## image.show()






    ### Fraction of valid assignments for farms by livestock
    stats_farms = stats[stats.feasible_sol][['livestock', 'cat', 'num_farms']
                        ].groupby(['cat', 'livestock']
                                  ).sum().reset_index()
    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.barplot', 
                      data=stats_farms, 
                      pf_x='livestock', pf_y='num_farms', pf_hue='cat',
                      pf_hue_order=['Optimal', 'Non-optimal', 'Uniform', 'Failed'],
                      pf_alpha=1,
                      xt_rotation=25, la_xlabel='', la_ylabel='',
                      la_title='Success of assignments by farms')

    ax = plot.subplot(fig=fig, grid=gs[0,2], func='sns.scatterplot', 
                      data=stats,
                      pf_x='num_farms', pf_y='glw_cells', pf_hue='time', 
                      pf_palette='YlOrRd', pf_edgecolor='black',
                      la_xlabel='Farms', la_ylabel='Cells',
                      la_title='Time taken (seconds)',
                      ag_xmin=0, ag_ymin=0)


function rich(){ # example farm to cell assignment
    #python ../scripts/farm_cats_to_cells.py -s 51 -c 3 -t cattle
    python ../scripts/farms_to_cells.py -s virginia -c richmond -t cattle
}

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
    Ngk = {}
    Hgk = {}
    for subtype in subtypes:
        Ngk[subtype] = fdf[subtype].values
        Hgk[subtype] = hdf[subtype].values
    Wmin = fdf.size_min.to_list()
    Wmax = fdf.size_max.to_list()
    ell = fdf.shape[0]

    # Not sure if the total population is required here.
    H = heads[heads.category=='county_total'].value.values[0]

    stats.num_farms = sum(Ni)
    stats.num_categories = ell

    # MILP starts here
    # Create a new model
    model = gp.Model("Optimal Livestock Distribution")

    
    
    # Create variables
    y = model.addVars(ell, m, vtype=GRB.INTEGER, name="y")
    x = model.addVars(subtypes, ell, m, vtype=GRB.INTEGER, name="x")
    h = model.addVars(subtypes, ell, m, vtype=GRB.INTEGER, name="h")
    lambda1 = model.addVar(vtype=GRB.INTEGER, name="lambda1")
    
    # Set objective
    model.setObjective(lambda1, GRB.MINIMIZE)

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
                    Ngk[g][i])
            model.addConstr(gp.quicksum(h[g, i, j] for j in range(m)) == 
                    Hgk[g][i])

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
    ## # instances = df[col].drop_duplicates().values
    ## df = df.groupby(col).apply(subtype_sums, include_groups=False).reset_index()
    ## df = df.drop('level_7', axis=1)

    ## df = adjust_farms_by_total_heads(df)

def subtype_sums(df):
    state = df.name[0]
    county = df.name[1]
    cat = df.name[3]
    unit = df.name[4]

    if not cat in ['state_by_farmsize', 'county_by_farmsize']:
        return df
    if unit == 'heads':
        #agg = df[df.subtype!='all'].value # keep it as is. will be ignored
        agg = df[df.subtype!='all'].value.sum()
    elif unit == 'operations':
        # Sometimes, the total reported farms is > than sum of subtypes. 
        # We will keep the value as is since it is consistent with the
        # generated reports. However, it might lead to infeasible solutions
        # downstream. However, there are instances where this information is 
        # omitted. In such cases, we have assigned the max of all subtypes.
        if df[df.subtype=='all'].shape[0]:
            val = df[df.subtype=='all'].value.values[0]
            sum_subtypes = df[df.subtype!='all'].value.sum()
            max_subtypes = df[df.subtype!='all'].value.max()
            agg = min(sum_subtypes, max(val, max_subtypes))
        else:
            agg = df.value.sum()
    if agg < -1:
        agg = -1

    if df[df.subtype=='all'].shape[0] == 1:
        df.loc[df.subtype=='all', 'value'] = agg
    elif df[df.subtype=='all'].shape[0] == 0:
        ## if state=='arkansas' and county=='jefferson' and unit=='operations' and cat=='county_by_farmsize':
        ##     set_trace()
        name = df.name
        new_row = df.head(1).copy()
        new_row.loc[:, ['subtype', 'value']] = ('all', agg)
        df = pd.concat([df, new_row], ignore_index=True)
        df.name = name
    else:
        raise ValueError('More than one field found for "all".')

    return df



def adjust_farms_by_total_heads(df):
    df = df.reset_index(drop=True)
    heads = df[(df.unit=='heads') & (df.category=='county_by_farmsize') & 
            (df.subtype=='all')].copy()
    farms = df[(df.unit=='operations') & (df.category=='county_by_farmsize') &
            (df.subtype=='all')].copy()
    heads['lb'] = np.ceil(heads.value / heads.size_max)
    heads['ub'] = np.floor(heads.value / heads.size_min)

    ind = farms.index
    farms = farms.merge(heads[
        ['state', 'county', 'subtype', 'size_min', 'size_max', 'lb', 'ub']], 
            on=['state', 'county', 'subtype', 'size_min', 'size_max'], how='left')
    farms.value = farms[['value', 'ub']].min(axis=1)
    farms.value = farms[['value', 'lb']].max(axis=1)
    df.loc[ind, 'value'] = farms.value.values
    return df
