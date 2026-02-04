## def transform_data(data, transform, logmin=LOG_EPSILON, clip=(None, None)):
##     if clip[0] is not None:
##         data.loc[:] = np.where(data<clip[0], clip[0], data)
##     if clip[1] is not None:
##         data.loc[:] = np.where(data>clip[1], clip[1], data)
## 
##     if transform == 'linear':
##         return data
##     elif transform == 'log':
##         nm = np.where(data>0, data, np.nan)
##         nm = np.log10(nm)
##         data.loc[:] = np.where(np.isnan(nm), logmin, nm)
##         return data
##     else:
##         raise ValueError(f'"{transform}" not supported transform type.')
## 
## def transform_ticks_labels(ticks, labels, transform):
##     if transform == 'linear':
##         pass
##     elif transform == 'log':
##         ticks = list(range(int(np.floor(ticks[0])), int(np.ceil(ticks[-1]))+1))
##         labels = [rf'$10^{{{x}}}$' for x in ticks]
##     else:
##         raise ValueError(f'"{transform}" not supported transform type.')
##     return ticks, labels

## def custom_legend(axobj, **kwargs):
##     return
## 
##     ax = axobj.ax
##     fig = axobj.fig.fig
##     colormap = axobj.cmap
## 
##     arg = augment_dict(kwargs, axobj.default_colorbar)
## 
##     # title
##     title_arg, arg = extract_from_dict_by_prefix(arg, 'title_')
##     title_arg = font_map(axobj.fonts, title_arg)
##     title = arg['title']
##     del arg['title']
## 
##     # add colorbar
##     mappable = cm.ScalarMappable(cmap=colormap, norm=axobj.norm)
##     cbar = fig.colorbar(mappable, ax=ax, **kwargs)
##     cbar.set_label(title, **title_arg)
##     axobj._legend_container = cbar
##     return

def risk_maps():
    # load risk maps
    df = pd.read_csv('../results/risk_scores_sp1_cr1.csv')

    # combine poultry
    df.loc[df.subtype.isin(['ckn-broilers', 'ckn-layers', 'ckn-pullets', 
                            'turkeys', 'ducks']), 'subtype'] = 'poultry'
    df = risk.combine_probs(df, id_vars=['county_code', 'subtype'])
    livestock_list = df.subtype.drop_duplicates().tolist()

    # time range
    months = pd.date_range(start='2024-11', end='2025-02', freq="MS"
                           ).strftime('%Y-%m').tolist()

    # rank
    pdf = risk.percentile_categorize(
            df, cols=months,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])
    
    # set color
    colors = plot.COLORS['cbYlOrRd']
    colormap = {
            "Very high": colors[7],
            "High": colors[6],
            "Medium": colors[4],
            "Low": colors[2],
            'Very low': colors[0]
            }

    # load maps
    regions, states = utils.load_shapes()
    # qmap = {1: 'Jan-Mar', 2: 'Apr-Jun', 3: 'Jul-Sep', 4: 'Oct-Dec'}

    # AA: will need central valley adjustment
    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=-.1, gs_hspace=-.5,
                                   color='tableau10',
                                   scilimits=[-2,2])
    i = 0
    for livestock in livestock_list:
        print(livestock)
        j = 0
        for month in months:
            print(i,j)
            if i == 2:
                xlabel = month
            else:
                xlabel = ''
            if j == 0:
                ylabel = livestock
            else:
                ylabel = ''

            tdf = pdf[pdf.subtype==livestock][['county_code', 'subtype', month]]
            tdf = tdf.rename(columns={month: 'risk'})
            tdf.risk = tdf.risk.astype(str)
            gdf = regions[['county_code', 'geometry']
                          ].merge(tdf, on=['county_code'], how='left')
            gdf.loc[gdf.risk.isnull(), 'risk'] = 'Very low'

            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                              data=gdf, 
                              pf_color=gdf.risk.map(colormap),
                              la_xlabel=xlabel, 
                              la_ylabel=ylabel, fs_xlabel='large')
            j += 1
        i += 1
    legend_elements = [Patch(facecolor=colormap[key], label=key)
                       for key in colormap]
    ax.legend(handles=legend_elements, 
              loc="lower right", bbox_to_anchor=(0.25,-.17), fontsize=14, 
              title_fontsize=10)
    plot.savefig(f'risk_maps.pdf')
@cli.command()
def wastewater():
    # pending load farms
    # load risk maps
    rs = pd.read_csv('../results/risk_scores_sp1_cr1.csv')
    tdf = utils.combine_probs(rs.drop('subtype', axis=1), 
                              id_vars=['time', 'county_code'])
    tdf['subtype'] = 'all'
    rs = pd.concat([rs, tdf])

    # load events
    __, events, __, __ = risk.load_features()

    # load ww
    ww = utils.wastewater()

    # load county neighbors
    cn = pd.read_parquet('../intermediate_data/county_hops.parquet',
                         columns=['source', 'target', 'length'])

    # evaluation period
    eval_period = ['2024-04', '2025-01']

    # get prevalence for the eval period
    ww = ww[(ww.month>=eval_period[0]) & (ww.month<=eval_period[1])].copy()

    ## # merge persistence with ground truth
    ## pdf = pdf.reset_index().merge(ww, on='county_code', how='left').fillna(0)

    ## df = pdf[(pdf.subtype=='all') & (pdf.risk_threshold==0.1)]

    ## # do roc stuff
    ## score = pdf.groupby(['subtype', 'risk_threshold']).apply(
    ##         lambda x: roc_auc_score(x.present, x.persistence))

    # county neighbors
    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y
    cn = cn[['county_x', 'county_y']]

    cnrs = cn.merge(rs, left_on='county_y', right_on='county_code', how='left')
    cnrs = cnrs[~cnrs.county_code.isnull()]
    cndf = cnrs.groupby(['subtype', 'risk_threshold', 'county_x']
                        )['persistence'].max().reset_index()
    ## cndf = cndf.merge(ww, left_on='county_x', right_on='county_code')

    ## df = cndf[(cndf.subtype=='all') & (cndf.risk_threshold==0.01)]

    ## # do roc stuff
    ## score = cndf.groupby(['subtype', 'risk_threshold']).apply(
    ##         lambda x: roc_auc_score(x.present, x.persistence))
    ## pdf = pdf.drop('present', axis=1)

    # cnpdf = cn.merge(pdf, left_on='county_y', right_on='county_code', how='left')
    ## cnpdf = cnpdf[~cnpdf.county_code.isnull()]
    ## cndf = cnpdf.groupby(['subtype', 'risk_threshold', 'county_x']
    ##                      )['persistence'].max().reset_index()
    ## cndf = cndf.merge(ww, left_on='county_x', right_on='county_code')

    ## df = cndf[(cndf.subtype=='all') & (cndf.risk_threshold==0.01)]

    ## # do roc stuff
    ## score = cndf.groupby(['subtype', 'risk_threshold']).apply(
    ##         lambda x: roc_auc_score(x.present, x.persistence))

    # rank rs by period and total
    labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    percentiles = [0, 50, 75, 90, 95, 100]
    tdf = rs[['time', 'subtype', 'county_code', '1']].rename(columns={'1': 'risk'})
    rrs = tdf.groupby('subtype', as_index=False, group_keys=True).apply(
            utils.percentile_categorize, timecol='time', riskcol='risk',
            start_time='2024-05', end_time='2025-01',
            percentiles=percentiles, labels=labels)
    ## rrs = utils.percentile_categorize(
    ##         rs, timecol='month', riskcol='risk',
    ##         start_time=start_month, end_time=end_month,
    ##         percentiles=[0, 50, 75, 90, 95, 100],
    ##         labels=labels)
    
    # merge risks and ww
    rrs = rrs.merge(ww, left_on=['time', 'county_code'], 
                    right_on=['month', 'county_code'], how='left').fillna(-1)
    rrs[rrs.present!=-1].to_csv(
            'ww_rank_across_periods_without_county_adjacency.csv', 
            index=False)

    ## cndf.to_csv('ww_rank_across_periods_with_county_adjacency.csv', index=False)

    subtypes = ['all', 'turkeys', 'ckn-layers', 'milk']
    for subtype in subtypes:
        tdf = rrs[(rrs.present==1) & (rrs.subtype==subtype)]
        df = tdf.rank_across_periods.value_counts() / tdf.count().values[0] * 100
        df = df.reset_index()
        if subtype == 'all':
            title = 'total'
        else:
            title = subtype

        fig, gs = plot.initiate_figure(x=8, y=4, 
                                       gs_nrows=1, gs_ncols=2,
                                       gs_wspace=.3, gs_hspace=.2,
                                       la_title=f'County-based evaluation: {title}',
                                       la_fontsize='large', la_y=.95,
                                       color='tableau10')
        ax = plot.subplot(fig=fig, grid=gs[0,0],
                          func='sns.barplot', data=df, 
                          pf_x='rank_across_periods', pf_y='count', 
                          pf_order=labels,
                          pf_color=plot.get_style('color', 2),
                          la_title='', la_xlabel='', 
                          la_ylabel=r'\% +ve reports',
                          xt_rotation=50,
                          lg_title=False)

        tdf = rrs[(rrs.present==0) & (rrs.subtype==subtype)]
        df = tdf.rank_across_periods.value_counts() / tdf.count().values[0] * 100
        df = df.reset_index()
        ax = plot.subplot(fig=fig, grid=gs[0,1],
                          func='sns.barplot', data=df, 
                          pf_x='rank_across_periods', pf_y='count', 
                          pf_color=plot.get_style('color', 4),
                          pf_order=labels,
                          la_title='', la_xlabel='', 
                          la_ylabel=r'\% -ve reports',
                          xt_rotation=50,
                          lg_title=False)
        plot.savefig(f'ww_county_{subtype}.pdf')

    df = cndf[cndf.present==1].rank_across_periods.value_counts() \
            / cndf[cndf.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_pos.pdf')
    df = cndf[cndf.present==0].rank_across_periods.value_counts() \
            / cndf[cndf.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_neg.pdf')

    # for choropleth
    start_month = '2024-11'
    end_month = '2025-01'
    tdf = rs[['county_code', 'month', 'risk']][(rs.month>=start_month) & 
                                               (rs.month<=end_month)].drop(
                                                       'month', axis=1)
    tdf = utils.combine_probs(tdf, id_vars='county_code')
    tdf['month'] = 0    # dummy
    rrs = utils.percentile_categorize(
            tdf, timecol='month', riskcol='risk',
            start_time=0, end_time=0,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)

    # merge risks and ww
    wwa = ww[(ww.month>=start_month) & (ww.month<=end_month)].groupby(
            'county_code', as_index=False)['present'].max()
    rrs = rrs.merge(wwa, on='county_code', how='left').fillna(-1)

    df = rrs[['county_code', 'rank_across_periods']].copy()
    df.rank_across_periods = df.rank_across_periods.astype(str)
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['state_code', 'geometry']].dissolve(by='state_code')
    regions = regions.merge(df, on='county_code', how='left').fillna('Very low')
    #regions.risk = np.log(regions.risk+1)
    colors = plot.COLORS['cbYlGnBu']
    colormap = {
            "Very high": colors[4],
            "High": colors[3],
            "Medium": colors[2],
            "Low": colors[1],
            'Very low': colors[0]
            }
    ww_ = ww.county_code.drop_duplicates().reset_index()
    rww = regions.merge(ww_, on='county_code')
    
    # plot
    fig, gs = plot.initiate_figure(x=10, y=8, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.3, gs_hspace=.6,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
                      data=regions, 
                      pf_color=regions.rank_across_periods.map(colormap),
                      la_title=f'Period: {start_month} to {end_month}',
                      fs_title='normalsize')
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,0], 
                      func='gpd.plot',
                      pf_edgecolor='black', pf_facecolor='none',
                      pf_linewidth=.3, data=states) 

    legend_elements = [Patch(facecolor=colormap[key], label=key)
                       for key in colormap]
    ax.legend(handles=legend_elements, 
              loc="lower right", bbox_to_anchor=(1,.01), fontsize=15, 
              title_fontsize=10)
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[:,:2], 
                      func='gpd.plot',
                      pf_edgecolor='red', pf_facecolor='none',
                      pf_linewidth=1, data=rww) 

    plot.savefig('ww_riskmap.pdf')

    # ww centric analysis
    # Given a ww cluster, is there a corresponding H5 outbreak?
    # within county outbreak
    pdf = ww[ww.present==1].merge(odf, on='county_code', how='left')
    pdf['match'] = True
    npdf = pdf.loc[pdf.end.isnull()].copy()
    npdf['match'] = False

    ppdf = pdf[~pdf.start.isnull()].copy()
    
    ppdf.start = pd.to_datetime(ppdf.start)
    ppdf.end = pd.to_datetime(ppdf.end)
    ppdf.month = pd.to_datetime(ppdf.month)

    ppdf.match = ((ppdf.month.dt.year-ppdf.start.dt.year)*12 + \
            (ppdf.month.dt.month-ppdf.start.dt.month)+w_minus >= 0) & \
            ((ppdf.end.dt.year-ppdf.month.dt.year)*12 + \
            (ppdf.end.dt.month-ppdf.month.dt.month)-w_plus >= 0)

    # ww centric analysis: adjacent county outbreak
    # Given a ww incidence, is there a corresponding outbreak in the current or
    # adjacent county?
    ### load county neighbors
    # county neighbors
    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y
    cn = cn[['county_x', 'county_y']]

    wwcn = cn.merge(ww, left_on='county_y', right_on='county_code', how='right')
    wwcn = wwcn[~wwcn.county_code.isnull()]

    pdfcn = wwcn[wwcn.present==1].merge(odf, left_on='county_x',
                                        right_on='county_code', how='left')
    pdfcn['match'] = True
    npdfcn = pdfcn.loc[pdfcn.end.isnull()].copy()
    npdfcn['match'] = False

    ppdfcn = pdfcn[~pdfcn.start.isnull()].copy()
    
    ppdfcn.start = pd.to_datetime(ppdfcn.start)
    ppdfcn.end = pd.to_datetime(ppdfcn.end)
    ppdfcn.month = pd.to_datetime(ppdfcn.month)

    ppdfcn.match = ((ppdfcn.month.dt.year-ppdfcn.start.dt.year)*12 + \
            (ppdfcn.month.dt.month-ppdfcn.start.dt.month)+w_minus >= 0) & \
            ((ppdfcn.end.dt.year-ppdfcn.month.dt.year)*12 + \
            (ppdfcn.end.dt.month-ppdfcn.month.dt.month)-w_plus >= 0)
    set_trace()



    set_trace()

    # outfile prefix
    prefix = f'ww_wp-{w_plus}_wm-{w_minus}'

    # get 

        ax = plot.oneplot(fg_x=5, fg_y=4, 
                          func='sns.histplot', data=res[res.subtype==subtype], 
                          pf_x='ahead', pf_hue='rank_across_periods', 
                          pf_kde=True, pf_stat='percent', pf_discrete=True,
                          pf_hue_order=['Very high', 'High', 'Medium', 'Low', 'Very low'],
                          pf_multiple='dodge',
                          la_xlabel=r'$k$-step ahead forecast', 
                          la_ylabel=r'\% instances', la_title=subtype, lg_title='')
## @cli.command()
## def performance():
##     # load risk maps
##     rs = rs[rs.subtype=='all']
## 
##     # load and process ww
##     ### positive
##     wwo = pd.read_csv('../../data/NWSS_wastewater/WWS_H5_detections.csv.gz')
##     wwo['month'] = wwo.rec_date.str[:7]
## 
##     ### positive
##     tdf = wwo[(wwo.rec_y!=0) & (~wwo.rec_y.isnull())].copy()
##     wwp = tdf[['month', 'counties_served']].drop_duplicates()
##     wwp['present'] = 1
## 
##     ### negative
##     tdf = wwo[(wwo.rec_y==0)].copy()
##     wwn = tdf[['month', 'counties_served']].drop_duplicates()
##     wwn['present'] = 0
## 
##     ww = pd.concat([wwp, wwn])
## 
##     start_month = ww.month.min()
##     end_month = ww.month.max()
##     print('start', start_month, 'end', end_month, 'pos', wwp.shape[0], 
##           'neg', wwn.shape[0])
##     ww.counties_served = ww.counties_served.apply(ast.literal_eval)
##     ww = ww.explode('counties_served').rename(columns=
##                                               {'counties_served': 'county_code'})
##     ww = ww[~ww.county_code.isnull()]
##     ww.county_code = ww.county_code.astype(int)
##     ww = ww.drop_duplicates()
## 
##     # rank rs by period and total
##     labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
##     rrs = utils.percentile_categorize(
##             rs, timecol='month', riskcol='risk',
##             start_time=start_month, end_time=end_month,
##             percentiles=[0, 50, 75, 90, 95, 100],
##             labels=labels)
##     
##     # merge risks and ww
##     rrs = rrs.merge(ww, on=['month', 'county_code'], how='left').fillna(-1)
##     rrs[rrs.present!=-1].to_csv(
##             'ww_rank_across_periods_without_county_adjacency.csv', 
##             index=False)
## 
##     # county neighbors
##     cn = pd.read_parquet('county_neighbors.parquet')
##     cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
##     cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y
##     cn = cn[['county_x', 'county_y']]
## 
##     cnrrs = cn.merge(rrs, left_on='county_y', right_on='county_code', how='left')
##     cnrrs = cnrrs[~cnrrs.county_code.isnull()]
##     cndf = cnrrs.groupby(['county_x', 'month'])['rank_across_periods'].max().reset_index()
##     cndf = cndf.merge(ww, left_on=['county_x', 'month'], 
##                       right_on=['county_code', 'month'])
##     cndf.to_csv('ww_rank_across_periods_with_county_adjacency.csv', index=False)
## 
##     df = rrs[rrs.present==1].rank_across_periods.value_counts() \
##             / rrs[rrs.present==1].count().values[0] * 100
##     df = df.reset_index()
##     ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
##                       func='sns.barplot', data=df, 
##                       pf_y='rank_across_periods', pf_x='index', 
##                       pf_order=labels,
##                       pf_color=2,
##                       la_title='County-based evaluation', la_xlabel='', 
##                       la_ylabel=r'\% +ve reports',
##                       xt_rotation=50,
##                       lg_title=False)
##     plot.savefig('ww_county_pos.pdf')
##     df = rrs[rrs.present==0].rank_across_periods.value_counts() \
##             / rrs[rrs.present==0].count().values[0] * 100
##     df = df.reset_index()
##     ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
##                       func='sns.barplot', data=df, 
##                       pf_y='rank_across_periods', pf_x='index', 
##                       pf_order=labels,
##                       pf_color=4,
##                       la_title='County-based evaluation', la_xlabel='', 
##                       la_ylabel=r'\% -ve reports',
##                       xt_rotation=50,
##                       lg_title=False)
##     plot.savefig('ww_county_neg.pdf')
## 
##     df = cndf[cndf.present==1].rank_across_periods.value_counts() \
##             / cndf[cndf.present==1].count().values[0] * 100
##     df = df.reset_index()
##     ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
##                       func='sns.barplot', data=df, 
##                       pf_y='rank_across_periods', pf_x='index', 
##                       pf_order=labels,
##                       pf_color=2,
##                       la_title='Adj. County-based evaluation', la_xlabel='', 
##                       la_ylabel=r'\% +ve reports',
##                       xt_rotation=50,
##                       lg_title=False)
##     plot.savefig('ww_adj_county_pos.pdf')
##     df = cndf[cndf.present==0].rank_across_periods.value_counts() \
##             / cndf[cndf.present==0].count().values[0] * 100
##     df = df.reset_index()
##     ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
##                       func='sns.barplot', data=df, 
##                       pf_y='rank_across_periods', pf_x='index', 
##                       pf_order=labels,
##                       pf_color=4,
##                       la_title='Adj. County-based evaluation', la_xlabel='', 
##                       la_ylabel=r'\% -ve reports',
##                       xt_rotation=50,
##                       lg_title=False)
##     plot.savefig('ww_adj_county_neg.pdf')
## 
##     # for choropleth
##     start_month = '2024-11'
##     end_month = '2025-01'
##     tdf = rs[['county_code', 'month', 'risk']][(rs.month>=start_month) & 
##                                                (rs.month<=end_month)].drop(
##                                                        'month', axis=1)
##     tdf = utils.combine_probs(tdf, id_vars='county_code')
##     tdf['month'] = 0    # dummy
##     rrs = utils.percentile_categorize(
##             tdf, timecol='month', riskcol='risk',
##             start_time=0, end_time=0,
##             percentiles=[0, 50, 75, 90, 95, 100],
##             labels=labels)
## 
##     # merge risks and ww
##     wwa = ww[(ww.month>=start_month) & (ww.month<=end_month)].groupby(
##             'county_code', as_index=False)['present'].max()
##     rrs = rrs.merge(wwa, on='county_code', how='left').fillna(-1)
## 
##     df = rrs[['county_code', 'rank_across_periods']].copy()
##     df.rank_across_periods = df.rank_across_periods.astype(str)
##     regions = loader.load('usa_county_shapes', contiguous_us=True)
##     states = regions[['state_code', 'geometry']].dissolve(by='state_code')
##     regions = regions.merge(df, on='county_code', how='left').fillna('Very low')
##     #regions.risk = np.log(regions.risk+1)
##     colors = plot.COLORS['cbYlGnBu']
##     colormap = {
##             "Very high": colors[4],
##             "High": colors[3],
##             "Medium": colors[2],
##             "Low": colors[1],
##             'Very low': colors[0]
##             }
##     ww_ = ww.county_code.drop_duplicates().reset_index()
##     rww = regions.merge(ww_, on='county_code')
##     
##     # plot
##     fig, gs = plot.initiate_figure(x=10, y=8, 
##                                    gs_nrows=1, gs_ncols=1,
##                                    gs_wspace=.3, gs_hspace=.6,
##                                    color='tableau10')
##     ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
##                       data=regions, 
##                       pf_color=regions.rank_across_periods.map(colormap),
##                       la_title=f'Period: {start_month} to {end_month}',
##                       fs_title='normalsize')
##     ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,0], 
##                       func='gpd.plot',
##                       pf_edgecolor='black', pf_facecolor='none',
##                       pf_linewidth=.3, data=states) 
## 
##     legend_elements = [Patch(facecolor=colormap[key], label=key)
##                        for key in colormap]
##     ax.legend(handles=legend_elements, 
##               loc="lower right", bbox_to_anchor=(1,.01), fontsize=15, 
##               title_fontsize=10)
##     ax = plot.subplot(fig=fig, ax=ax, grid=gs[:,:2], 
##                       func='gpd.plot',
##                       pf_edgecolor='red', pf_facecolor='none',
##                       pf_linewidth=1, data=rww) 
## 
##     plot.savefig('ww_riskmap.pdf')

@cli.command()
def persistence():
    # load risk maps
    rs = pd.read_csv('../results/risk_scores_sp1_cr1.csv')
    tdf = utils.combine_probs(rs.drop('subtype', axis=1), 
                              id_vars=['time', 'county_code'])
    tdf['subtype'] = 'all'
    rs = pd.concat([rs, tdf])

    # evaluation period
    eval_period = ['2024-06', '2025-01']

    # thresholds
    ts = [0.001, 0.005, 0.01, 0.1, 0.3]

    # compute risk persistence
    ## for now 1-month-ahead forecast
    df = rs[['county_code', 'subtype', 'time', '1']][
            (rs.time>=eval_period[0]) & (rs.time<=eval_period[1])]
    df = df.rename(columns={'1': 'score'})

    pdf_list = [] 
    for th in ts:
        pdf_list.append(df.groupby('subtype').apply(risk_persistence, threshold=th))
    pdf = pd.concat(pdf_list)

    # load and process ww
    wwo = pd.read_csv('../../data/NWSS_wastewater/WWS_H5_detections.csv.gz')
    wwo['month'] = wwo.rec_date.str[:7]

    ### positive
    tdf = wwo[(wwo.rec_y!=0) & (~wwo.rec_y.isnull())].copy()
    wwp = tdf[['month', 'counties_served']].drop_duplicates()
    wwp['present'] = 1

    ### negative
    tdf = wwo[(wwo.rec_y==0)].copy()
    wwn = tdf[['month', 'counties_served']].drop_duplicates()
    wwn['present'] = 0

    ww = pd.concat([wwp, wwn])

    start_month = ww.month.min()
    end_month = ww.month.max()
    print('start', start_month, 'end', end_month, 'pos', wwp.shape[0], 
          'neg', wwn.shape[0])
    ww.counties_served = ww.counties_served.apply(ast.literal_eval)
    ww = ww.explode('counties_served').rename(columns=
                                              {'counties_served': 'county_code'})
    ww = ww[~ww.county_code.isnull()]
    ww.county_code = ww.county_code.astype(int)
    ww = ww.drop_duplicates()

    wwp.counties_served = wwp.counties_served.apply(ast.literal_eval)
    wwp = wwp.explode('counties_served').rename(columns=
                                              {'counties_served': 'county_code'})

    set_trace()

    # get prevalence for the eval period
    ww = ww[(ww.month>=eval_period[0]) & (ww.month<=eval_period[1])].groupby('county_code')[
            'present'].max().reset_index()

    # merge persistence with ground truth
    pdf = pdf.reset_index().merge(ww, on='county_code', how='left').fillna(0)

    # do roc stuff
    score = pdf.groupby(['subtype', 'risk_threshold']).apply(
            lambda x: roc_auc_score(x.present, x.persistence))

    # county neighbors
    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y
    cn = cn[['county_x', 'county_y']]

    pdf = pdf.drop('present', axis=1)

    cnpdf = cn.merge(pdf, left_on='county_y', right_on='county_code', how='left')
    cnpdf = cnpdf[~cnpdf.county_code.isnull()]
    cndf = cnpdf.groupby(['subtype', 'risk_threshold', 'county_x']
                         )['persistence'].max().reset_index()
    cndf = cndf.merge(ww, left_on='county_x', right_on='county_code')

    # do roc stuff
    score = cndf.groupby(['subtype', 'risk_threshold']).apply(
            lambda x: roc_auc_score(x.present, x.persistence))
    set_trace()

    # rank rs by period and total
    labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    rrs = utils.percentile_categorize(
            rs, timecol='month', riskcol='risk',
            start_time=start_month, end_time=end_month,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)
    
    # merge risks and ww
    rrs = rrs.merge(ww, on=['month', 'county_code'], how='left').fillna(-1)
    rrs[rrs.present!=-1].to_csv(
            'ww_rank_across_periods_without_county_adjacency.csv', 
            index=False)
    set_trace()

    cndf.to_csv('ww_rank_across_periods_with_county_adjacency.csv', index=False)

    df = rrs[rrs.present==1].rank_across_periods.value_counts() \
            / rrs[rrs.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_county_pos.pdf')
    df = rrs[rrs.present==0].rank_across_periods.value_counts() \
            / rrs[rrs.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_county_neg.pdf')

    df = cndf[cndf.present==1].rank_across_periods.value_counts() \
            / cndf[cndf.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_pos.pdf')
    df = cndf[cndf.present==0].rank_across_periods.value_counts() \
            / cndf[cndf.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_neg.pdf')

    # for choropleth
    start_month = '2024-11'
    end_month = '2025-01'
    tdf = rs[['county_code', 'month', 'risk']][(rs.month>=start_month) & 
                                               (rs.month<=end_month)].drop(
                                                       'month', axis=1)
    tdf = utils.combine_probs(tdf, id_vars='county_code')
    tdf['month'] = 0    # dummy
    rrs = utils.percentile_categorize(
            tdf, timecol='month', riskcol='risk',
            start_time=0, end_time=0,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)

    # merge risks and ww
    wwa = ww[(ww.month>=start_month) & (ww.month<=end_month)].groupby(
            'county_code', as_index=False)['present'].max()
    rrs = rrs.merge(wwa, on='county_code', how='left').fillna(-1)

    df = rrs[['county_code', 'rank_across_periods']].copy()
    df.rank_across_periods = df.rank_across_periods.astype(str)
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['state_code', 'geometry']].dissolve(by='state_code')
    regions = regions.merge(df, on='county_code', how='left').fillna('Very low')
    #regions.risk = np.log(regions.risk+1)
    colors = plot.COLORS['cbYlGnBu']
    colormap = {
            "Very high": colors[4],
            "High": colors[3],
            "Medium": colors[2],
            "Low": colors[1],
            'Very low': colors[0]
            }
    ww_ = ww.county_code.drop_duplicates().reset_index()
    rww = regions.merge(ww_, on='county_code')
    
    # plot
    fig, gs = plot.initiate_figure(x=10, y=8, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.3, gs_hspace=.6,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
                      data=regions, 
                      pf_color=regions.rank_across_periods.map(colormap),
                      la_title=f'Period: {start_month} to {end_month}',
                      fs_title='normalsize')
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,0], 
                      func='gpd.plot',
                      pf_edgecolor='black', pf_facecolor='none',
                      pf_linewidth=.3, data=states) 

    legend_elements = [Patch(facecolor=colormap[key], label=key)
                       for key in colormap]
    ax.legend(handles=legend_elements, 
              loc="lower right", bbox_to_anchor=(1,.01), fontsize=15, 
              title_fontsize=10)
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[:,:2], 
                      func='gpd.plot',
                      pf_edgecolor='red', pf_facecolor='none',
                      pf_linewidth=1, data=rww) 

    plot.savefig('ww_riskmap.pdf')

@cli.command()
def wastewater_old():
    # load risk maps
    rs = pd.read_csv('../results/risk_scores_sp1_cr1.csv')
    rs = rs[rs.subtype=='all']

    # load and process ww
    ### positive
    wwo = pd.read_csv('../../data/NWSS_wastewater/WWS_H5_detections.csv.gz')
    wwo['month'] = wwo.rec_date.str[:7]

    ### positive
    tdf = wwo[(wwo.rec_y!=0) & (~wwo.rec_y.isnull())].copy()
    wwp = tdf[['month', 'counties_served']].drop_duplicates()
    wwp['present'] = 1

    ### negative
    tdf = wwo[(wwo.rec_y==0)].copy()
    wwn = tdf[['month', 'counties_served']].drop_duplicates()
    wwn['present'] = 0

    ww = pd.concat([wwp, wwn])

    start_month = ww.month.min()
    end_month = ww.month.max()
    print('start', start_month, 'end', end_month, 'pos', wwp.shape[0], 
          'neg', wwn.shape[0])
    ww.counties_served = ww.counties_served.apply(ast.literal_eval)
    ww = ww.explode('counties_served').rename(columns=
                                              {'counties_served': 'county_code'})
    ww = ww[~ww.county_code.isnull()]
    ww.county_code = ww.county_code.astype(int)
    ww = ww.drop_duplicates()

    # rank rs by period and total
    labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    rrs = utils.percentile_categorize(
            rs, timecol='month', riskcol='risk',
            start_time=start_month, end_time=end_month,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)
    
    # merge risks and ww
    rrs = rrs.merge(ww, on=['month', 'county_code'], how='left').fillna(-1)
    rrs[rrs.present!=-1].to_csv(
            'ww_rank_across_periods_without_county_adjacency.csv', 
            index=False)

    # county neighbors
    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y
    cn = cn[['county_x', 'county_y']]

    cnrrs = cn.merge(rrs, left_on='county_y', right_on='county_code', how='left')
    cnrrs = cnrrs[~cnrrs.county_code.isnull()]
    cndf = cnrrs.groupby(['county_x', 'month'])['rank_across_periods'].max().reset_index()
    cndf = cndf.merge(ww, left_on=['county_x', 'month'], 
                      right_on=['county_code', 'month'])
    cndf.to_csv('ww_rank_across_periods_with_county_adjacency.csv', index=False)

    df = rrs[rrs.present==1].rank_across_periods.value_counts() \
            / rrs[rrs.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_county_pos.pdf')
    df = rrs[rrs.present==0].rank_across_periods.value_counts() \
            / rrs[rrs.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_county_neg.pdf')

    df = cndf[cndf.present==1].rank_across_periods.value_counts() \
            / cndf[cndf.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_pos.pdf')
    df = cndf[cndf.present==0].rank_across_periods.value_counts() \
            / cndf[cndf.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_neg.pdf')

    # for choropleth
    start_month = '2024-11'
    end_month = '2025-01'
    tdf = rs[['county_code', 'month', 'risk']][(rs.month>=start_month) & 
                                               (rs.month<=end_month)].drop(
                                                       'month', axis=1)
    tdf = utils.combine_probs(tdf, id_vars='county_code')
    tdf['month'] = 0    # dummy
    rrs = utils.percentile_categorize(
            tdf, timecol='month', riskcol='risk',
            start_time=0, end_time=0,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)

    # merge risks and ww
    wwa = ww[(ww.month>=start_month) & (ww.month<=end_month)].groupby(
            'county_code', as_index=False)['present'].max()
    rrs = rrs.merge(wwa, on='county_code', how='left').fillna(-1)

    df = rrs[['county_code', 'rank_across_periods']].copy()
    df.rank_across_periods = df.rank_across_periods.astype(str)
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['state_code', 'geometry']].dissolve(by='state_code')
    regions = regions.merge(df, on='county_code', how='left').fillna('Very low')
    #regions.risk = np.log(regions.risk+1)
    colors = plot.COLORS['cbYlGnBu']
    colormap = {
            "Very high": colors[4],
            "High": colors[3],
            "Medium": colors[2],
            "Low": colors[1],
            'Very low': colors[0]
            }
    ww_ = ww.county_code.drop_duplicates().reset_index()
    rww = regions.merge(ww_, on='county_code')
    set_trace()
    
    # plot
    fig, gs = plot.initiate_figure(x=10, y=8, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.3, gs_hspace=.6,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
                      data=regions, 
                      pf_color=regions.rank_across_periods.map(colormap),
                      la_title=f'Period: {start_month} to {end_month}',
                      fs_title='normalsize')
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,0], 
                      func='gpd.plot',
                      pf_edgecolor='black', pf_facecolor='none',
                      pf_linewidth=.3, data=states) 

    legend_elements = [Patch(facecolor=colormap[key], label=key)
                       for key in colormap]
    ax.legend(handles=legend_elements, 
              loc="lower right", bbox_to_anchor=(1,.01), fontsize=15, 
              title_fontsize=10)
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[:,:2], 
                      func='gpd.plot',
                      pf_edgecolor='red', pf_facecolor='none',
                      pf_linewidth=1, data=rww) 

    plot.savefig('ww_riskmap.pdf')

@cli.command()
def human():
    # load risk maps
    rs = pd.read_csv('../results/risk_scores_sp1_cr1.csv')
    rs = rs[rs.subtype=='all']

    # load and process human
    hdf = pd.read_excel('../../data/h5n1/human_global_health.xlsx')
    hdf = hdf[hdf.news_type=='incidence']
    hdf = hdf.sort_values('date')
    hdf['cumm'] = hdf['count'].cumsum()

# OLD STUFF
@cli.command()
def farms_pop():
    farms = utils.load_poultry_farms()
    pf = farms[farms.heads>100].copy()
    pf = pf.sort_values(by='type')
    ax = plot.oneplot(fg_x=5, fg_y=4, 
                      func='sns.histplot', data=pf, 
                      pf_x='type', 
                      la_title='', la_xlabel='', la_ylabel=r'\#farms',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('poultry_farms.pdf')

    df = pf.groupby('type')['heads'].sum().reset_index()

    ax = plot.oneplot(fg_x=5, fg_y=4,
                      func='sns.barplot', data=df, 
                      pf_x='type', pf_y='heads',
                      pf_color=plot.get_style('color',1),
                      la_title='', la_xlabel='', la_ylabel=r'\#heads',
                      sp_yscale='log',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('poultry_heads.pdf')

@cli.command()
def farms_pop():
    farms = utils.load_poultry_farms()
    pf = farms[farms.heads>100].copy()
    pf = pf.sort_values(by='type')
    ax = plot.oneplot(fg_x=5, fg_y=4, 
                      func='sns.histplot', data=pf, 
                      pf_x='type', 
                      la_title='', la_xlabel='', la_ylabel=r'\#farms',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('poultry_farms.pdf')

    df = pf.groupby('type')['heads'].sum().reset_index()

    ax = plot.oneplot(fg_x=5, fg_y=4,
                      func='sns.barplot', data=df, 
                      pf_x='type', pf_y='heads',
                      pf_color=plot.get_style('color',1),
                      la_title='', la_xlabel='', la_ylabel=r'\#heads',
                      sp_yscale='log',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('poultry_heads.pdf')

@cli.command()
def poultry_birds_corr():
    bdf = utils.h5n1_birds(agg_by_quarter=True)
    pdf = utils.h5n1_poultry(agg_by_quarter=True, commercial=True)
    pdf = pdf.drop('type', axis=1)
    pdf = pdf.groupby(['state_code', 'county_code', 'year', 'quarter'], 
                      as_index=False).sum()

    pbdf = bdf.merge(pdf, on=['state_code', 'county_code', 'year', 
                              'quarter'], how='outer', indicator=True)
    pbdf._merge = pbdf._merge.map({'left_only': 'birds', 
                                   'right_only': 'poultry',
                                   'both': 'both'})
    pbdf._merge = pbdf._merge.astype(str)
    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=-.1, gs_hspace=-.2,
                                   color='tableau10')

    colors = plot.COLORS['tableau10']
    colormap = {
            "birds": colors[0],
            "poultry": colors[1],
            "both": colors[2],
            }
    legend_elements = [Patch(facecolor=colormap[key], label=key) 
                       for key in colormap]
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['statefp', 'geometry']].dissolve(by='statefp')
    rpbdf = regions.merge(pbdf, left_on=['statefp', 'countyfp'],
                          right_on=['state_code', 'county_code']).fillna(0)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1,2,3,4]:
            if j:
                ylabel = ''
            else:
                ylabel = f'{year}'
            if i == 2:
                xlabel = f'{quarter}'
            else:
                xlabel = ''
            tdf = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf.incidences.sum().astype(int)
            pi = tdf.reports.sum().astype(int)
            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.boundary.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                              data=tdf, 
                              pf_color=tdf._merge.map(colormap),
                              pf_markersize=2,
                              la_ylabel=ylabel, fs_ylabel='large',
                              la_title=f'b: {bi}, p: {pi}',
                              la_xlabel=xlabel, fs_xlabel='large')
            j += 1
        i += 1
    ax.legend(handles=legend_elements, 
              loc="lower left", bbox_to_anchor=(.7,.9), 
              fontsize=15, title_fontsize=12)
    plot.savefig('poultry_birds_corr.pdf')

@cli.command()
def incidence_distance():
    print('loading data ...')
    pdf = utils.h5n1_poultry(commercial=True).reset_index()
    pdf.confirmed = pd.to_datetime(pdf.confirmed)
    tdf = pdf[['index', 'confirmed', 'type', 'state_code', 'county_code']]
    counties = loader.load('usa_county_shapes', contiguous_us=True)
    counties['centroid'] = counties.centroid

    print('cross product ...')
    tdf_pairs = tdf.merge(tdf, how='cross', suffixes=['_x', '_y'])
    tdf_pairs = tdf_pairs[tdf_pairs.index_x!=tdf_pairs.index_y].copy()
    tdf_pairs['ddiff'] = np.abs((tdf_pairs.confirmed_x-tdf_pairs.confirmed_y
                                ).dt.days)
    tdf_pairs = tdf_pairs[tdf_pairs.ddiff<=60]

    print('merging with counties ...')
    tdf_pairs = tdf_pairs.merge(
            counties[['statefp', 'countyfp', 'centroid']], 
            left_on=['state_code_x', 'county_code_x'],
            right_on=['statefp', 'countyfp'], how='left')
    tdf_pairs = tdf_pairs.drop(['statefp', 'countyfp'], axis=1)
    tdf_pairs = tdf_pairs.rename(columns={'county': 'county_x'})
    tdf_pairs = tdf_pairs.merge(
            counties[['statefp', 'countyfp', 'centroid']], 
            left_on=['state_code_y', 'county_code_y'],
            right_on=['statefp', 'countyfp'], how='left')
    tdf_pairs = tdf_pairs.drop(['statefp', 'countyfp'], axis=1)
    tdf_pairs = tdf_pairs.rename(columns={'county': 'county_y'})

    # find distance
    print('find distance ...')
    gdf = gpd.GeoDataFrame(tdf_pairs, geometry='centroid_y', crs='epsg:4327')
    gdf = gdf.to_crs(epsg=3527)
    gdf.centroid_x = gdf.centroid_x.to_crs(epsg=3527)
    gdf['dist'] = gdf.centroid_x.distance(gdf.centroid_y) * 0.000621371
    gdf = gdf.sort_values('dist')
    total_instances = gdf.shape[0]
    # same_county_instances = gdf[gdf.dist<1]['dist'].count()

    # find hops
    print('find hops ...')
    hops = pd.read_parquet('../intermediate_data/county_hops.parquet')
    gdf = gdf.merge(hops, how='left',
                    on=['state_code_x', 'county_code_x', 'state_code_y',
                        'county_code_y'])
    gdf = gdf[~gdf.source.isnull()]
    same_county_instances = gdf[gdf['length']<1]['length'].count()

    print('plot ...')
    types = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'backyard']
    ntypes = len(types)
    fig, gs = plot.initiate_figure(x=5*ntypes, y=5*ntypes, 
                                   gs_nrows=ntypes, gs_ncols=ntypes,
                                   gs_wspace=.3, gs_hspace=.3,
                                   color='tableau10')
    ## ax = plot.subplot(fig=fig, grid=gs[0:2,2:], func='sns.histplot', 
    ##                   data=gdf[gdf.dist<=300], pf_x='dist',
    ##                   la_title=f'All pairs: {total_instances} pairs', 
    ##                   fs_title='Large',
    ##                   la_xlabel='Miles',  fs_xlabel='large',
    ##                   la_ylabel='Instances', fs_ylabel='large',
    ##                   lg_title=False)
    ax = plot.subplot(fig=fig, grid=gs[0:2,2:], func='sns.histplot', 
                      data=gdf[gdf['length']<=10], pf_x='length',
                      la_title=f'All pairs: {total_instances} pairs', 
                      fs_title='Large',
                      la_xlabel='Hops',  fs_xlabel='large',
                      la_ylabel='Instances', fs_ylabel='large',
                      lg_title=False)
    plot.text(ax=ax, 
              data=f'Same/adjacent county: {same_county_instances*100/total_instances: .2g}\\%',
              x=5, y=650, fontsize='large')

    i = 0
    for tp1 in types:
        j = 0
        for tp2 in types:
            if j > i :
                continue
            print(tp1, tp2)
            tdf = gdf[(gdf.type_x==tp1) & (gdf.type_y==tp2)].copy()
            # same_county_instances = tdf[tdf.dist<1]['dist'].count()
            same_county_instances = tdf[tdf['length']<1]['length'].count()
            total_instances = tdf.shape[0]
            same_county_perc = same_county_instances * 100 / total_instances
            if same_county_perc > 10:
                col = 2
            else:
                col = 1
            tdf.dist = tdf.dist.astype(int)
            ## ax = plot.subplot(fig=fig, grid=gs[i,j], func='sns.histplot', 
            ##                   data=tdf[tdf.dist<=300], pf_x='dist',
            ##                   pf_color = plot.get_style('color', col),
            ##                   la_title=f'{tp1}, {tp2}: {tdf.shape[0]}', 
            ##                   la_xlabel='Miles', 
            ##                   la_ylabel='',
            ##                   lg_title=False)
            ax = plot.subplot(fig=fig, grid=gs[i,j], func='sns.histplot', 
                              data=tdf[tdf['length']<=10], pf_x='length',
                              pf_color = plot.get_style('color', col),
                              la_title=f'{tp1}, {tp2}: {tdf.shape[0]}', 
                              la_xlabel='Hops', 
                              la_ylabel='',
                              lg_title=False)
            plot.text(ax=ax, 
                      data=f'Same county: {same_county_perc: .2g}\\%',
                      x=3, y=ax.get_ylim()[1]*3/4)
            j += 1
        i += 1
    plot.savefig('poultry_outbreaks_distance.pdf')

@cli.command()
def incidence_vs_farms():
    print('loading data ...')
    pdf = utils.h5n1_poultry(commercial=False,agg_by='month')
    # pdf = pdf[pdf.year==2022].copy()
    cpdf = pdf.groupby(['state_code', 'county_code', 'type'], as_index=False)[
            'reports'].sum()
    farms = utils.load_poultry_farms(aggregate='county', commercial=True)
    cpdf = cpdf.rename(columns={'type': 'subtype'})
    bind = pd.read_parquet('birds_h5n1_ind_buffer.parquet')
    bind = bind.drop(['year', 'month', 'dist'], 
                                      axis=1).drop_duplicates()

    print('correlation ...')
    farms = farms.merge(cpdf, on=['state_code', 'county_code', 'subtype'], 
                        how='left').fillna(0)
    xx = farms.groupby('subtype').apply(lambda x: spearmanr(x.farms, x.reports))
    print(f'rank correlation not conditioned on bird data: {xx}')

    sfarms = farms.merge(bind, on=['state_code', 'county_code'])
    xx = sfarms.groupby('subtype').apply(lambda x: spearmanr(x.farms, x.reports))
    print(f'rank correlation not conditioned on bird data: {xx}')

    # binned farms analysis
    tdf = sfarms.groupby('subtype').apply(utils.bin_farms)

@cli.command()
def events_vs_farms():
    type_events_vs_farms_dairy()
    type_events_vs_farms_poultry('all')
    type_events_vs_farms_poultry('turkeys')
    type_events_vs_farms_poultry('ducks')
    type_events_vs_farms_poultry('ckn-layers')
    type_events_vs_farms_poultry('ckn-broilers')
    type_events_vs_farms_poultry('ckn-pullets')

def type_events_vs_farms_dairy():
    delta = 15
    reps = pd.read_csv('../results/dairy_event_clusters.csv').fillna(-1)
    reps['type'] = 'milk'
    reps['production'] = -1
    reps = reps[reps.delta==delta]

    # production is used just for count
    sdf = reps.groupby(['event0', 'type']).agg(
            {'county': 'first', 'production': 'count'})
    sdf = sdf.rename(columns={'production': 'osize'}).reset_index()

    regions = loader.load('usa_county_shapes', contiguous_us=True)
    regions['county'] = regions.state_code*1000 + regions.county_code
    regions = utils.fit_central_valley(regions)
    regions = regions[['county', 'geometry']].drop_duplicates()
    regions = regions.to_crs(epsg=32633)
    regions['area'] = regions.geometry.area / 2.59e6

    fndf = pd.read_parquet(
            '../intermediate_data/farm_neighborhood_milk.parquet')

    farms = utils.load_dairy_farms(aggregate='county', commercial=True,
                                     commercial_threshold=100)
    farms = farms[['state_code', 'county_code', 'county', 'subtype', 
                   'farms', 'heads']].rename(columns={'subtype': 'type'})
    farms = utils.fit_central_valley(farms)
    farms = farms.groupby('county', as_index=False)[
            ['heads', 'farms']].sum()

    sdf = utils.fit_central_valley(sdf)
    sdf = sdf.merge(farms, on='county', how='left').fillna(0)

    fndf = utils.fit_central_valley(fndf)
    tdf = fndf.groupby(['fid', 'county'])['fid_n'].count().reset_index()
    cfndf = tdf.groupby('county')['fid_n'].sum().reset_index()
    sdf = sdf.merge(cfndf, on='county')
    sdf = sdf.rename(columns={'fid_n': 'local_nbhd'})
    sdf.local_nbhd = sdf.local_nbhd / sdf.farms

    sdf = sdf.merge(regions[['county', 'area']], on='county')
    sdf['fd'] = sdf.farms / sdf.area

    print(f'dairy: ratio: {sdf[sdf.osize>2].shape[0]/sdf.shape[0]}')

    tdf = sdf.groupby('farms')['osize'].apply(lambda x: pd.DataFrame(
        {'outbreaks': [(x>=3).sum()], 'events': [x.shape[0]]}))
    tdf = tdf.reset_index().drop('level_1', axis=1)
    tdf = tdf.sort_values('farms', ascending=False)
    tdf[['outbreaks', 'events']] = tdf[['outbreaks', 'events']].cumsum()
    tdf = tdf.sort_values('farms')
    tdf['ratio'] = tdf.outbreaks / tdf.events

    # sdf = sdf[sdf.farms>=sdf.osize]
    nreps = sdf.osize.sum()
    nevents = tdf.events.head(1).values[0]
    noutbreaks = tdf.outbreaks.head(1).values[0]

    fig, gs = plot.initiate_figure(x=5*3, y=4*3, 
                                   la_title=f'Dairy: reports: {nreps}, events: {nevents}, outbreaks: {noutbreaks}, delta: {delta}', 
                                   la_fontsize='large',
                                   la_y=.93,
                                   gs_nrows=3, gs_ncols=3,
                                   gs_wspace=.2, gs_hspace=.4,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0],
                      func='sns.scatterplot', data=sdf, 
                      pf_x='farms', pf_y='osize',
                      sp_yscale='log', sp_xscale='log',
                      la_ylabel='outbreak size', la_xlabel='county farm count',
                      la_title='')
    ax = plot.subplot(fig=fig, grid=gs[0,1],
                      func='sns.scatterplot', data=sdf, 
                      pf_x='fd', pf_y='osize',
                      sp_yscale='log',
                      la_ylabel='outbreak size', la_xlabel='county farm density',
                      la_title='')
    ax = plot.subplot(fig=fig, grid=gs[0,2],
                      func='sns.scatterplot', data=sdf, 
                      pf_x='local_nbhd', pf_y='osize',
                      sp_yscale='log',
                      la_ylabel='outbreak size', 
                      la_xlabel='Average local neighborhood',
                      la_title='')
    ## ax = plot.subplot(fig=fig, grid=gs[0,1],
    ##                   func='sns.scatterplot', data=sdf, 
    ##                   pf_x='heads', pf_y='osize',
    ##                   la_ylabel='outbreak size', la_xlabel='county head count',
    ##                   la_title='')
    ax = plot.subplot(fig=fig, grid=gs[1,0],
                      func='sns.lineplot', data=tdf, 
                      pf_y='ratio', pf_x='farms',
                      sp_xscale='log',
                      la_ylabel='outbreaks/events', la_xlabel='county farm count',
                      la_title='Reverse cummulative sum')
    ax = plot.subplot(fig=fig, grid=gs[1,1],
                      func='sns.lineplot', data=tdf, 
                      pf_y='events', pf_x='farms',
                      la_ylabel='No. of events', la_xlabel='county farm count',
                      la_title='Reverse cummulative sum')
    plot.savefig(f'events_vs_farms_milk.pdf')

def type_events_vs_farms_poultry(type):
    preps = pd.read_csv('../results/poultry_event_clusters.csv').fillna(-1)
    dreps = pd.read_csv('../results/dairy_event_clusters.csv').fillna(-1)
    dreps['type'] = 'milk'
    dreps['production'] = -1
    reps = pd.concat([preps, dreps])
    reps = reps[reps.delta==15]
    # production is used just for count
    sdf = reps.groupby(['event0', 'type']).agg(
            {'county': 'first', 'production': 'count'})
    sdf = sdf.rename(columns={'production': 'osize'}).reset_index()
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    regions = regions.to_crs(epsg=32633)
    regions['area'] = regions.geometry.area / 2.59e6
    regions['county'] = regions.state_code*1000 + regions.county_code
    if type != 'milk':
        fndf = pd.read_parquet(
                '../intermediate_data/farm_neighborhood_poultry.parquet')
    else:
        fndf = pd.read_parquet(
                '../intermediate_data/farm_neighborhood_milk.parquet')

    farms = pd.concat([
        utils.load_poultry_farms(commercial=True, aggregate='county',
                                 commercial_threshold=50),
        utils.load_dairy_farms(aggregate='county', commercial=True,
                                     commercial_threshold=50)])
    farms = farms[['county', 'subtype', 'farms', 'heads']].rename(
            columns={'subtype': 'type'})

    sdf = sdf.merge(farms, on=['county', 'type'], how='left').fillna(0)

    if type == 'all':
        tdf = sdf[(sdf.type!='all') & (sdf.type!='milk')].groupby('county')[
                'farms'].sum().reset_index()
        sdf = sdf[sdf.type=='all']
        sdf = sdf.merge(tdf, on='county', suffixes=('_x', ''))
        sdf = sdf.drop('farms_x', axis=1)
        sdf = sdf[sdf.farms<=2000]
    else:
        sdf = sdf[sdf.type==type]
        fndf = fndf[(fndf.subtype==type) & (fndf.subtype_n==type)]

    tdf = fndf.groupby(['fid', 'county'])['fid_n'].count().reset_index()
    cfndf = tdf.groupby('county')['fid_n'].sum().reset_index()
    sdf = sdf.merge(cfndf, on='county')
    sdf = sdf.rename(columns={'fid_n': 'local_nbhd'})
    sdf.local_nbhd = sdf.local_nbhd / sdf.farms

    sdf = sdf.merge(regions[['county', 'area']], on='county')
    sdf['fd'] = sdf.farms / sdf.area

    print(f'{type}: ratio: {sdf[sdf.osize>2].shape[0]/sdf.shape[0]}')

    tdf = sdf.groupby('farms')['osize'].apply(lambda x: pd.DataFrame(
        {'outbreaks': [(x>=3).sum()], 'events': [x.shape[0]]}))
    tdf = tdf.reset_index().drop('level_1', axis=1)
    tdf = tdf.sort_values('farms', ascending=False)
    tdf[['outbreaks', 'events']] = tdf[['outbreaks', 'events']].cumsum()
    tdf = tdf.sort_values('farms')
    tdf['ratio'] = tdf.outbreaks / tdf.events

    # sdf = sdf[sdf.farms>=sdf.osize]
    nreps = sdf.osize.sum()
    nevents = tdf.events.head(1).values[0]
    noutbreaks = tdf.outbreaks.head(1).values[0]

    fig, gs = plot.initiate_figure(x=5*3, y=4*3, 
                                   la_title=f'{type}: {nreps}, {nevents}, {noutbreaks}', 
                                   la_fontsize='large',
                                   la_y=.93,
                                   gs_nrows=3, gs_ncols=3,
                                   gs_wspace=.2, gs_hspace=.4,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0],
                      func='sns.scatterplot', data=sdf, 
                      pf_x='farms', pf_y='osize',
                      la_ylabel='outbreak size', la_xlabel='county farm count',
                      la_title='')
    ax = plot.subplot(fig=fig, grid=gs[0,1],
                      func='sns.scatterplot', data=sdf, 
                      pf_x='fd', pf_y='osize',
                      la_ylabel='outbreak size', la_xlabel='county farm density',
                      la_title='')
    ax = plot.subplot(fig=fig, grid=gs[0,2],
                      func='sns.scatterplot', data=sdf, 
                      pf_x='local_nbhd', pf_y='osize',
                      la_ylabel='outbreak size', 
                      la_xlabel='Average local neighborhood',
                      la_title='')
    ## ax = plot.subplot(fig=fig, grid=gs[0,1],
    ##                   func='sns.scatterplot', data=sdf, 
    ##                   pf_x='heads', pf_y='osize',
    ##                   la_ylabel='outbreak size', la_xlabel='county head count',
    ##                   la_title='')
    ax = plot.subplot(fig=fig, grid=gs[1,0],
                      func='sns.lineplot', data=tdf, 
                      pf_y='ratio', pf_x='farms',
                      la_ylabel='outbreaks/events', la_xlabel='county farm count',
                      la_title='Reverse cummulative sum')
    ax = plot.subplot(fig=fig, grid=gs[1,1],
                      func='sns.lineplot', data=tdf, 
                      pf_y='events', pf_x='farms',
                      la_ylabel='No. of events', la_xlabel='county farm count',
                      la_title='Reverse cummulative sum')
    plot.savefig(f'events_vs_farms_{type}.pdf')
def _type_conditional_risk(events, dates, model):
    events["present"] = 1  # Assign 1 for presence
    binary_matrix = events.pivot_table(index="county_code", columns="ym", 
                                       values="present", 
                                       aggfunc="max").fillna(0)
    binary_matrix = binary_matrix.reindex(columns=dates, 
                                          fill_value=0).astype(int)
    set_trace()
    conditional_risk = binary_matrix.apply(_compute_post_probs, axis=1, model=model, 
                                           result_type='expand')
    conditional_risk.columns = binary_matrix.columns
    return conditional_risk

def _compute_post_probs(row, model=None):
    return model.predict_proba(row.values.reshape(-1,1)) @ model.emissionprob_[:,1]

@cli.command()
def events_duration():
    reps = pd.read_csv('../results/poultry_spillover_events.csv').fillna(-1)
    reps = reps[(reps.delta==30) & (reps.type!='all')]
    reps.confirmed = pd.to_datetime(reps.confirmed)

    reps = reps.sort_values('confirmed')

    ddf = reps.groupby(['event0', 'type'], as_index=False)['confirmed'].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]).days)


    reps
    set_trace()

@cli.command()
def conditional_risk():
    cdf = pd.read_csv('../intermediate_data/poultry_baseline_conditional_risk.csv')
    fig, gs = plot.initiate_figure(x=5, y=4*2, 
                                   gs_nrows=2, gs_ncols=1,
                                   gs_wspace=-.1, gs_hspace=.8,
                                   color='tableau10')

    # same county
    cdf['probability'] = cdf.instances / cdf.total
    tdf = cdf[cdf.hop==0].pivot(index='source', columns='target', values='probability')
    ax = plot.subplot(fig=fig, grid=gs[0,0], 
                      func='sns.heatmap', data=tdf, pf_annot=True,
                      xt_rotation=40,
                      la_title='Same county')
    tdf = cdf[cdf.hop==1].pivot(index='source', columns='target', values='probability')
    ax = plot.subplot(fig=fig, grid=gs[1,0], 
                      func='sns.heatmap', data=tdf, pf_annot=True,
                      xt_rotation=40,
                      la_title='Adjacent county')
    plot.savefig('poultry_conditional_risk.pdf')
def baseline_conditional_risk_model():
    # reps = pd.read_csv('../results/poultry_spillover_events.csv').fillna(-1)
    reps = utils.h5n1_poultry()
    reps = reps[reps.year==2022]

    reps = reps[~reps.type.isnull()]
    reps.county = reps.state_code*1000 + reps.county_code
    reps.confirmed = pd.to_datetime(reps.confirmed)
    reps = reps.sort_values('confirmed').reset_index(drop=True)

    cn = pd.read_parquet(COUNTY_NEIGHBORS)
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y

    types = reps.type.drop_duplicates().to_list()
    types.sort()
    cdf = pd.DataFrame(product(types, types, [0,1]), columns=['source', 'target', 'hop'])
    cdf['instances'] = 0

    for i,row in reps.iterrows():
        tdf = reps.iloc[i+1:]
        tdf['timediff'] = (tdf.confirmed - row.confirmed).dt.days
        neighbors = cn[cn.county_x==row.county].county_y.tolist()
        tdf = tdf[tdf.timediff<=30]

        # same county
        for t in tdf[tdf.county==row.county].type.drop_duplicates().tolist():
            ind = (cdf.source==row.type) & (cdf.target==t) & (cdf.hop==0)
            cdf.loc[ind, 'instances'] = cdf.loc[ind, 'instances'] + 1

        # neighboring counties
        for t in tdf[(tdf.county!=row.county) & 
                     (tdf.county.isin(neighbors))].type.drop_duplicates().tolist():
            ind = (cdf.source==row.type) & (cdf.target==t) & (cdf.hop==1)
            cdf.loc[ind, 'instances'] = cdf.loc[ind, 'instances'] + 1

    # Normalize by number of events per type to get a notion of probability
    rc = reps.groupby('type')['county'].count().reset_index()
    rc = rc[rc.type.isin(types)].sort_values('type')
    cdf = cdf.merge(rc, left_on='source', right_on='type')
    cdf = cdf.rename(columns={'county': 'total'})
    cdf = cdf.drop('type', axis=1)
    set_trace()

    ## cdf = cdf.div(rc.county.to_numpy(), axis=0)
    ## cndf = cndf.div(rc.county.to_numpy(), axis=0)

    cdf.to_csv('poultry_baseline_conditional_risk.csv', index=False)
@cli.command()
@click.option('--commercial_threshold', default=100, help="commercial threshold")
def old_farm_neighborhood(commercial_threshold):
    nmdf = pd.read_parquet('../intermediate_data/glw_moore_10.parquet') # obtained from neighborhood_graph.py
    cells = nmdf[['x', 'y']].drop_duplicates()
    cells[['x_', 'y_']] = cells[['x', 'y']]
    cells['dist'] = 1
    nmdf = pd.concat([nmdf, cells])

    # farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    farms = load_poultry_farms()
    farms = farms[(farms.heads>=commercial_threshold) & (farms.subtype!='all')]

    # farm count by county
    fc = farms.groupby(['state_code', 'county_code', 'subtype'], as_index=False
                       )['fid'].count()
    fc = fc.rename(columns={'fid': 'fc'})

    # first aggregate farms by x,y,subtype
    cdf = farms.groupby(['x', 'y', 'subtype']).agg(
            {'state_code': 'first', 'county_code': 'first',
             'fid': 'count'}).reset_index()
    cdf = cdf.rename(columns={'fid': 'farms'})

    # merge with farms by neighborhood
    fdf = farms[['fid', 'x', 'y', 'state_code', 'county_code', 'subtype']].merge(
            nmdf, on=['x', 'y'], how='left')
    fdf = fdf.merge(cdf, left_on=['x_', 'y_'], right_on=['x', 'y'],
                       suffixes=('', '_n'))
    fdf = fdf.drop(['x', 'y', 'x_n', 'y_n', 'x_', 'y_'], axis=1)

    # group by county
    fdf['1_by_d'] = fdf.farms / fdf.dist
    fdf['1_by_d2'] = fdf.farms / fdf.dist**2
    df = fdf.groupby(['fid', 'subtype', 'state_code', 'county_code', 
                      'state_code_n', 'county_code_n'])[
            ['1_by_d', '1_by_d2']].sum().reset_index()

    # merge with fc
    df = df.merge(fc, left_on=['state_code_n', 'county_code_n', 'subtype'],
                  right_on=['state_code', 'county_code', 'subtype'], 
                  suffixes=('','_x'))

    # removing the farm itself from the term
    ind = (df.state_code==df.state_code_n) & (df.county_code==df.county_code_n) \
            & (df.subtype==df.subtype_n)
    df.loc[ind, '1_by_d'] = df[ind]['1_by_d'] - 1
    df.loc[ind, '1_by_d2'] = df[ind]['1_by_d2'] - 1
    df.loc[ind, 'fc'] = df[ind].fc - 1
    df = df.drop(['state_code', 'county_code', 'state_code_x', 'county_code_x'], 
                 axis=1)
    df = df.rename(columns={'state_code_n': 'state_code', 
                            'county_code_n': 'county_code'})

def bird_incidence_weighted(df, h5n1_ind, year):
    h5n1_ind = h5n1_ind[h5n1_ind.year==year].copy()
    h5n1_ind['ind'] = True
    thdf = h5n1_ind[['state_code', 'county_code']].pivot
    set_trace()
    for q in [1,2,3,4]:
        if q == 1:
            year_m_1 = year - 1
            q_m_1 = 4
        else:
            year_m_1 = year
            q_m_1 = q - 1
        tind = h5n1_ind[(h5n1_ind.year==year) & (h5n1_ind.quarter==q)]
        tind_m_1 = h5n1_ind[(h5n1_ind.year==year_m_1) & (h5n1_ind.quarter==q_m_1)]
        df = df.merge(tind[['x', 'y', 'incidences']], on=['x', 'y'], how='left')
        df = df.merge(tind_m_1[['x', 'y', 'incidences']], on=['x', 'y'], 
                      how='left')
        df[f'birds{q}_W'] = df[f'birds{q}_W'] * (~df.incidences_x.isnull())
        df[f'birds{q}_W_m_1'] = df[f'birds{q_m_1}_W'] * (~df.incidences_y.isnull())
        df = df.drop(['incidences_x', 'incidences_y'], axis=1)
    df['year'] = year
    return df
    # likelihood function
    hdf = h5n1_poultry.groupby(['statefp', 'countyfp', 'year']).agg(
            gt_inf_farms=('value', 'count'),
            gt_inf_heads=('value', 'sum')).reset_index()

    ddf = df.groupby(['state_code', 'county_code', 'year']).agg(
            risk=('risk', 'sum'),
            inf_farms=('infected_farms', 'sum'),
            inf_heads=('risk', 'sum')).reset_index()

    edf = ddf.merge(
            hdf, 
            left_on=['state_code', 'county_code', 'year'],
            right_on=['statefp', 'countyfp', 'year'],
            how='right')

    edf = edf[~edf.inf_farms.isnull()]

    eval['inf_farms'] = abs(edf.gt_inf_farms - edf.inf_farms).sum() / \
            edf.gt_inf_farms.sum()
            
    tedf = edf[edf.gt_inf_heads!=-1]
    eval['inf_heads'] = abs(edf.gt_inf_heads - edf.inf_heads).sum() / \
            edf.inf_heads.sum()

    ## hdf = h5n1_poultry[h5n1_poultry.value>=1000][['statefp', 'countyfp']
    ##         ].value_counts().reset_index()

    ## ddf = df[df.size_category.isin(['l', 'vl'])].groupby(
    ##     ['state_code', 'county_code']
    ##     )['infected_farms'].sum().reset_index()

    # select all instances in 
    ## edf = ddf.merge(hdf[['statefp', 'countyfp', 'quarter']], 
    ##         left_on=['state_code', 'county_code', 'quarter'],
    ##         right_on=['statefp', 'countyfp', 'quarter'],
    ##         indicator=True)
    ## edf = edf[edf._merge=='both']
    ## edf = edf.merge(
    ##         hdf, 
    ##         on=['statefp', 'countyfp', 'size_category', 'quarter'],
    ##         how='left')
    ## edf['count'] = edf['count'].fillna(0)
    ## eval = mean_squared_error(edf.inf_heads, edf.gt_inf_heads) / \
    ##         edf.gt_inf_heads.sum()
    ## eval = 1-pearsonr(edf.inf_heads, edf.gt_inf_heads).statistic
def risk_agg_score(df, par):

    df.size_category = df.size_category.astype(str)
    cols = ['risk_prob' + str(i) for i in range(1,5)]
   
    # risk
    riskdf = df[cols].multiply(df.heads, axis=0)
    riskdf['state_code'] = df.state_code
    riskdf['county_code'] = df.county_code
    riskdf['size_category'] = df.size_category
    sum_risk = riskdf.groupby(['state_code', 'county_code', 'size_category']
                              )[cols].sum()
    sum_risk.columns = [1,2,3,4]

    srdf = sum_risk.reset_index().melt(
            id_vars=['state_code', 'county_code', 'size_category'], 
            var_name='quarter', value_name='risk')

    # farms infected
    farmsdf = df[['size_category', 'state_code', 'county_code'] + cols]
    expected_farms = farmsdf.groupby(['state_code', 'county_code', 
                                      'size_category']).sum()
    expected_farms.columns = [1,2,3,4]

    sedf = expected_farms.reset_index().melt(id_vars=[
        'state_code', 'county_code', 'size_category'], var_name='quarter', 
                                             value_name='infected_farms')
    set_trace()

    odf = srdf.merge(sedf, on=['state_code', 'county_code', 'size_category', 
                               'quarter'])

    for k,v in par.items():
        odf[k] = v

    return odf

def poultry():
    df = pd.read_csv('../../data/h5n1/poultry.csv')
    df['year'] = pd.to_datetime(df.confirmed).dt.year
    df = df.sort_values(['year', 'quarter'])
    ## fq = df.groupby('state')[['year', 'quarter']].first().reset_index()
    ## df = df.merge(fq, on=['state', 'year', 'quarter'], how='right')
    ## h5n1 = df.groupby(['state', 'county', 'statefp', 'countyfp', 'quarter'],
    ##            as_index=False)['value'].sum()

    df['size_category'] = pd.cut(df.value, bins=FARM_SIZE, 
            labels=FARM_SIZE_NAMES, right=False)
    df.size_category = df.size_category.astype('str')
    df.to_parquet('h5n1_poultry.parquet')
    pdf = df.groupby(['statefp', 'countyfp', 'year', 'quarter'])['value'].agg(
            ['sum', 'count']).reset_index()
    pdf = pdf.rename(columns={'sum': 'heads', 'count': 'farms'})
    pdf = pdf[pdf.farms!=0]
    return pdf

def poultry_ranked_by_bird_incidence():

    df = pd.read_parquet(
            '../intermediate_data/county_buffered_counts.parquet')

    # h5n1
    pdf = poultry()
    pdf = pdf.merge(df, right_on=['state_code', 'county_code'], 
                    left_on=['statefp', 'countyfp'], how='left')
    pdf = pdf[~pdf.farms_y.isnull()]
    set_trace()

def poultry_incidence_county_buffered():

    # loading distance-weighted
    ## df = pd.read_parquet(
    ##         '../intermediate_data/distance_weighted_farm_counts.parquet')
    df = pd.read_parquet(
            '../intermediate_data/county_buffered_counts.parquet')

    # h5n1
    pdf = poultry()
    pdf = pdf.merge(df, right_on=['state_code', 'county_code'], 
                    left_on=['statefp', 'countyfp'], how='left')
    pdf = pdf[~pdf.farms_y.isnull()]
    set_trace()

    sum_wt_infected = df.groupby('state_code', as_index=False
                                 )[cols + ['all']].apply(infected_farm)
    sum_wt_infected.columns = ['state_code',1,2,3,4]
    swdf = sum_wt_infected.melt(id_vars='state_code', var_name='quarter', 
                                value_name='likelihood')


def infected_farm(df):
    pop = df['all']
    df = df.drop('all', axis=1)
    set_trace()
    df = (df.div(pop, axis=0) * 100).map(trapezoidal)
    return df.sum()

def trapezoidal(x):
    return np.piecewise(x, 
                        [x<0, (x>=0) & (x<5), (x>=5) & (x<15), 
                         (x>=15) & (x<20), x>=20],
                        [0, lambda x: (x-0)/(5-0)*1, 
                         1, lambda x: (20-x)/(20-15)*1, 0])

