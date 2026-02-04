DESC='''
Analyze population subset relevant to agriculture as occupation.
'''

import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import seaborn as sns

import plot
import utils_pums

FORMAT="%(levelname)s:%(funcName)s:%(message)s"
SHAPES = 'zip:///Users/abhijin/github/AgAid_digital_twin/pums/data/puma_wa_shapes.zip!wa_puma/wa_puma.shp'
WA_FIPS = '53'

def heatmap(ax, df, title):
    to_plot = gpd.GeoDataFrame(df[
        ['puma', 'serialno', 'geometry']].groupby(
           'puma').aggregate({'serialno': 'count', 'geometry': 'first'}))
    to_plot.plot(column='serialno', 
            legend=True,
            legend_kwds={'shrink': 0.5},
            ax=ax)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def main():

    logging.basicConfig(level=logging.INFO,format=FORMAT)

    person = utils_pums.select_ag()
    puma_shapes = gpd.read_file(SHAPES)

    # Geographic distribution
    puma_shapes = puma_shapes.astype({'PUMA': int})
    person = person.merge(puma_shapes,
           left_on='puma',
           right_on='PUMA')
    # plot
    fig = plot.initiate_plot(10,6)
    gs = plot.subplot_grids(fig,2,2)

    # total
    df = person
    heatmap(fig.add_subplot(gs[0,0]), df, f'Total ($n={person.shape[0]}$)')

    # occp
    df = person[~person.occp.isnull()]
    heatmap(fig.add_subplot(gs[0,1]), df, f'Occupation (OCCP) ({df.shape[0]})')

    # indp
    df = person[~person.indp.isnull()]
    heatmap(fig.add_subplot(gs[1,0]), df, f'Industry (INDP) ({df.shape[0]})')

    # fod1p
    df = person[~person.fod1p.isnull()]
    heatmap(fig.add_subplot(gs[1,1]), df, f'Field of degree (FOD1P) ({df.shape[0]})')


    plt.savefig('people_by_puma.pdf', bbox_inches='tight')


if __name__== "__main__":
    main()
