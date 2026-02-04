DESC='''plot functions
By AA
'''

# A note on arguments
##  sp = subplot_args
##  pf = plot_args
##  ag = axes_grid_args
##  la = label_args
##  fs = fontsize_args
##  xt = xtick_args
##  yt = ytick_args
##  lg = legend_args

from cycler import cycler
from itertools import product
import logging
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc, patches
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub, search, match
from scipy.interpolate import griddata
import seaborn as sns
try:
    import geopandas as gpd
    from shapely.geometry import Point
except:
    print('Warning: GIS modules not loaded.')

# contextily EPSG 3857

COLORS = {
        'mathematica': ['#5e82b5','#e09c24','#8fb030','#eb634f','#8778b3','#c46e1a','#5c9ec7','#fdbf6f'],
        'tableau10': ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac'],
        'tableau20':['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
                     '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'],
        'not_red' :['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
                     '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'],
        'grand_budapest': ['#5b1a18','#fd6467','#f1bb7b','#d67236'],
        'red_blue': ['#0060ad', '#dd181f'],
        'datanovia': ["#FFDB6D", "#C4961A", "#F4EDCA", "#D16103", "#C3D7A4", 
            "#52854C", "#4E84C4", "#293352"],
        'cbYlOrRd': ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a',
                     '#e31a1c','#b10026']
        }
SNS_AXIS_PLOTS = ['sns.lineplot', 'sns.barplot', 'sns.histplot','sns.kdeplot',  'sns.countplot', 
                  'sns.ecdfplot', 'sns.boxplot', 'sns.violinplot', 'sns.heatmap', 
                  'sns.scatterplot', 'contour']
AXIS_NORMAL = ['sns.lineplot', 'sns.ecdfplot', 'sns.scatterplot', 'contour']
AXIS_HEAT = ['sns.heatmap']
AXIS_HIST = ['sns.barplot', 'sns.histplot','sns.kdeplot',  'sns.countplot']
AXIS_BOX = ['sns.boxplot', 'sns.violinplot']
AXIS_CHOROPLETH = ['gpd.plot', 'gpd.boundary.plot']
AXIS_NONE = ['pie']
NON_SNS = ['hlines', 'vlines', 'text', 'lines', 'arrow', 'pie']

NON_FUNC_PARAMS = ['fig', 'subplot', 'title', 'xlabel', 'ylabel', 'data']
HATCH = ['++', 'xx', '\\', '.', 'o', '|', '*']

RC_PARAMS = {
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'legend.frameon': True,
        'legend.framealpha': .7,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.borderpad': .1,
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'text.usetex': True,
        'text.latex.preamble': '',
        'axes.prop_cycle': cycler(color=COLORS['mathematica']),
        'axes.formatter.limits': [-5,6] # important for scientific notation
        }

DEFAULT_FONTS = {
        'fontsize': 17,
        'title': 'large',
        'xlabel': 'normalsize',
        'ylabel': 'normalsize',
        'xtick': 'small',
        'ytick': 'small',
        'legend': 'normalsize',
        'legend_title': 'normalsize',
        'colorbar': 'normalsize',
        }


# latex fontsizes
FONT_TABLE=pd.DataFrame({9: {'miniscule': 4, 'tiny': 5, 'scriptsize': 6, 'footnotesize': 7, 'small': 8, 'normalsize': 9, 'large': 10, 'Large': 11, 'LARGE': 12, 'huge': 14, 'Huge': 17, 'HUGE': 20}, 10: {'miniscule': 5, 'tiny': 6, 'scriptsize': 7, 'footnotesize': 8, 'small': 9, 'normalsize': 10, 'large': 11, 'Large': 12, 'LARGE': 14, 'huge': 17, 'Huge': 20, 'HUGE': 25}, 11: {'miniscule': 6, 'tiny': 7, 'scriptsize': 8, 'footnotesize': 9, 'small': 10, 'normalsize': 11, 'large': 12, 'Large': 14, 'LARGE': 17, 'huge': 20, 'Huge': 25, 'HUGE': 30}, 12: {'miniscule': 7, 'tiny': 8, 'scriptsize': 9, 'footnotesize': 10, 'small': 11, 'normalsize': 12, 'large': 14, 'Large': 17, 'LARGE': 20, 'huge': 25, 'Huge': 30, 'HUGE': 36}, 14: {'miniscule': 8, 'tiny': 9, 'scriptsize': 10, 'footnotesize': 11, 'small': 12, 'normalsize': 14, 'large': 17, 'Large': 20, 'LARGE': 25, 'huge': 30, 'Huge': 36, 'HUGE': 48}, 17: {'miniscule': 9, 'tiny': 10, 'scriptsize': 11, 'footnotesize': 12, 'small': 14, 'normalsize': 17, 'large': 20, 'Large': 25, 'LARGE': 30, 'huge': 36, 'Huge': 48, 'HUGE': 60}, 20: {'miniscule': 10, 'tiny': 11, 'scriptsize': 12, 'footnotesize': 14, 'small': 17, 'normalsize': 20, 'large': 25, 'Large': 30, 'LARGE': 36, 'huge': 48, 'Huge': 60, 'HUGE': 72}, 25: {'miniscule': 11, 'tiny': 12, 'scriptsize': 14, 'footnotesize': 17, 'small': 20, 'normalsize': 25, 'large': 30, 'Large': 36, 'LARGE': 48, 'huge': 60, 'Huge': 72, 'HUGE': 84}, 30: {'miniscule': 12, 'tiny': 14, 'scriptsize': 17, 'footnotesize': 20, 'small': 25, 'normalsize': 30, 'large': 36, 'Large': 48, 'LARGE': 60, 'huge': 72, 'Huge': 84, 'HUGE': 96}, 36: {'miniscule': 14, 'tiny': 17, 'scriptsize': 20, 'footnotesize': 25, 'small': 30, 'normalsize': 36, 'large': 48, 'Large': 60, 'LARGE': 72, 'huge': 84, 'Huge': 96, 'HUGE': 108}, 48: {'miniscule': 17, 'tiny': 20, 'scriptsize': 25, 'footnotesize': 30, 'small': 36, 'normalsize': 48, 'large': 60, 'Large': 72, 'LARGE': 84, 'huge': 96, 'Huge': 108, 'HUGE': 120}, 60: {'miniscule': 20, 'tiny': 25, 'scriptsize': 30, 'footnotesize': 36, 'small': 48, 'normalsize': 60, 'large': 72, 'Large': 84, 'LARGE': 96, 'huge': 108, 'Huge': 120, 'HUGE': 132}})

AXES_COLOR='#999999'
GRID_COLOR='#cccccc'
TICKS_COLOR='#222222'

MAJOR_TICK_LINEWIDTH = .75
MINOR_TICK_LINEWIDTH = .25

FIGSIZE=(8,6)
LINESTYLES=['solid','dashed']
LINEWIDTH=[1]
MARKERS=['.','o','v','^','s']
# LINE_PROPS=pd.DataFrame(product(['solid','dashed'],MARKERS,MATHEMATICA[0:7],LINEWIDTH),columns=['style','marker','color','width'])

pd.options.display.float_format = '{:.10g}'.format

class RasterPlot:
    def __init__(self, rows=None, cols=None, total=None):
        self.pid = 0
        if total:
            self.num_cols = int(np.ceil(np.sqrt(total)))
            self.num_rows = int(np.ceil(np.sqrt(total)))
            self.num_plots = total
        elif rows and cols:
            self.num_cols = cols
            self.num_rows = rows
            self.num_plots = cols * rows
        else:
            raise ValueError('Either explicitly provide (rows=* and cols=*) or (total=*)')

    def rows_cols(self):
        return self.num_rows, self.num_cols

    def new(self):
        self.pid += 1
        if self.pid > self.num_plots:
            raise ValueError('Number of plots exceeds specified number.')
        return (self.pid-1) // self.num_cols, (self.pid-1) % self.num_cols

# The helper functions are arranged in the order in which they should be called
# See argvals for all possible arguments.
def initiate_figure(**kwargs):
    argvals = {
            'mode': 'figure',
            'subplot_mode': 'gridspec',
            'x': FIGSIZE[0],
            'y': FIGSIZE[1],
            'gs_wspace': 0.2,
            'gs_hspace': 0.2,
            'gs_nrows': 1,
            'gs_ncols': 1,
            'color': 'mathematica', 
            'st_y': 0.95,
            'st_fontsize': 'large'
            }

    for k in kwargs.keys():
        argvals[k] = kwargs[k] 

    # Setting font families and latex stuff
    # AA: need to be able to modify this too at some point
    for k,v in RC_PARAMS.items():
        rcParams[k] = v

    # Setting color
    if 'color' in kwargs.keys():
        rcParams['axes.prop_cycle'] = cycler(color=COLORS[kwargs['color']])
        sns.set_palette(COLORS[kwargs['color']])

    # Setting scientific notation limits
    if 'scilimits' in kwargs.keys():
        rcParams['axes.formatter.limits'] = kwargs['scilimits']

    # Setting default font sizes
    fs_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'fs_'}
    for k,v in fs_args.items():
        DEFAULT_FONTS[k] = v

    # Figure
    if argvals['mode'] == 'figure':
        fig_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'fg_'}
        if argvals['subplot_mode'] == 'gridspec':
            fig = plt.figure(figsize=[argvals['x'],argvals['y']], **fig_args)
            subplot_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'gs_'}
            gs = fig.add_gridspec(subplot_args['nrows'], subplot_args['ncols'])
            gs.update(wspace=subplot_args['wspace'], 
                    hspace=subplot_args['hspace']) # set the spacing between axes.
            # AA: Under construction
            # Idea is to organize it exactly like subplots for uniformity
            label_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'la_'}
            if 'title' in label_args.keys():
                suptitle_args = {k[3:]: v for k,v in argvals.items() 
                                 if k[0:3] == 'la_'}
                suptitle_args['fontsize'] = FONT_TABLE[DEFAULT_FONTS[
                    'fontsize']][suptitle_args['fontsize']]
                fig.suptitle(kwargs['suptitle'], **suptitle_args)

            return fig, gs
        # AA: not used currently
        elif argvals['subplot_mode'] == 'subplots':
            raise('Not supported. Some issues.')
            subplot_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'sp_'}
            fig, ax = plt.subplots(**subplot_args, figsize=[argvals['x'],argvals['y']])
            if type(ax) != np.ndarray:
                return fig, [ax]
            else:
                return fig, ax
    elif mode == 'facetgrid':
        return  # currently, nothing to return

def subplot(**kwargs):

    try:
        logging.info(kwargs['la_title'])
    except KeyError:
        logging.info('Empty title')

    # Collect all arguments that pertain to func
    for k in kwargs.keys():
        if k in ['fig', 'ax', 'func', 'data', 'text', 'hatch', 'sharey', 'sharex', 'grid']:
            continue
        try:
            pat = search('[^_]*_', k)[0]
        except TypeError:
            raise ValueError(f'"{k}": Likely forgot a prefix.')
        if pat not in ['sp_', 'pf_', 'ag_', 'la_', 'fs_', 'xt_', 'yt_', 'lg_', 'tx_', 'cb_']:
            raise ValueError(f'Unsupported input class "{k}".')
    subplot_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'sp_'}
    plot_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'pf_'}
    axes_grid_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'ag_'}
    label_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'la_'}
    fontsize_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'fs_'}
    xtick_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'xt_'}
    ytick_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'yt_'}
    legend_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'lg_'}
    colorbar_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'cb_'}

    fig = kwargs['fig']
    func = kwargs['func']

    # Create subplot if not exists
    if 'ax' in kwargs.keys():   # axis given
        ax = kwargs['ax']
    else:
        axis_list = fig.axes
        try:
            if 'sharex' in subplot_args.keys():
                subplot_args['sharex'] = axis_list[subplot_args['sharex']]
            if 'sharey' in subplot_args.keys():
                subplot_args['sharey'] = axis_list[subplot_args['sharey']]
        except IndexError:
            print('Ignoring "sharex/sharey" ...')
            for s in ['sharex', 'sharey']:
                try:
                    del subplot_args[s]
                except:
                    pass

        # Some custom fields need to be removed before creating ax. They will
        # be added back.
        subplot_args_ = {}
        for k in subplot_args.keys():
            if k in ['xlim', 'ylim']:
                continue
            subplot_args_[k] = subplot_args[k]
        ax = fig.add_subplot(kwargs['grid'], **subplot_args_)
        kwargs['ax'] = ax

    # Decide function
    try:
        plot_args['data'] = kwargs['data']
    except KeyError:
        print('WARNING: "data" field absent.')
    funcobj = subplot_func(plot_args=plot_args, **kwargs)

    func_res = funcobj(ax=ax, **plot_args)

    # Hatching
    try:
        if kwargs['hatch']:
            ax = subplot_hatch(ax=ax, func=func)
    except KeyError:
        pass

    subplot_axes_grid(func=func, ax=ax, plot_args=plot_args, 
            subplot_args=subplot_args, **axes_grid_args)
    subplot_labels(ax=ax, **label_args)
    fontsizes = subplot_fonts(ax=ax, func=func, **fontsize_args)

    if xtick_args:
        if 'setticks' in xtick_args.keys():
            try:
                ax.set_xticks(ax.get_xticks(), labels=xtick_args['setticks'])
                print('Warning: renaming xtick labels did not work.')
            except:
                ax.set_xticks(xtick_args['setticks'])
            del xtick_args['setticks']
        ax.tick_params(axis='x', **xtick_args)
    if ytick_args:
        ax.tick_params(axis='y', **ytick_args)

    set_legend(ax, legend_args, FONT_TABLE[fontsizes['fontsize']], 
            fontsize_args)

    if colorbar_args:
        colorbar(func_res, ax, FONT_TABLE[fontsizes['fontsize']], colorbar_args)
    return ax

def set_legend(ax, legend_args, fonts_table, fontsize_args):
    # Only if a legend is present and there are some modifications to be made
    try:
        legend_visible = legend_args['visible']
        legend_args.pop('visible')
    except KeyError:
        legend_visible = True

    try:
        if not legend_args['title']:
            ax.get_legend().set_title('')
        else:
            ax.get_legend().set_title(legend_args['title'])
    except:
        pass

    # set fonts
    legend_fonts = {}
    for type in ['legend', 'legend_title']:
        if type in fontsize_args.keys():
            legend_fonts[type] = fontsize_args[type]
        else:
            legend_fonts[type] = DEFAULT_FONTS[type]

    legend_handles = ax.get_legend_handles_labels()[0]
    if len(legend_handles):
        if legend_visible:
            ax.legend(**legend_args, 
                    fontsize=fonts_table[legend_fonts['legend']], 
                    title_fontsize=fonts_table[legend_fonts['legend_title']])
        else:
            ax.legend().set_visible(False)
    else: 
        try:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(fonts_table[legend_fonts['legend_title']])
        except:
            pass

def set_legend_invisible(ax):
    ax.saved_legend_handles = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    return

def separate_legend(fig=None, axes=None, labels=None, **kwargs):
    handles = []
    derived_labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.saved_legend_handles
        handles = handles + ax_handles
        derived_labels = derived_labels + ax_labels

    if labels == None:
        labels = derived_labels

    if 'fontsize' in kwargs.keys():
        kwargs['fontsize'] = FONT_TABLE[DEFAULT_FONTS['fontsize']][
                kwargs['fontsize']]

    fig.legend(handles, labels, **kwargs)

def subplot_func(**kwargs):
    funcname = kwargs['func']
    plot_args = kwargs['plot_args']
    ax = kwargs['ax']
    try:
        plot_args['data'] = kwargs['data']
    except KeyError:
        pass
    if funcname in ['gpd.plot', 'gpd.boundary.plot']:
        plot_args.pop('data')
        func = eval(f'kwargs["data"].plot')
    elif funcname in SNS_AXIS_PLOTS:
        if funcname in ['sns.barplot', 'sns.histplot', 'sns.countplot']:
            argvals = {
                    'alpha': 1,
                    'edgecolor': 'white'
                    }
            if funcname == 'sns.histplot':
                try:
                    if plot_args['element'] == 'step':
                        argvals = {'alpha': 1}
                except KeyError:
                    pass



        elif funcname in AXIS_BOX:
            argvals = {
                    'boxprops': {'edgecolor': 'white'},
                    'whiskerprops': {'color': TICKS_COLOR},
                    }
        elif funcname in AXIS_NORMAL:
            argvals = {}
        elif funcname in AXIS_HEAT:
            argvals = {}
        elif funcname in AXIS_CHOROPLETH:
            argvals = {}
        elif funcname == 'sns.kdeplot':
            argvals = {}
        else:
            raise ValueError(f'The plot "{funcname}" is not supported.')

        for k,v in argvals.items():
            if k not in plot_args.keys():
                plot_args[k] = v

        func = eval(funcname)
    elif funcname in NON_SNS:
        func = eval(funcname)
    else:
        raise ValueError(f'Unsupported function type {kwargs["func"]}.')
    return func

def subplot_hatch(**kwargs):
    ax = kwargs['ax']
    func = kwargs['func']
    if func in AXIS_HIST:
        for bars, h in zip(ax.containers, cycler(hatch=HATCH)()):
            for bar in bars:
                bar.set_hatch(h['hatch'])
    # For box plots, this is tricky
    if func in AXIS_BOX:
        pp = [patch for patch in ax.patches if 
                type(patch) == patches.PathPatch]
        huelist = []
        for patch in pp:
            huelist.append(patch.get_facecolor())
        hues = pd.DataFrame.from_records(huelist).drop_duplicates().reset_index(
                drop=True)

        for patch in pp:
            col = patch.get_facecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set_hatch(HATCH[hue])
            edgecol = patch.get_edgecolor() # assumes same edgecolor for all

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            col = patch.get_facecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set_edgecolor(edgecol)
            patch.set_hatch(HATCH[hue])

        #ax.legend(handles=for_legend)

    return ax


# Set axes and grid
# axis_type is the main argument. If it is not defined, the axis_type is set
# by the plot function type. If a plot function is not considered, then a 
# default value is used.
# To minimize arguments passed, we use an encoding for axis_type.
# axy|gxy|mxy
# a: axis, g: grid, m: minor
# ag_axis_type='axy:gxy:m'
def subplot_axes_grid(**kwargs):
    ax = kwargs['ax']
    axis_type = 'axy:gxy:mxy'
    if 'axis_type' in kwargs.keys():
        axis_type = kwargs['axis_type']
    elif kwargs['func'] in AXIS_CHOROPLETH + AXIS_NONE:
        axis_type = 'a:g:m'
    elif kwargs['func'] in AXIS_HEAT:
        axis_type = 'a:gxy:m'
    elif kwargs['func'] in AXIS_NORMAL:
        axis_type = 'axy:gxy:mxy'
    elif kwargs['func'] in AXIS_HIST or kwargs['func'] in AXIS_BOX:
        if 'orient' in kwargs['plot_args']:
            if kwargs['plot_args']['orient'] == 'h':
                axis_type = 'a:gx:m'
            else:
                axis_type = 'a:gy:m'
        else:
            axis_type = 'a:gy:m'
    elif kwargs['func'] in ['hlines', 'lines', 'arrow', 'text']:
        axis_type = 'a:g:m'

    if axis_type == 'ignore':
        return

    seg = axis_type.split(':')
    err = 'Wrong encoding of axis type: required "axy:gxy:mxy".'
    axis_x = True
    axis_y = True
    grid_x = True
    grid_y = True
    minor_x = True
    minor_y = True

    for a in seg:
        if len(a) > 3 or a[0] not in ['a', 'g', 'm']:
            raise ValueError(err)
        if a[0] == 'a':
            if 'x' not in a: axis_x = False
            if 'y' not in a: axis_y = False
        if a[0] == 'g':
            if 'x' not in a: grid_x = False
            if 'y' not in a: grid_y = False
        if a[0] == 'm':
            if 'x' not in a: minor_x = False
            if 'y' not in a: minor_y = False

    # Default values
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color(AXES_COLOR)
    ax.spines['left'].set_color(AXES_COLOR)

    if not axis_x:
        ax.spines['bottom'].set_visible(False)
    if not axis_y:
        ax.spines['left'].set_visible(False)

    if grid_y and grid_x:
        ax.grid(color=GRID_COLOR, which='major', linewidth=1)
    elif grid_y and not grid_x:
        ax.grid(axis='y', color=GRID_COLOR, which='major', 
                linewidth=MAJOR_TICK_LINEWIDTH)
    elif grid_x and not grid_y:
        ax.grid(axis='x', color=GRID_COLOR, which='major', 
                linewidth=MAJOR_TICK_LINEWIDTH)
    else:
        pass

    # AA: pending
    if minor_x or minor_y:
        ax.grid(True,color='#dddddd',which='minor',
                linewidth=MINOR_TICK_LINEWIDTH)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    if not axis_x:
        ax.tick_params(axis='x', which='both', length=0)
    if not axis_y:
        ax.tick_params(axis='y', which='both', length=0)

    if kwargs['func'] in AXIS_CHOROPLETH + AXIS_NONE:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_axisbelow(True)
    ax.tick_params(color=AXES_COLOR, labelcolor=TICKS_COLOR, which='both')

    # Scale
    # logscale: sp
    if 'xscale' in kwargs.keys():
        ax.set_xscale(kwargs['xscale'])
        if kwargs['xscale'] == 'log':
            ax.minorticks_off()
    if 'yscale' in kwargs.keys():
        ax.set_yscale(kwargs['yscale'])
        if kwargs['yscale'] == 'log':
            ax.minorticks_off()

    # Axes limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    subplot_args = kwargs['subplot_args']
    if 'xlim' in subplot_args.keys():
        xmin_,xmax_ = subplot_args['xlim']
        if xmin_ != 'default':
            xmin = xmin_
        if xmax_ != 'default':
            xmax = xmax_
    if 'ylim' in subplot_args.keys():
        ymin_,ymax_ = subplot_args['ylim']
        if ymin_ != 'default':
            ymin = ymin_
        if ymax_ != 'default':
            ymax = ymax_
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Scientific notation
    xscilimits = [-5, 6]
    yscilimits = [-5, 6]
    if 'xformat' in kwargs.keys():
        if 'xscilimits' in kwargs.keys():
            xscilimits = kwargs['xscilimits']

        ax.ticklabel_format(axis='x', 
                            style=kwargs['xformat'], 
                            scilimits=xscilimits)

    if 'yformat' in kwargs.keys():
        if 'yscilimits' in kwargs.keys():
            yscilimits = kwargs['yscilimits']

        ax.ticklabel_format(axis='y', 
                            style=kwargs['xformat'], 
                            scilimits=xscilimits)

    subplot_args = kwargs['subplot_args'] 
    if 'sharex' in subplot_args.keys():
        plt.setp(ax.get_xticklabels(), visible=False)
    if 'sharey' in subplot_args.keys():
        plt.setp(ax.get_yticklabels(), visible=False)

    return

def contour(ax=None, **kwargs):
    data = kwargs['data']

    xcol = kwargs['x']
    ycol = kwargs['y']
    zcol = kwargs['z']

    data = data.sort_values(by=[xcol, ycol])

    xmin = data[xcol].min()
    xmax = data[xcol].max()
    ymin = data[ycol].min()
    ymax = data[ycol].max()

    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_z = griddata((data[xcol].values, data[ycol].values), 
                      data[zcol].values, 
                      (grid_x, grid_y), method='nearest')

    if not 'mode' in kwargs.keys():
        kwargs['mode'] = 'lines'
        func = ax.contour
    elif kwargs['mode'] == 'lines':
        func = ax.contour
    elif kwargs['mode'] == 'filled':
        func = ax.contour
    else:
        raise(f'contour: Unsupported mode {kwargs["mode"]}.')
        
    kwargs_ = kwargs.copy()
    for ele in ['x', 'y', 'z', 'data', 'mode']:
        del kwargs_[ele]

    contour = func(grid_x, grid_y, grid_z, **kwargs_)

    return contour

def colorbar(funcres, ax, fontsizes, cb_args):
    if 'orientation' not in cb_args.keys():
        cb_args['orientation'] = 'vertical'
    if 'shrink' not in cb_args.keys():
        cb_args['shrink'] = 0.8

    cbar_obj_args = {'width': 0, 'length': 0}
    for ele in ['labelsize']:
        if ele in cb_args.keys():
            if ele == 'labelsize':
                cbar_obj_args[ele] = fontsizes[cb_args[ele]]
            else:
                cbar_obj_args[ele] = cb_args[ele]
            del cb_args[ele]
    cbar = plt.colorbar(funcres, ax=ax, **cb_args)
    cbar.outline.set_edgecolor('white')

    cbar.ax.tick_params(**cbar_obj_args)

    return

def subplot_labels(**kwargs):
    ax = kwargs['ax']
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'])
    return

def savefig(filename, pad_inches=0.05, **kwargs):
    # use pad_inches=0 to completely remove white space
    plt.savefig(filename, bbox_inches='tight', **kwargs)
    return

def subplot_fonts(**kwargs):
    ax = kwargs['ax']

    argvals = {k: v for k,v in DEFAULT_FONTS.items()}
    for k,v in kwargs.items():
        argvals[k] = v

    font_set = FONT_TABLE[argvals['fontsize']]

    ax.set_title(ax.get_title(), fontsize=font_set[argvals['title']])
    ax.set_xlabel(ax.get_xlabel(), 
            fontsize=font_set[argvals['xlabel']])
    ax.set_ylabel(ax.get_ylabel(), 
            fontsize=font_set[argvals['ylabel']])
    ax.tick_params(axis='x', which='major', 
            labelsize=font_set[argvals['xtick']])
    ax.tick_params(axis='y', which='major', 
            labelsize=font_set[argvals['ytick']])

    # color bar
    if kwargs['func'] in ['gpd.plot']:
        try:
            # Note that axes is arranged the following way: 
            # (ax1, cbar1, ax2, cbar2, ...)
            cbar = ax.figure.axes[-1]
            cbar.tick_params(labelsize=font_set[argvals['colorbar']])
        except Exception as err:
            logging.warning(f'Some error related to colorbar: {err}')
            pass
    return argvals

def update_rc_params(field=None, mode='replace', value=None):
    if mode == 'append':
        rcParams[field] = rcParams[field] + value
    elif mode == 'replace':
        rcParams[field] = value
    return

# Get actual coordinates of points in pixels w.r.t. (0,0).
# Helps in downstream tikz tasks.
# Tip: Use pad_inches=0 during plot.savefig()
def get_real_coordinates(fig, ax, units='cm', x=None, y=None):
    # Shift coordinates by making them relative to (0,0)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xc = x - xmin
    yc = y - ymin

    # Set units
    if units == 'px':
        units_ratio = fig.dpi
    elif units == 'cm':
        units_ratio = 2.54 # 1 inch = 2.54cm
    elif units == 'in':
        units_ratio = 1

    # Get ratio to scale
    coord_width = xmax - xmin
    coord_height = ymax - ymin
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    real_width = bbox.width * units_ratio
    real_height = bbox.height * units_ratio
    x_ratio = real_width / coord_width
    y_ratio = real_height / coord_height

    return xc*x_ratio, yc*y_ratio, real_width, real_height

# For fancy square grids
def square_grid_cells(axis=None, num_cells_x=None, num_cells_y=None, 
        labels_step_x=1, labels_step_y=1, type_x=None, type_y=None):

    # Set xticks
    # We are setting minor ticks due to an offset issue. Hence, reducing the 
    # number of cells by half.
    xmin, xmax = axis.get_xlim()
    xticks = np.linspace(xmin, xmax, num=int(np.floor(num_cells_x/2))+1)
    xtick_labels = [None] * len(xticks)

    # check if all 
    for i in range(0,len(xticks),labels_step_x):
        if type_x:
            xtick_labels[i] = type_x(xticks[i])
        else:
            xtick_labels[i] = xticks[i]
    axis.set_xticks(xticks, labels=xtick_labels)

    # set yticks
    ymin, ymax = axis.get_ylim()
    yticks = np.linspace(ymin, ymax, num=int(np.floor(num_cells_y/2))+1)
    ytick_labels = [None] * len(yticks)
    for i in range(0,len(yticks),labels_step_y):
        if type_y:
            ytick_labels[i] = type_y(yticks[i])
        else:
            ytick_labels[i] = yticks[i]
    axis.set_yticks(yticks, labels=ytick_labels)

    # set aspect
    axis.set_aspect((xmax-xmin)/(ymax-ymin)*num_cells_y/num_cells_x)

    # grid needs to be redrawn
    axis.tick_params(axis='both', which='minor', labelleft=False, 
            labelbottom=False)
    axis.grid(color=GRID_COLOR, which='both', linewidth=.5)

    return

# This will return the ith color for now
def get_style(k, i):
    j = 0
    for d in rcParams['axes.prop_cycle']():
        if i == j:
            return d[k]
        j += 1

# Mandatory dictionary fields: x, y, text for each list element
def text(ax=None, data=None, x='x', y='y', textcol='text', 
         fontsize='normalsize', **kwargs):
    if type(data) == pd.DataFrame:
        for t in data.to_dict('records'):
            ax.text(t[x], t[y], t[textcol], 
                    fontsize=FONT_TABLE[DEFAULT_FONTS['fontsize']][fontsize],
                    **kwargs)
    elif type(data) is str:
            ax.text(x, y, data, 
                    fontsize=FONT_TABLE[DEFAULT_FONTS['fontsize']][fontsize],
                    **kwargs)
    return

def vlines(ax=None, lines=None, **kwargs):
    if type(lines) == pd.DataFrame:
        lines = lines.to_dict('records')

    for l in lines:
        ax.vlines(l['x'], l['ymin'], l['ymax'], **kwargs)

def hlines(ax=None, data=None, y='y', xmin='xmin', xmax='xmax', **kwargs):
    if type(data) == pd.DataFrame:
        lines = data.to_dict('records')

    if 'palette' in kwargs.keys():
        palette = kwargs['palette']
        del kwargs['palette']

    i = 0
    for l in lines:
        if 'color' in l.keys():
            kwargs['color'] = palette[l['color']]
        ax.hlines(l[y], l[xmin], l[xmax], **kwargs)

    return ax

def lines(ax=None, data=None, 
          sx='source_x', sy='source_y',
          tx='target_x', ty='target_y', 
          **kwargs):
    if type(data) == pd.DataFrame:
        lines = data.to_dict('records')

    if 'palette' in kwargs.keys():
        palette = kwargs['palette']
        del kwargs['palette']

    i = 0
    for l in lines:
        if 'color' in l.keys():
            try:
                kwargs['color'] = palette[l['color']]
            except:
                pass
        ax.plot([l[sx], l[tx]], [l[sy], l[ty]], **kwargs)

    return ax

def table(df, out):
    # Create a table plot
    fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the size
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.savefig("table_image.png", bbox_inches='tight', dpi=300)

def arrow(ax=None, data=None, 
          sx='source_x', sy='source_y',
          tx='target_x', ty='target_y', 
          **kwargs):
    if type(data) == pd.DataFrame:
        lines = data.to_dict('records')

    if 'palette' in kwargs.keys():
        palette = kwargs['palette']
        del kwargs['palette']

    i = 0
    for l in lines:
        if 'color' in l.keys():
            try:
                kwargs['color'] = palette[l['color']]
            except:
                pass
        ax.arrow(l[sx], l[sy], l[tx]-l[sx], l[ty]-l[sy], head_width=.3, **kwargs)

    return ax

def pie(ax=None, **kwargs):
    counts = kwargs['data']
    if 'inside_display' not in kwargs.keys():
        inside_display = '%1.1f%%'
    elif kwargs['inside_display'] == 'perc':
        inside_display = '%1.1f%%'
    elif kwargs['inside_display'] == 'count':
        inside_display = lambda pct: autopct_counts(pct, counts)

    try:
        del kwargs['inside_display']
    except:
        pass

    ax.pie(counts, **kwargs, autopct=inside_display)

    return ax

def autopct_counts(pct, counts):
    total = counts.sum()
    count = int(round(pct * total / 100.0))
    return f"{count}"

def main():
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--properties_files', required=True,
            nargs='*', help='Cascade property files in format specified by "--input_format".')
    parser.add_argument('-f', '--input_format', default='parquet',
            choices=['json', 'parquet'],
            help='Input format.')
    parser.add_argument('-p','--plot_request_file',
            help=f'Plots to visualize will be specified through a JSON file with some mandatory and some optional fields. The structure is still evolving. Some template(s) or example(s) will be provided in the future.')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    # set logger
    if args.debug:
       logging.basicConfig(level=logging.DEBUG,format=FORMAT)
    elif args.quiet:
       logging.basicConfig(level=logging.WARNING,format=FORMAT)
    else:
       logging.basicConfig(level=logging.INFO,format=FORMAT)

    start = time()

    # Reading plot request file
    logging.info(f'Reading plot list file: {args.plot_request_file} ...')
    with open(args.plot_request_file) as f:
        plot_request = load(f)
    data_request = plot_request['data']
    del plot_request['data']

    # Preparing data
    logging.info(f'Reading {len(args.properties_files)} property file names ..')
    cascade_props = data.CascadeProperties(args.properties_files,
            data_request=data_request, input_format=args.input_format, 
            argparse=vars(args))

    # Start plotting
    logging.info(f'Start plotting ...')
    for k, item in plot_request.items():
        logging.info(f'Plotting {k} ...')
        plotter.plot(cascade_props, k, item)
        
    logging.info(f'Time taken: {(time()-start)//60}.')
    logging.info('Done.')

if __name__=='__main__':
    main()
