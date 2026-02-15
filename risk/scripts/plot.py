from cycler import cycler, Cycler
from functools import lru_cache
try:
    import geoplot as gplt
    import geoplot.crs as gcrs
except ImportError:
    print('Failed to load geoplot. Continuing anyway ...')
import logging
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc, patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, AutoLocator, AutoMinorLocator, FuncFormatter, ScalarFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from pathlib import Path
from pdb import set_trace
import seaborn as sns

from kbviz import colors

FONT_TABLE=pd.DataFrame({
    9: {'miniscule': 4, 'tiny': 5, 'scriptsize': 6, 'footnotesize': 7, 
        'small': 8, 'normalsize': 9, 'large': 10, 'Large': 11, 'LARGE': 12, 
        'huge': 14, 'Huge': 17, 'HUGE': 20}, 
    10: {'miniscule': 5, 'tiny': 6, 'scriptsize': 7, 'footnotesize': 8, 
         'small': 9, 'normalsize': 10, 'large': 11, 'Large': 12, 'LARGE': 14, 
         'huge': 17, 'Huge': 20, 'HUGE': 25}, 
    11: {'miniscule': 6, 'tiny': 7, 'scriptsize': 8, 'footnotesize': 9,
         'small': 10, 'normalsize': 11, 'large': 12, 'Large': 14, 'LARGE': 17,
         'huge': 20, 'Huge': 25, 'HUGE': 30}, 
    12: {'miniscule': 7, 'tiny': 8, 'scriptsize': 9, 'footnotesize': 10,
         'small': 11, 'normalsize': 12, 'large': 14, 'Large': 17, 'LARGE': 20,
         'huge': 25, 'Huge': 30, 'HUGE': 36}, 
    14: {'miniscule': 8, 'tiny': 9, 'scriptsize': 10, 'footnotesize': 11,
         'small': 12, 'normalsize': 14, 'large': 17, 'Large': 20, 'LARGE': 25,
         'huge': 30, 'Huge': 36, 'HUGE': 48}, 
    17: {'miniscule': 9, 'tiny': 10, 'scriptsize': 11, 'footnotesize': 12,
         'small': 14, 'normalsize': 17, 'large': 20, 'Large': 25, 'LARGE': 30,
         'huge': 36, 'Huge': 48, 'HUGE': 60}, 
    20: {'miniscule': 10, 'tiny': 11, 'scriptsize': 12, 'footnotesize': 14,
         'small': 17, 'normalsize': 20, 'large': 25, 'Large': 30, 'LARGE': 36,
         'huge': 48, 'Huge': 60, 'HUGE': 72}, 
    25: {'miniscule': 11, 'tiny': 12, 'scriptsize': 14, 'footnotesize': 17,
         'small': 20, 'normalsize': 25, 'large': 30, 'Large': 36, 'LARGE': 48,
         'huge': 60, 'Huge': 72, 'HUGE': 84}, 
    30: {'miniscule': 12, 'tiny': 14, 'scriptsize': 17, 'footnotesize': 20,
         'small': 25, 'normalsize': 30, 'large': 36, 'Large': 48, 'LARGE': 60,
         'huge': 72, 'Huge': 84, 'HUGE': 96},
    36: {'miniscule': 14, 'tiny': 17, 'scriptsize': 20, 'footnotesize': 25,
         'small': 30, 'normalsize': 36, 'large': 48, 'Large': 60, 'LARGE': 72,
         'huge': 84, 'Huge': 96, 'HUGE': 108},
    48: {'miniscule': 17, 'tiny': 20, 'scriptsize': 25, 'footnotesize': 30,
         'small': 36, 'normalsize': 48, 'large': 60, 'Large': 72, 'LARGE': 84,
         'huge': 96, 'Huge': 108, 'HUGE': 120},
    60: {'miniscule': 20, 'tiny': 25, 'scriptsize': 30, 'footnotesize': 36,
         'small': 48, 'normalsize': 60, 'large': 72, 'Large': 84, 'LARGE': 96,
         'huge': 108, 'Huge': 120, 'HUGE': 132}})

# Constants that will finally go to the config file
AXES_COLOR = '#999999'
GRID_COLOR = '#cccccc'

TICKS_COLOR = '#222222'
TICK_LENGTH = 3
COLORBAR_TICK_LENGTH = 3
MAJOR_TICK_LINEWIDTH = .75
MINOR_TICK_LINEWIDTH = .25
TICK_NBINS = 6

HATCH = ['++', 'xx', '\\\\', '..', 'o', '|', '*']
HATCH_LW = .5
HATCH_COLOR = 'white'
PATCH_EDGE_COLOR = 'white' 
#####

MIN_ZORDER = 5
COLORBAR_ZORDER = 10
INFINITY = 10**10
GEOPLOT_XLIM = (-INFINITY, INFINITY)
GEOPLOT_EXTENT = .97
NUM_MINORS = 4

BOXPLOT_MEDIAN_LW = 2
BOXPLOT_COMPONENT_COLOR = 'black'
BOXPLOT_USE_FACE_COLOR_FLAG = True
FILL_FALSE_FACECOLOR = (1,1,1,.8)
FILL_FALSE_LW = 2
LINEPLOT_LW = 2

EPSILON = 1e-5
LOG_EPSILON = -5

def palette_map(palette):
    return eval(f'colors.{palette}')

RC_PARAMS = {
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'legend.frameon': True,
        'legend.framealpha': .7,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.borderpad': .1,
        'hatch.linewidth': HATCH_LW,
        'hatch.color': HATCH_COLOR,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'axes.prop_cycle': cycler(color=palette_map('tableau')),
        'axes.formatter.limits': [-5,6] # important for scientific notation
        }

class Fig():

    def __init__(self, x=None, y=None, colors='tableau', fontsize=20, 
                 pretty=True, **kwargs): 

        # Setting figure-wide defaults: needs revisiting: What if I want to change fonts?
        self.default_fig = {'x': 5, 'y': 4}
        self.default_grid = {'nrows': 1, 'ncols': 1, 'wspace': 0.2, 'hspace': 0.2}
        self.default_title = {'fontsize': 'Large'}
        self.default_xlabel = {'fontsize': 'large'}
        self.default_ylabel = {'fontsize': 'large'}
        self._subplot_matrix = {}

        # AA: may be set _default_legend, default_font, etc.
        logging.info('Initializing figure (rcParams, figure dimensions, fonts, colors) ...')
        for k,v in RC_PARAMS.items():
            rcParams[k] = v

        plt.clf()

        self._called = set()
        self._initiated = False
        self.__finalized = False

        self.subplots = []
        self.fig = plt.figure(figsize=[x,y], **kwargs)

        # Font table
        self.fonts = FONT_TABLE[fontsize]

        # Setting color
        color_set = palette_map(colors)
        self.palette = cycler(color=color_set)
        rcParams['axes.prop_cycle'] = self.palette
        sns.set_palette(color_set)

        # Storing size
        self.width, self.height = self.fig.get_size_inches()

        return

    def grid(self, **kwargs):
        logging.info('Setting grid ...')
        self._called = called(self._called, 'grid')
        arg = augment_dict(kwargs, self.default_grid)
        self._grid = self.fig.add_gridspec(**arg)
        return

    def _add_subplot(self, row=None, col=None, **kwargs):
        fig = self.fig
        subplot_matrix_keys = self._subplot_matrix.keys()
        gs = self._grid

        # set row and col using a raster search if either one is unspecified.
        # note that we do not bother about the case when only one is specified.
        break_flag = False
        if row is None or col is None:
            for r in range(gs.nrows):
                for c in range(gs.ncols):
                    if (r,c) not in subplot_matrix_keys:
                        break_flag = True
                        break
                if break_flag:
                    break
            row = r
            col = c
            logging.info(f'Either row or col was None. Assigning ({row},{col}) based on raster.')
        if (row,col) in subplot_matrix_keys:
            raise ValueError(
                    f'A subplot already exists in the specified location ({row},{col}).')
        sp = fig.add_subplot(self._grid[row, col], **kwargs) 
        self._subplot_matrix[(row,col)] = sp

        # get bounding box for the subplot
        nrows = self._grid.nrows
        ncols = self._grid.ncols
        xr = list(self._grid[row,col].colspan)
        yr = list(self._grid[row,col].rowspan)
        spxmin = xr[0] / ncols
        spxmax = (xr[-1] + 1) / ncols
        spymin = yr[0] / nrows
        spymax = (yr[-1] + 1) / nrows

        return sp, Bbox.from_extents(spxmin, spymin, spxmax, spymax) 

    def title(self, value='', pretty=True, **kwargs):
        logging.info('Setting figure title ...')
        arg = augment_dict(kwargs, self.default_title)
        arg = font_map(self.fonts, arg)
        if pretty:
            value = fr'\parbox[b]{{{self.width:.2f}in}}{{\center {value}}}'
        self.fig.suptitle(value, **kwargs)
        return

    def xlabel(self, value='', width_ratio=1, **kwargs):
        logging.info('Setting figure xlabel ...')
        arg = augment_dict(kwargs, self.default_xlabel)
        arg = font_map(self.fonts, arg)
        value = fr'\parbox[b]{{{self.width*width_ratio:.2f}in}}{{\center {value}}}'
        self.fig.supxlabel(value, **kwargs)
        return

    def ylabel(self, value='', width_ratio=1, **kwargs):
        logging.info('Setting figure ylabel ...')
        arg = augment_dict(kwargs, self.default_ylabel)
        arg = font_map(self.fonts, arg)
        value = fr'\parbox[b]{{{self.height*width_ratio:.2f}in}}{{\center {value}}}'
        self.fig.supylabel(value, **kwargs)
        return

    def _initiate(self):
        ensure_called(self, 'grid')
        self._initiated = True
        logging.info('Figure ready for plotting ...')

    def finalize(self):
        i = 0
        for sp in self.subplots:
            logging.info(f'Finalizing subplot {i} ...')
            sp.finalize()
            i += 1
        self.__finalized = True

    def savefig(self, filename, pad_inches=0.05, **kwargs):
        # use pad_inches=0 to completely remove white space
        self.finalize()
        plt.savefig(filename, bbox_inches='tight', **kwargs)
        plt.close()
        return

class Subplot():
    def __init__(self, fig=None, row=None, col=None, 
                 xlim=(None, None), ylim=(None, None), 
                 xscale=None, yscale=None, 
                 square_cells=False, **kwargs): 

        self._load_defaults()
        self._called = set()
        self.plot_elements = []
        self.__finalized = False
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale

        # figure properties
        if fig is None:
            raise ValueError('fig is None')
        self.fig = fig

        if not fig._initiated:
            fig._initiate()

        self._square_cells = square_cells

        self.default_subplot = {}

        self.sharex = None
        if 'sharex' in kwargs.keys():
            kwargs['sharex'] = kwargs['sharex'].ax
            self.sharex = kwargs['sharex']
        self.sharey = None
        if 'sharey' in kwargs.keys():
            kwargs['sharey'] = kwargs['sharey'].ax
            self.sharey = kwargs['sharey']
        if 'projection' in kwargs.keys():   # this is for geo plots
            kwargs['projection'] = projection_map(kwargs['projection'])
            self.projection = kwargs['projection']

        subplot_args = augment_dict(kwargs, self.default_subplot)

        self.ax, self.bbox = fig._add_subplot(row=row, col=col,**subplot_args) 
        fig.subplots.append(self)

        # store subplot sizes and position
        self.fonts = fig.fonts

        return

    def _load_defaults(self):
        self.default_title = dict(fontsize='large', pad=0)
        self.default_xlabel = {'fontsize': 'normalsize'}
        self.default_ylabel = {'fontsize': 'normalsize'}
        self.default_xticks = {'labelsize': 'small', 'color': AXES_COLOR, 
                               'labelcolor': TICKS_COLOR, 'which': 'both',
                               'length': TICK_LENGTH}
        self.default_yticks = {'labelsize': 'small', 'color': AXES_COLOR, 
                               'labelcolor': TICKS_COLOR, 'which': 'both',
                               'length': TICK_LENGTH}

        self.default_axis = {}
        self.default_grid = {}
        self._lock_extent = False
        return

    def title(self, value='', width_ratio=1, **kwargs):
        #not_called(self._called, 'plot')
        ax = self.ax
        self._called = called(self._called, 'title')

        arg = augment_dict(kwargs, self.default_title)
        width = ax.get_position().width * self.fig.width * width_ratio
        value = fr'\parbox[b]{{{width:.2f}in}}{{\center {value}}}'
        arg = font_map(self.fonts, arg)
        ax.set_title(value, **arg)

    def xlabel(self, value='', width_ratio=1, **kwargs):
        self._called = called(self._called, 'xlabel')

        width = self.ax.get_position().width * self.fig.width * width_ratio

        self.label(value, default_dict=self.default_xlabel, parboxwidth=width, 
                   func=self.ax.set_xlabel, **kwargs)

    def ylabel(self, value='', width_ratio=1, **kwargs):
        self._called = called(self._called, 'ylabel')
        height = self.ax.get_position().height * self.fig.height * width_ratio

        self.label(value, default_dict=self.default_ylabel, parboxwidth=height, 
                   func=self.ax.set_ylabel, **kwargs)

    def label(self, val, default_dict=None, parboxwidth=None, func=None, **kwargs):
        #ensure_called(self, 'grid')
        arg = augment_dict(kwargs, default_dict)
        val = fr'\parbox[b]{{{parboxwidth:.2f}in}}{{\center {val}}}'
        arg = font_map(self.fonts, arg)
        func(val, **arg)

    def _axes_limits(self):
        not_called(self._called, 'plot')
        self._called = called(self._called, '_axes_limits')

        ax = self.ax

        # Axes limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ## if self.pretty:
        ##     xmin, xmax, ymin, ymax = get_extent(ax)

        xmin_, xmax_ = self.xlim
        xlim_manipulated = False
        if xmin_ is not None:
            xmin = xmin_
            xlim_manipulated = True
        if xmax_ is not None:
            xmax = xmax_
            xlim_manipulated = True

        ymin_, ymax_ = self.ylim
        ylim_manipulated = False
        if ymin_ is not None:
            ymin = ymin_
            ylim_manipulated = True
        if ymax_ is not None:
            ymax = ymax_
            ylim_manipulated = True

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return

    def _axes_scale(self):
        self._called = called(self._called, '_axes_scale')
        ensure_called(self, '_axes_limits')

        ax = self.ax
        if not self.xscale is None:
            ax.set_xscale(self.xscale)
        if not self.yscale is None:
            ax.set_yscale(self.yscale)
        return

    def _grid(self, axis_encoding=None, **kwargs):
        self._called = called(self._called, '_grid')
        ensure_called(self, '_axes_scale')

        ax = self.ax
        arg = augment_dict(kwargs, self.default_grid)

        # Axis encoding: axy:gxy:mxy:txy
        if axis_encoding is None:
            # If not given, use the first plot element's axis encoding.
            pe = self.plot_elements[0]
            axis_encoding = pe._set_axis_encoding()

        seg = axis_encoding.split(':')
        err = 'Wrong encoding of axis type: required "axy:gxy:mxy:txy".'
        axis_x = True
        axis_y = True
        grid_x = True
        grid_y = True
        minor_x = True
        minor_y = True

        for a in seg:
            if len(a) > 3 or a[0] not in ['a', 'g', 'm', 't']:
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
            #ax.spines['bottom'].set_color(GRID_COLOR)
        if not axis_y:
            ax.spines['left'].set_visible(False)
            #ax.spines['left'].set_color(GRID_COLOR)

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

        if minor_x or minor_y:
            ax.grid(True,color='#dddddd',which='minor',
                    linewidth=MINOR_TICK_LINEWIDTH)

            if self.xscale != 'log':
                ax.xaxis.set_minor_locator(AutoMinorLocator(NUM_MINORS))
            
            if self.yscale != 'log':
                ax.yaxis.set_minor_locator(AutoMinorLocator(NUM_MINORS))

        # Axis and grid lines will occur below other artists: zorder
        ax.set_axisbelow(True)

    def ticks(self, labels=None, ticks=None, default_dict=None, spine=None, 
              axis=None, set_ticks=None, set_ticklabels=None,
              get_ticks=None, get_ticklabels=None, share=None, 
              axis_encoding=None, prune=None, labeldict={}, **kwargs):
        
        #print(type(self.ax.xaxis.get_major_locator()))

        ensure_called(self, '_grid')

        # Axis encoding: axy:gxy:mxy:txy
        if axis_encoding is None:
            # If not given, use the first plot element's axis encoding.
            pe = self.plot_elements[0]
            axis_encoding = pe._set_axis_encoding()

        seg = axis_encoding.split(':')
        err = 'Wrong encoding of axis type: required "axy:gxy:mxy:txy".'
        ticks_axis = True

        for a in seg:
            if len(a) > 3 or a[0] not in ['a', 'g', 'm', 't']:
                raise ValueError(err)
            if a[0] == 't':
                if axis not in a: ticks_axis = False

        if not ticks_axis:
            set_ticks([])
            return

        # This needs more control than title and label functions 
        ax = self.ax
        arg = augment_dict(kwargs, default_dict)
        arg = font_map(self.fonts, arg)

        ## # Separate out args corresponding to text
        ## text_args = ['ha', 'va', 'rotation_mode']
        ## arg_labels = {k: v for k,v in arg.items() if k in text_args}
        ## arg = prune_dict(arg, text_args)
        
        # Plot-based settings
        if not spine.get_visible(): 
            arg['length'] = 0

        # Some parameters can only be set using function set_xticklabels
        # To run this function without warnings, set_xticks need to be set.
        # WARNING: This order is important. tick_params seems to be changing
        # xlim/ylim when it shouldn't be. So, storing ticks before tick_params.
        if ticks is None:
            ##for nbins in range(TICK_NBINS, TICK_NBINS+4):
            ##    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
            ##    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
            ##    ticks = get_ticks()
            ##    print(ticks)
            ##    if len(ticks) > TICK_NBINS - 1:
            ##        break
            ticks = get_ticks()
            if labels is None:
                labels = [xx.get_text() for xx in get_ticklabels()]
        else:
            if labels is None:
                labels = [str(x) for x in ticks]

        ax.tick_params(axis=axis, **arg)
        ax.tick_params(axis=axis, which='minor', length=0)
        
        if prune == 'left':
            labels[0] = ''
        elif prune == 'right':
            labels[-1] = ''
        elif prune == 'both':
            labels[0] = ''
            labels[-1] = ''
        elif prune is None:
            pass
        else:
            raise ValueError('Invalid option for "prune".')

        set_ticks(ticks)

        # Labels
        #labeldict = augment_dict(labeldict, arg)
        if 'rotation' in labeldict.keys():
            pass
        set_ticklabels(labels, **labeldict)

        if not share is None:
            plt.setp(get_ticklabels(), visible=False)

    def xticks(self, ticks=None, labels=None, axis_encoding=None, prune=None, 
               labeldict={}, **kwargs):
        self._called = called(self._called, 'xticks')

        self.ticks(default_dict=self.default_xticks, axis='x',
                   prune=prune, ticks=ticks, labels=labels,
                   set_ticks=self.ax.set_xticks, 
                   set_ticklabels=self.ax.set_xticklabels,
                   get_ticks=self.ax.get_xticks,
                   get_ticklabels=self.ax.get_xticklabels,
                   spine=self.ax.spines['bottom'], share=self.sharex, 
                   axis_encoding=axis_encoding, labeldict=labeldict, **kwargs) 

    def yticks(self, ticks=None, labels=None, axis_encoding=None, prune=None, 
               labeldict={}, **kwargs):
        self._called = called(self._called, 'yticks')

        self.ticks(default_dict=self.default_yticks, axis='y',
                   prune=prune, ticks=ticks, labels=labels,
                   set_ticks=self.ax.set_yticks, 
                   set_ticklabels=self.ax.set_yticklabels,
                   get_ticks=self.ax.get_yticks,
                   get_ticklabels=self.ax.get_yticklabels,
                   spine=self.ax.spines['left'], share=self.sharey,
                   axis_encoding=axis_encoding, labeldict=labeldict, **kwargs) 

    def get_ticks(self, axis=None):
        ax = self.ax
        if axis == 'x':
            return ax.get_xticks(), [x.get_text() for x in ax.get_xticklabels()]
        elif axis == 'y':
            return ax.get_yticks(), [x.get_text() for x in ax.get_yticklabels()]
        else:
            raise ValueError('Axis can be "x/y".')

    def _set_square_cells(self):
        self._called = called(self._called, '_set_square_cells')
        ensure_called(self, 'xticks')
        ensure_called(self, 'yticks')

        if not self._square_cells:
            return
        ax = self.ax

        # Ensure ticks are set or retrieved after plotting
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Number of grid cells (tick intervals)
        if self.xscale == 'log':
            dx = np.log10(xticks[1]) - np.log10(xticks[0])
        else:
            dx = xticks[1] - xticks[0]

        if self.yscale == 'log':
            dy = np.log10(yticks[1]) - np.log10(yticks[0])
        else:
            dy = yticks[1] - yticks[0]

        # Compute aspect ratio to make data cells square
        ax.set_aspect(dx/dy)
        return

    def text(self):
        pass

    def finalize(self):
        # Mandatory functions
        not_called(self._called, 'plot')
        ensure_called(self, '_set_square_cells')
        ensure_called(self, 'xlabel')
        ensure_called(self, 'ylabel')
        #ensure_called(self, 'check_for_legend_call')

class Boxplot():
    funcobj = 'sns.boxplot'
    orientation = 'v'   # y is dependent variable

    def __init__(self, subplot=None, row=0, col=0, data=None, **kwargs):
        self._load_defaults()

        if data is None:
            raise ValueError('Data not provided.')
        else:
            self.data = data

        if subplot is None:
            raise ValueError('Subplot axis not provided.')
        self.subplot = subplot

        # The plot element or layer
        subplot._called = called(subplot._called, 'plot', repeat=True)
        kwargs['data'] = self.data

        self._default_func['zorder'] = MIN_ZORDER + len(subplot.plot_elements)
        self._plot(**kwargs)

        # Add to subplot
        subplot.plot_elements.append(self)

        return

    def _load_defaults(self):
        self._default_func = {}
        self._default_legend = {'fontsize': 'small', 'title_fontsize': 'small', 
                               'title': '', 'labelspacing': .1}

    def _plot(self, **kwargs):
        # This is sns specific.
        func = eval(self.funcobj)
        arg = augment_dict(kwargs, self._default_func)

        ax = self.subplot.ax

        self.hatch = False
        if 'hatch' in arg.keys():
            self.hatch = arg['hatch']
            arg = prune_dict(arg, ['hatch'])

        arg = self._set_colors(**arg)

        # fill handled by us
        if 'fill' in arg.keys():
            self.fill = arg['fill']
            #arg = prune_dict(arg, ['fill'])
        else:
            self.fill = True
        if 'element' in arg.keys():
            if arg['element'] in ['poly', 'step']:    # A histplot feature, actually
                arg = prune_dict(arg, ['edgecolor'])
                arg['linewidth'] = LINEPLOT_LW

        func(ax=ax, **arg)
        if 'orient' in arg.keys():
            self.orientation = arg['orient']

        self._extract_legend()
        self._update_patches(kwargs)

    def _set_colors(self, **kwargs):

        if 'palette' in kwargs.keys():
            kwargs['palette'] = color_map(
                    kwargs['palette'], rcParams['axes.prop_cycle'])
        if 'color' in kwargs.keys():
            kwargs['color'] = color_map(
                    kwargs['color'], rcParams['axes.prop_cycle'])
        return kwargs

    def _update_patches(self, plot_args):
        ax = self.subplot.ax

        pp = [patch for patch in ax.patches if type(patch) == patches.PathPatch]
        # There are len(pp) boxes.

        for i, patch in enumerate(pp):
            if BOXPLOT_USE_FACE_COLOR_FLAG == True:
                ele_color = patch.get_facecolor()
            else:
                ele_color = BOXPLOT_COMPONENT_COLOR
            # We assume that the 4th element (starting from 0) is the median.
            # The last element is the outlier, which may or may not be present.
            num_elements = len(ax.lines) // len(pp) 
            for ele in range(4):
                ax.lines[i*num_elements+ele].set_color(ele_color)
            ax.lines[i*num_elements+4].set_linewidth(BOXPLOT_MEDIAN_LW)
            if num_elements == 6:
                ax.lines[i*num_elements+5].set_markeredgecolor(ele_color)
            if self.fill:
                patch.set(edgecolor=HATCH_COLOR)
            else:
                patch.set_facecolor(FILL_FALSE_FACECOLOR)
                patch.set_edgecolor(ele_color)
                if not self.hatch:
                    ax.lines[i*num_elements+4].set_color(ele_color)

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            if self.fill:
                patch.set(edgecolor=HATCH_COLOR)
            else:
                patch.set(facecolor=FILL_FALSE_FACECOLOR, 
                          edgecolor=patch.get_facecolor(),
                          linewidth=FILL_FALSE_LW)

        if not self.hatch:
            return

        huelist = []
        for patch in pp:
            if self.fill:
                huelist.append(patch.get_facecolor())
            else:
                huelist.append(patch.get_edgecolor())
        hues = pd.DataFrame.from_records(huelist).drop_duplicates().reset_index(
                drop=True)

        for patch in pp:
            if self.fill:
                col = patch.get_facecolor()
            else:
                col = patch.get_edgecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set(hatch=HATCH[hue])

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            if self.fill:
                col = patch.get_facecolor()
            else:
                col = patch.get_edgecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set(hatch=HATCH[hue])

    def _set_axis_encoding(self):    # specific to boxplot, violinplot, barplot
        if self.orientation == 'v':
            return 'a:gy:m:txy'
        elif self.orientation == 'h':
            return 'a:gx:m:txy'

    # This seems to be plot specific.
    def _extract_legend(self):
        ax = self.subplot.ax
        handles, labels = ax.get_legend_handles_labels()
        self.legend_data = {'mode': 'legend', 'handles': handles, 
                       'labels': labels}
        if len(handles):
            ax.legend().remove()
        return

    def legend(self, scope='subplot', **kwargs):

        # self._called = called(self._called, 'legend', repeat=True)
        # ensure_called(self, '_plot')

        handles = self.legend_data['handles']
        labels = self.legend_data['labels']
        arg = augment_dict(kwargs, self._default_legend)
        arg = font_map(self.subplot.fonts, arg)

        if scope == 'subplot':
            # Check if a legend exists
            ax = self.subplot.ax
            legend = ax.get_legend()

            if legend:
                logging.info('Detected a legend. In append mode ...')
                ax.add_artist(legend)
            else:
                logging.info('No legend detected. Creating new legend ...')
            ax.legend(handles, labels, **arg)
        elif scope == 'figure':
            # create a new figure legend
            fig = self.subplot.fig.fig
            fig.legend(handles, labels, **arg)
        else:
            raise ValueError(f'Invalid scope "{scope}".')
        return

class Violinplot(Boxplot):
    funcobj = 'sns.violinplot'

    def _update_patches(self, plot_args):
        return
        ax = self.subplot.ax
        huelist = []

        # Change the edge color of each violin
        for pc in ax.collections:
            pc.set_edgecolor(pc.get_facecolor())  # change to your desired color

        # Update legend
        handles = self.legend_data['handles']
        labels = self.legend_data['labels']
        for patch in handles:
            patch.set(edgecolor=patch.get_facecolor())
        # No hatch support
        return

class Barplot(Boxplot):
    funcobj = 'sns.barplot'

    def _load_defaults(self):
        super()._load_defaults()
        self._default_func = {'edgecolor': PATCH_EDGE_COLOR, 
                              'linewidth': LINEPLOT_LW, 
                              'width': .9}

    def _update_patches(self, plot_args):
        ax = self.subplot.ax
        huelist = []

        for bars, h in zip(ax.containers, cycler(hatch=HATCH)()):
            for bar in bars:
                if self.fill:
                    bar.set_edgecolor(HATCH_COLOR)
                else:
                    bar.set(edgecolor=bar.get_facecolor(),
                            linewidth=FILL_FALSE_LW,
                            facecolor=FILL_FALSE_FACECOLOR)

        # Update legend
        handles = self.legend_data['handles']
        labels = self.legend_data['labels']
        for patch in handles:
            if self.fill:
                patch.set(edgecolor=HATCH_COLOR)
            else:
                try:
                    patch.set(facecolor=FILL_FALSE_FACECOLOR, 
                              edgecolor=patch.get_facecolor(),
                              linewidth=FILL_FALSE_LW)
                except: # it might be a Line2D object
                    pass
                    ## patch.set(facecolor=FILL_FALSE_FACECOLOR, 
                    ##           edgecolor=patch.get_edgecolor(),
                    ##           linewidth=FILL_FALSE_LW)
                
        if not self.hatch:
            return

        for bars, h in zip(ax.containers, cycler(hatch=HATCH)()):
            for bar in bars:
                bar.set_hatch(h['hatch'])
                if self.fill:
                    huelist.append(bar.get_facecolor())
                else:
                    huelist.append(bar.get_edgecolor())
        hues = pd.DataFrame.from_records(huelist).drop_duplicates().reset_index(
                drop=True)

        # Update legend
        handles = self.legend_data['handles']
        labels = self.legend_data['labels']
        for patch in handles:
            if self.fill:
                col = patch.get_facecolor()
            else:
                col = patch.get_edgecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set(hatch=HATCH[hue])

class Lineplot(Boxplot):
    funcobj = 'sns.lineplot'

    def _load_defaults(self):
        self._default_func = {'linewidth': 2, 'alpha': 1}
        self._default_legend = {'fontsize': 'small', 'title_fontsize': 'small', 
                               'title': '', 'labelspacing': .1}

    def _update_patches(self, plot_args):
        return

    def _set_axis_encoding(self):
        return 'axy:gxy:mxy:txy'

class Scatterplot(Lineplot):
    funcobj = 'sns.scatterplot'

    def _load_defaults(self):
        self._default_func = {}
        self._default_legend = {'fontsize': 'small', 'title_fontsize': 'small', 
                               'title': '', 'labelspacing': .1}

class Histplot(Barplot):
    funcobj = 'sns.histplot'
    def _load_defaults(self):
        super()._load_defaults()
        self._default_func = {'edgecolor': PATCH_EDGE_COLOR, 'linewidth': .5, 
                             'alpha': 1}

    # This seems to be plot specific.
    def _extract_legend(self):
        ax = self.subplot.ax
        try:
            handles = ax.get_legend().legend_handles
        except:
            self.legend_data = {'mode': 'legend', 'handles': [],
                           'labels': []}
            return

        labels = []
        for ele in ax.get_legend().get_texts():
            labels.append(ele.get_text())
        self.legend_data = {'mode': 'legend', 'handles': handles, 
                       'labels': labels}
        if len(handles):
            ax.get_legend().remove()
        return

class Choropleth(Boxplot):
    funcobj = 'gplt.choropleth'

    def _load_defaults(self):
        self._default_func = {'legend': False, 'linewidth': .2, 
                              'edgecolor': AXES_COLOR, 'cmap': 'viridis', 'cmap_kws': {}}
        self._default_legend = {'length': COLORBAR_TICK_LENGTH, 'title': '', 
                                'title_fontsize': 'normalsize', 'label_color': TICKS_COLOR,
                                'labelsize': 'small'}

    def _set_axis_encoding(self):
        return 'a:g:m:t'

    def _plot(self, transform='linear', **kwargs):

        func = eval(self.funcobj)
        arg = augment_dict(kwargs, self._default_func)

        subplot = self.subplot
        ax = subplot.ax

        # color
        # This is a geoplot issue for categorical variables.
        if 'lim' not in arg.keys():
            arg['lim'] = (None, None)
        sm, ticks, labels = generate_cmap(
            self.data, arg['cmap'], kwargs['hue'], lim=arg['lim'], 
            **arg['cmap_kws'])
        arg = prune_dict(arg, ['cmap_kws'])

        self.legend_data = {'mode': 'colorbar', 'sm': sm, 
                       'ticks': ticks, 'labels': labels}

        arg = prune_dict(arg, ['data', 'cmap'])
        arg['legend'] = False   # see to it that Geoplot specific legend is 
                                # is not created when categorical

        if not 'extent' in arg.keys():
            self._update_boundary()
        else:
            subplot._extent = arg['extent']
            subplot._lock_extent = True
        arg['extent'] = subplot._extent  # rewriting if passed

        arg = prune_dict(arg, ['lim'])

        if self.data[arg['hue']].dtype.name == 'category':
            self.data[arg['hue']] = self.data[arg['hue']].cat.codes

        func(self.data, ax=ax, cmap=sm.cmap, 
             projection=self.subplot.projection, **arg)

        self._extract_legend()

        return

    def _extract_legend(self):
        # There are issues with the geoplot colorbar, particularly for categorical
        # data. So, explicitly handling the colorbar for that case.
        ## if self.legend['mode'] != 'colorbar':
        ##     raise ValueError('Only colorbar supported currently.')
        ## if not self.legend['sm'] is None:
        ##     return  # categorical detected
        return

    # legend is a colorbar here
    def legend(self, cbaxis=None, scope='subplot', **kwargs):

        ## self._called = called(self._called, 'legend', repeat=True)
        ## ensure_called(self, '_plot')

        subplot = self.subplot
        ax = subplot.ax

        # Setting constructed ticks and labels that can be later modified 
        kwargs['ticks'] = self.legend_data['ticks']
        kwargs['labels'] = self.legend_data['labels']

        # Default settings for colorbar
        arg = augment_dict(kwargs, self._default_legend)
        arg = font_map(subplot.fonts, arg)

        # Default settings for colorbar
        generate_colorbar(ax=ax, fig=subplot.fig.fig, bbox=subplot.bbox, 
                          sm=self.legend_data['sm'], 
                          cbaxis=cbaxis, scope=scope, **arg)
        return

    def _update_boundary(self):
        subplot = self.subplot
        ax = subplot.ax

        if subplot._lock_extent:
            return

        if not hasattr(subplot, '_extent'):
            #subplot._extent = self.data.to_crs(pyproj_crs).total_bounds #((xmin, xmax), (ymin, ymax))
            try:
                subplot._extent = self.data.total_bounds #((xmin,. xmax), (ymin, ymax))
            except:
                raise('Check if "data" is a geodataframe. Did not find')
        else:
            oxmin, oymin, oxmax, oymax = subplot._extent
            xmin, ymin, xmax, ymax = self.data.total_bounds
            xmin = min(xmin, oxmin)
            xmax = max(xmax, oxmax)
            ymin = min(ymin, oymin)
            ymax = max(ymax, oymax)
            subplot._extent = (xmin, ymin, xmax, ymax)
        return

class Polyplot(Choropleth):
    funcobj = 'gplt.polyplot'

    def _load_defaults(self):
        # Here, label is custom.
        Boxplot._load_defaults(self)
        self._default_func = {'linewidth': .2, 'label': 'No label',
                             'edgecolor': AXES_COLOR,
                             'facecolor': 'none'}

    def _plot(self, **kwargs):

        func = eval(self.funcobj)
        arg = augment_dict(kwargs, self._default_func)

        subplot = self.subplot
        ax = subplot.ax

        # Unlike choropleth, polyplot does not contain legend.
        arg = prune_dict(arg, ['data'])

        if not 'extent' in arg.keys():
            self._update_boundary()
        else:
            subplot._extent = arg['extent']
            subplot._lock_extent = True
        arg['extent'] = subplot._extent  # rewriting if passed

        func(self.data, ax=ax, projection=self.subplot.projection, **arg)

        # Legend
        # Polyplot does not provide a legend. Need to be explicitly generated.
        self.legend_data = {'mode': 'legend',
                       'handles': [patches.Patch(
                           edgecolor=arg['edgecolor'],
                           facecolor=arg['facecolor'],
                           linewidth=arg['linewidth'])],
                       'labels': [arg['label']]}

        return

    def legend(self, **kwargs):
        Boxplot.legend(self, **kwargs)

class Heatmap(Boxplot):
    funcobj = 'sns.heatmap'

    def _load_defaults(self):
        super()._load_defaults()
        self._default_func = {'cmap': 'viridis', 'square': True, 'linecolor': AXES_COLOR,
                              'linewidth': .1, 'annot': False, 'cmap_kws': {}}
        self._default_legend = {'length': COLORBAR_TICK_LENGTH, 'title': '', 
                                'title_fontsize': 'normalsize', 'labelcolor': TICKS_COLOR,
                                'labelsize': 'small', 'prune': None}

    def _plot(self, lim=(None, None), transform='linear', **kwargs):
        # This is sns specific.
        func = eval(self.funcobj)
        arg = augment_dict(kwargs, self._default_func)

        self.transform = transform

        if 'order' in kwargs.keys():
            order = kwargs['order']
            if order is not None:
                self.data = self.data.loc[order, order]
                kwargs['data'] = self.data
                try:
                    kwargs['annot'] = kwargs['annot'].loc[order,order]
                except:
                    pass
            kwargs = prune_dict(kwargs, ['order'])

        # color
        sm, ticks, labels = generate_cmap(
                self.data.stack().to_frame(name='val'), arg['cmap'], 'val', 
                lim=lim, transform=transform, **arg['cmap_kws'])
        arg['cmap'] = sm.cmap
        arg = prune_dict(arg, ['cmap_kws'])
        if 'annot_kws' in kwargs.keys():
            kwargs['annot_kws'] = font_map(self.subplot.fonts, kwargs['annot_kws'])

        self.legend_data = {'mode': 'colorbar', 'sm': sm, 
                            'ticks': ticks, 'labels': labels}

        subplot = self.subplot
        ax = self.subplot.ax

        func(ax=ax, norm=sm.norm, **arg)
        self._extract_legend()

        arg = self._set_colors(**arg)
        self._update_patches(kwargs)
        
        if 'annot' in arg.keys():
            if arg['annot'] is not None or arg['annot'] is True:
                for text in ax.texts:
                    text.set_zorder(self._default_func['zorder'])

    def _set_axis_encoding(self):
        return 'a:g:m:txy'

    def _extract_legend(self):
        self.subplot.ax.collections[0].colorbar.remove()
        return
        ## ax = self.subplot.ax
        ## cbar = ax.collections[0].colorbar

        ## ticks = cbar.get_ticks()
        ## labels = [label.get_text() for label in cbar.ax.get_yticklabels()]

        ## self.legend_data = {'mode': 'colorbar', 'cmap': cbar.mappable.get_cmap(), 
        ##                     'sm': cbar.mappable, 'ticks': ticks, 'labels': labels}
        ## return

    def legend(self, cbaxis=None, scope='subplot', **kwargs):
        Choropleth.legend(self, cbaxis=cbaxis, scope=scope, **kwargs)

    def _update_patches(self, plot_args):
        return

def called(called_func, method, repeat=False):
    if method in called_func: 
        if repeat:
            logging.info(f'Calling {method} ...')
            return called_func
        else:
            raise ValueError(f'{method} already called once.')
    called_func.add(method)
    return called_func

def not_called(called_func, method):
    # call the method if it hasn't been called already
    if method not in called_func:
        raise ValueError(f'{method} should have been called.')

def ensure_called(object, method, *default_args, **default_kwargs):
    # call the method if it hasn't been called already
    if method not in object._called:
        logging.info(f'Running {method} with defaults ...')
        getattr(object, method)(*default_args, **default_kwargs)

def augment_dict(dict_tobe_augmented, dict_new_values):
    keys = dict_tobe_augmented.keys()
    for k,v in dict_new_values.items():
        if not k in keys:
            dict_tobe_augmented[k] = v
    return dict_tobe_augmented

def prune_dict(dict_tobe_pruned, keys):
    for k in keys:
        try:
            del dict_tobe_pruned[k]
        except:
            pass
    return dict_tobe_pruned

def divide_dict(dict_tobe_divided, keys):
    new_dict = {}
    for k in keys:
        try:
            new_dict[k] = dict_tobe_divided[k]
            del dict_tobe_divided[k]
        except:
            pass
    return dict_tobe_divided, new_dict

def extract_from_dict_by_prefix(dict_tobe_extracted, prefix):
    new_dict = {}
    lp = len(prefix)
    keys = list(dict_tobe_extracted.keys())
    for k in keys:
        if k[0:lp] == prefix:
            new_dict[k[lp:]] = dict_tobe_extracted[k]
            del dict_tobe_extracted[k]
    return new_dict, dict_tobe_extracted

# A convenient function that searches for font size attributes in a dictionary and 
# maps them from latex to font number.
def font_map(font_table, dct):
    for k,v in dct.items():
        if k in ['fontsize', 'labelsize', 'titlesize', 'title_fontsize']:
            dct[k] = font_table[dct[k]]
    return dct

# A convenient function that allows colors to be specified in different 
# ways.
def color_map(colors, color_table=None):
    # If color is an integer, standardize it.
    if isinstance(colors, (int, str)):
        colors = [colors]
        type = 'int'
    elif isinstance(colors, list):
        type = 'list'
        pass
    else:
        raise ValueError('Colors to color_map should be specified as an integer or list.') 

    # nothing to do if colors are already specified
    if isinstance(colors[0], str):
        return colors

    if color_table is None:
        logging.info('color_map: did not find a color table, skipping ...')
        return colors

    if isinstance(color_table, list):
        pass
    elif isinstance(color_table, Cycler):
        new_table = []
        for ele in list(color_table):
            new_table.append(ele['color'])
        color_table = new_table

    num_cols = len(color_table)
    true_colors = []
    for col in colors:
        true_colors.append(color_table[col % num_cols])
    if type == 'int':        
        return true_colors[0]
    else:
        return true_colors

def projection_map(map_string):
    return eval(map_string)

def decode_categorical_color_palette(palette, num_values):
    if palette[0:2] == 'cb':
        return(colors.cb[palette[2:]][num_values])

def generate_colorbar(ax=None, fig=None, bbox=None, sm=None, 
                      ticks=None, labels=None, cbaxis=None, scope=None,
                      **kwargs):
    cb_args = {}
    tick_args = {}
    title_args = {}
    label_args = {}
    for k,v in kwargs.items():
        if k in ['length', 'labelsize', 'labelcolor']:
            tick_args[k] = v
        if k in ['orientation', 'aspect', 'shrink', 'aspect', 'format', 'pad', 'location']:
            cb_args[k] = v
        if k in ['title', 'title_fontsize']:
            title_args[k] = v
        if k == 'prune':
            if kwargs[k] in ['left', 'both']:
                ticks = ticks[1:]
                labels = labels[1:]
            if kwargs[k] in ['right', 'both']:
                ticks = ticks[:-1]
                labels = labels[:-1]

    if cbaxis is None: 
        cbar = plt.colorbar(sm, ax=ax, **cb_args)
    else:
        cbaxis = list(cbaxis)   # if specified as a tuple
        if scope == 'figure':
            pass
        elif scope == 'subplot':
            # modify cbaxis
            cbaxis[0] = cbaxis[0] * bbox.width + bbox.x0
            cbaxis[1] = cbaxis[1] * bbox.height + bbox.y0
            cbaxis[2] = cbaxis[2] * bbox.width
            cbaxis[3] = cbaxis[3] * bbox.height
        else:
            raise ValueError(f'Invalid scope "{scope}".')

        cbar_ax = fig.add_axes(cbaxis)
        cbar = plt.colorbar(sm, cax=cbar_ax, **cb_args)
        cbar.ax.set_zorder(COLORBAR_ZORDER)

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels, **label_args)
    cbar.ax.tick_params(which='both', **tick_args)
    cbar.outline.set_linewidth(0)

    # Set scientific notation
    if labels is None:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # Use exponential if outside this range
        cbar.ax.yaxis.set_major_formatter(formatter)

    if 'title' in title_args.keys():
        cbar.set_label(title_args['title'], fontsize=title_args['title_fontsize'])

    cbar.minorticks_off()

    return

def generate_cmap(df, keyword, column, lim=(None,None), transform='linear',
                  set_bad=None, set_under=None):
    if df[column].dtypes.name == 'category':
        numcats = len(df[column].cat.categories)

        # generate cmap
        if keyword[0:2] == 'cb': # colorbrewer
            color_list = colors.cb[keyword[2:]][numcats]
            cmap = mcolors.ListedColormap(color_list)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.BoundaryNorm(
            boundaries=np.arange(numcats + 1) - 0.5, 
            ncolors=numcats))
        ticks = range(numcats)
        labels = df[column].cat.categories
    else:
        try:    # known cmap
            cmap = plt.get_cmap(keyword)
        except:
            if keyword[0:2] == 'cb': # colorbrewer
                maxcols = max(colors.cb[keyword[2:]].keys())
                color_list = colors.cb[keyword[2:]][maxcols] # as many colors as possible
            elif isinstance(keyword, list):
                color_list = keyword
            else:
                raise ValueError('Unsupported cmap format: should be a keyword or list.')
            cmap = mcolors.LinearSegmentedColormap.from_list(keyword, color_list, N=256)

        if lim[0] is None:
            vmin = np.nanmin(df[column])
        else:
            vmin = lim[0]
        if lim[1] is None:
            vmax = np.nanmax(df[column])
        else:
            vmax = lim[1]

        locator = MaxNLocator(nbins=TICK_NBINS)
        if transform == 'linear':
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            ticks = locator.tick_values(vmin, vmax)
        elif transform == 'log':
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            ticks = [10**x for x in 
                     locator.tick_values(np.log10(vmin), np.log10(vmax))]
            if 'int' in df[column].dtypes.name:
                ticks = [np.floor(x) for x in ticks]
    
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        sm.set_array([])  # Required for colorbar
        #locator = AutoLocator()
    
        # 5. Create labels
        labels = []
        for tick in ticks:
            if np.isclose(tick, np.round(tick)):  # If tick is very close to integer
                labels.append(f"{int(round(tick))}")
            else:
                labels.append(f"{tick:.2g}")  # Or however many decimals you want

    if set_bad is not None:
        cmap.set_bad(color=set_bad) # if the color is not specified correctly
    if set_under is not None:
        cmap.set_under(color=set_under) # if the color is below
    return sm, ticks, labels
        
def get_extent(ax, use_full_text_bounding_box=False):
    """Get true data extent (xmin, xmax, ymin, ymax) from a matplotlib Axes, including lines, patches, collections, and texts.
    
    Parameters:
    - ax: Matplotlib axes to inspect
    - use_full_text_bounding_box: If True, considers full text bounding box (not just anchor point)
    
    Returns:
    - xmin, xmax, ymin, ymax: The data bounds of the plot.
    """
    xvals = []
    yvals = []

    # 1. Main line(s)
    for line in ax.lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        if len(xdata) > 0:
            xvals.extend(xdata)
        if len(ydata) > 0:
            yvals.extend(ydata)

    # 2. Collections (e.g., scatter plots, CI bands, outliers)
    for coll in ax.collections:
        try:
            offsets = coll.get_offsets()
            if offsets.size > 0:
                xvals.extend(offsets[:, 0])
                yvals.extend(offsets[:, 1])
        except AttributeError:
            pass  # Some collections don't have offsets
        try:
            paths = coll.get_paths()
            for path in paths:
                verts = path.vertices
                if verts.size > 0:
                    xvals.extend(verts[:, 0])
                    yvals.extend(verts[:, 1])
        except AttributeError:
            pass

    # 3. Patches (e.g., bars, boxes)
    for patch in ax.patches:
        try:
            x0 = patch.get_x()
            y0 = patch.get_y()
            width = patch.get_width()
            height = patch.get_height()
            xvals.extend([x0, x0 + width])
            yvals.extend([y0, y0 + height])
        except AttributeError:
            pass  # Some patches might not have x/y (rare)

    # 4. Texts (optional: if you want to include full bounding box of text)
    if use_full_text_bounding_box:
        fig = ax.get_figure()
        renderer = fig.canvas.get_renderer()
        for text in ax.texts:
            bbox = text.get_window_extent(renderer=renderer)
            bbox_data = ax.transData.inverted().transform(bbox.get_points())
            xvals.extend(bbox_data[:, 0])
            yvals.extend(bbox_data[:, 1])
    else:
        # Just use the anchor point for text
        for text in ax.texts:
            pos = text.get_position()
            xvals.append(pos[0])
            yvals.append(pos[1])

    if len(xvals) == 0 or len(yvals) == 0:
        return None, None, None, None  # No data found

    xmin = np.min(xvals)
    xmax = np.max(xvals)
    ymin = np.min(yvals)
    ymax = np.max(yvals)

    return xmin, xmax, ymin, ymax

def cartopy_to_epsg(cartopy):
    if isinstance(cartopy, gplt.crs.WebMercator):
        return "EPSG:3857"
    elif isinstance(cartopy, gplt.crs.PlateCarree):
        return "EPSG:4326"
    # Add more mappings as needed
    return None

def standalone_legend(handles, labels, outfile, **kwargs):
    fig = plt.figure()
    fig.legend(handles, labels, **kwargs)
    fig.savefig(outfile, bbox_inches='tight')