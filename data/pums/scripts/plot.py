DESC='''plot functions
By AA
'''

from itertools import product
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub
import seaborn as sns

RC_PARAMS = {
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'legend.frameon': False,
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small'
        }

# latex fontsizes
FONT_TABLE=pd.DataFrame({'9': {'miniscule': 4, 'tiny': 5, 'scriptsize': 6, 'footnotesize': 7, 'small': 8, 'normalsize': 9, 'large': 10, 'Large': 11, 'LARGE': 12, 'huge': 14, 'Huge': 17, 'HUGE': 20}, '10': {'miniscule': 5, 'tiny': 6, 'scriptsize': 7, 'footnotesize': 8, 'small': 9, 'normalsize': 10, 'large': 11, 'Large': 12, 'LARGE': 14, 'huge': 17, 'Huge': 20, 'HUGE': 25}, '11': {'miniscule': 6, 'tiny': 7, 'scriptsize': 8, 'footnotesize': 9, 'small': 10, 'normalsize': 11, 'large': 12, 'Large': 14, 'LARGE': 17, 'huge': 20, 'Huge': 25, 'HUGE': 30}, '12': {'miniscule': 7, 'tiny': 8, 'scriptsize': 9, 'footnotesize': 10, 'small': 11, 'normalsize': 12, 'large': 14, 'Large': 17, 'LARGE': 20, 'huge': 25, 'Huge': 30, 'HUGE': 36}, '14': {'miniscule': 8, 'tiny': 9, 'scriptsize': 10, 'footnotesize': 11, 'small': 12, 'normalsize': 14, 'large': 17, 'Large': 20, 'LARGE': 25, 'huge': 30, 'Huge': 36, 'HUGE': 48}, '17': {'miniscule': 9, 'tiny': 10, 'scriptsize': 11, 'footnotesize': 12, 'small': 14, 'normalsize': 17, 'large': 20, 'Large': 25, 'LARGE': 30, 'huge': 36, 'Huge': 48, 'HUGE': 60}, '20': {'miniscule': 10, 'tiny': 11, 'scriptsize': 12, 'footnotesize': 14, 'small': 17, 'normalsize': 20, 'large': 25, 'Large': 30, 'LARGE': 36, 'huge': 48, 'Huge': 60, 'HUGE': 72}, '25': {'miniscule': 11, 'tiny': 12, 'scriptsize': 14, 'footnotesize': 17, 'small': 20, 'normalsize': 25, 'large': 30, 'Large': 36, 'LARGE': 48, 'huge': 60, 'Huge': 72, 'HUGE': 84}, '30': {'miniscule': 12, 'tiny': 14, 'scriptsize': 17, 'footnotesize': 20, 'small': 25, 'normalsize': 30, 'large': 36, 'Large': 48, 'LARGE': 60, 'huge': 72, 'Huge': 84, 'HUGE': 96}, '36': {'miniscule': 14, 'tiny': 17, 'scriptsize': 20, 'footnotesize': 25, 'small': 30, 'normalsize': 36, 'large': 48, 'Large': 60, 'LARGE': 72, 'huge': 84, 'Huge': 96, 'HUGE': 108}, '48': {'miniscule': 17, 'tiny': 20, 'scriptsize': 25, 'footnotesize': 30, 'small': 36, 'normalsize': 48, 'large': 60, 'Large': 72, 'LARGE': 84, 'huge': 96, 'Huge': 108, 'HUGE': 120}, '60': {'miniscule': 20, 'tiny': 25, 'scriptsize': 30, 'footnotesize': 36, 'small': 48, 'normalsize': 60, 'large': 72, 'Large': 84, 'LARGE': 96, 'huge': 108, 'Huge': 120, 'HUGE': 132}})

AXES_COLOR='#888888'

FIGSIZE=(15,8)
LINESTYLES=['solid','dashed']
MATHEMATICA=['#5e82b5','#e09c24','#8fb030','#eb634f','#8778b3','#c46e1a','#5c9ec7','#fdbf6f']
LINEWIDTH=[1]
GRAND_BUDAPEST=['#5b1a18','#fd6467','#f1bb7b','#d67236']
MARKERS=['.','o','v','^','s']
LINE_PROPS=pd.DataFrame(product(['solid','dashed'],MARKERS,MATHEMATICA[0:7],LINEWIDTH),columns=['style','marker','color','width'])

pd.options.display.float_format = '{:.10g}'.format
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

def initiate_plot(x,y):
    return plt.figure(figsize=[x,y])

def subplot_grids(fig,cols,rows):
    return fig.add_gridspec(rows,cols)
    # Use: ax=fig.add_subplot(gs[0,0])

def texify(string):
    string=sub('_','\_',string)
    string=sub('%','\%',string)
    return string

def set_plot_at_zero(axis):
    #axis.set_facecolor('#eeeeee')
    axis.spines['bottom'].set_position('zero')
    return

def set_plot_theme(axis):
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_color(AXES_COLOR)
    axis.spines['left'].set_color(AXES_COLOR)
    return

def set_grid(axis,minor=True):
    axis.grid(color='#cccccc',which='major',linewidth=1)
    if minor:
        axis.grid(True,color='#dddddd',which='minor',linewidth=.5)
    axis.tick_params(colors=AXES_COLOR,labelcolor='black',direction='inout')
    axis.set_axisbelow(True)
    return

def set_minor_tics(axis):
    axis.minorticks_on()
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    return

def set_hist(axis, mode='horizontal'):
    axis.spines['left'].set_visible(False)

    if mode=='horizontal':
        axis.grid(False,axis='x')
        axis.grid(True,axis='y')
    elif mode=='vertical':
        axis.grid(True,axis='x')
        axis.grid(False,axis='y')
    else:
        raise ValueError(f'Wrong mode {mode}.')

    axis.tick_params(colors=AXES_COLOR,labelcolor='black',direction='inout')
    axis.set_axisbelow(True)
    return

def square_grid_cells_by_x(axis, num_cells_x, num_cells_y, 
        labels_step_x, labels_step_y, type_x = None, type_y = None):

    # set xticks
    xmin, xmax = axis.get_xlim()
    xticks = np.linspace(xmin, xmax, num=num_cells_x+1)
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
    yticks = np.linspace(ymin, ymax, num=num_cells_y+1)
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
    axis.grid()

    return
    
def set_scientific(axis, xy):
    scientific_formatter = FuncFormatter(_scientific)
    if xy == 'y':
        axis.yaxis.set_major_formatter(scientific_formatter)
    else:
        axis.xaxis.set_major_formatter(scientific_formatter)
    return

def _scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.1E' % x


