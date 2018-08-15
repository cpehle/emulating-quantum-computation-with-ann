import matplotlib
from cycler import cycler

def set_plot_parameters():
    size_name = 'revtex'
    normal_figure_width = 243./72.
    normal_figure_default_height = 2.
    wide_figure_width = 482./72.
    wide_figure_default_height = 4. 
    fontsize = 8
    titlesize = 9
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker', [' ']))
    linewidth = 0.4
    matplotlib.rcParams['lines.linewidth'] = linewidth
    matplotlib.rcParams['patch.linewidth'] = linewidth
    matplotlib.rcParams['axes.linewidth'] = linewidth
    matplotlib.rcParams['axes.titlesize'] = titlesize
    matplotlib.rcParams['grid.linewidth'] = linewidth
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['xtick.major.width'] = linewidth
    matplotlib.rcParams['xtick.minor.width'] = linewidth
    matplotlib.rcParams['axes.prop_cycle'] = monochrome
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['figure.autolayout'] = True
    matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
    matplotlib.rcParams['font.monospace'] = ['Computer Modern Typewriter']
    return normal_figure_width, wide_figure_width