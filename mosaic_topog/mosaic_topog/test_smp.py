import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import os
import h5py
import importlib
import glob
import yaml
import sys

import mosaic_topog.utilities as util
import mosaic_topog.flsyst as flsyst
import mosaic_topog.show as show
import mosaic_topog.calc as calc
import mosaic_topog.singleMosaicProcesses as smp


def intracone_dist_tests():
    """

    """
    print('hi imma blue test')
    intracone_dist_bin_edges


# ----------tests for intracone_dist----------

def intracone_dist_bin_edges():
    """

    """    # generate test mosaic
    spacing = 1
    coord_unit = 'AU'
    [coord, jitx, jity] = calc.hexgrid(1, spacing, [0, 10], [0, 10], 0)
    coord = coord.squeeze()
    
    smp.viewMosaic(coord, coord_unit, 'w', 'unit test')

    #send this test mosaic to the intracone_dist_common
    bin_width = .1
    dist_area_norm = False
    [dist, mean_nearest, std_nearest, hist, bin_edge, annulus_area] = smp.intracone_dist_common(coord, bin_width, dist_area_norm)

    # set up inputs to plot
    xlab = 'distance, ' + coord_unit
    tit = 'intracone distance (' + str(coord.shape[0]) + " cones)"
    ylab = 'bin count (binsize = ' + str(bin_edge[1]-bin_edge[0])
    x = bin_edge[1:]/2

    # view histogram
    
    ax = show.line(x, hist, 'unit test', plot_col='w', title=tit, xlabel=xlab, ylabel=ylab)

    ax.figure


# ----------tests for basic_stats----------


# ----------run
if __name__ == '__main__':
    intracone_dist_tests()
    