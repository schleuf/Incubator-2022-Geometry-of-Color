from cmath import nan
import random
import numpy as np
import mosaic_topog.calc as calc
import mosaic_topog.show as show
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pytest

# Functions
# ---------
# mapStringToNan
#   returns a dict in which input strings are keys to a numpy array 
#   containing only a NaN
#
# indsNotInList
#    return the indices of a list where the nth value is not contained
#    in another list
#
#  removeListInds
#    remove more than one index from a list at once
#


def reformat_stat_hists_for_plot(bin_edges, hist_mean, hist_std):
    hist_x = np.empty(hist_mean.shape[0]*2+1)
    hist_y = np.empty(hist_mean.shape[0]*2+1)
    hist_y_plus = np.empty(hist_mean.shape[0]*2+1)
    hist_y_minus = np.empty(hist_mean.shape[0]*2+1)

    hist_x[:] = np.nan
    hist_y[:] = np.nan
    hist_y_plus[:] = np.nan
    hist_y_plus[:] = np.nan

    hist_x[0] = 0
    hist_y[0] = 0
    hist_y_plus[0] = 0
    hist_y_minus[0] = 0

    for ind, bin in enumerate(np.arange(0, hist_mean.shape[0]-1)):
        if ind < hist_mean.shape[0]-1:
            hist_x[ind*2+1:ind*2+3] = [bin_edges[ind], bin_edges[ind]]
            hist_y[ind*2+1:ind*2+3] = [hist_mean[ind], hist_mean[ind+1]]

            hist_y_plus[ind*2+1:ind*2+3] = [hist_mean[ind] + hist_std[ind], hist_mean[ind+1] + hist_std[ind+1]]
            hist_y_minus[ind*2+1:ind*2+3] = [hist_mean[ind] - hist_std[ind], hist_mean[ind+1] - hist_std[ind+1]]
        else:
            hist_x[ind*2+1] = bin_edges[ind]
            hist_y[ind*2+1] = hist_mean[ind]
            hist_y_plus[ind*2+1] = hist_mean[ind] + hist_std[ind]
            hist_y_minus[ind*2+1] = hist_mean[ind] - hist_std[ind]
            
    return hist_x, hist_y, hist_y_plus, hist_y_minus


def numSim(process, num_sim, sim_to_gen):
    """
    return the number of simulations to be generated
    
    Parameters
    ----------
    process : the process we're finding out how many to generate
    num_sim : a single number (means same number generated for all mosaics) 
              or a list of numbers corresponding to each simulation to be 
              generated
    sim_to_gen : user input variable, list of simulations to generate

    Returns
    -------
    numSim : the number of simulations to produce for the given process 
"""
    if len(num_sim) < len(sim_to_gen) or len(num_sim) > len(sim_to_gen):
        if len(num_sim) == 1:
            numSim = int(num_sim[0])
        else:
            raise Exception('inappropriate # of values in input variable ""numSim""')
    
    elif len(num_sim) == len(sim_to_gen): 
        for ind, sim in enumerate(sim_to_gen):
            if sim == process:
                sim_ind = ind 
        numSim = int(num_sim[sim_ind])
    
    else:
        raise Exception('inappropriate # of values in input variable ""numSim""')

    return numSim


def randCol():
    rand_col = np.array([random.randint(0, 255),
                         random.randint(0, 255),
                         random.randint(0, 255)])
    rand_col = rand_col/255
    return rand_col


def trim_random_edge_points(coord, num_cones, img_x, img_y):
    if len(coord.shape) == 2:
        coord = np.expand_dims(coord, 0)
    diff_num_cones = coord.shape[1] - num_cones

    # print('trimming: num cone difference')
    # print(diff_num_cones)
    if diff_num_cones > 0:
        temp = []
        [regions, vertices, ridge_vertices, ridge_points, point_region] = calc.voronoi(coord)
        bound_cones = calc.get_bound_voronoi_cells(coord, img_x, img_y)
        bound_regions = []
        
        for m in np.arange(0, coord.shape[0]):
            bound_regions.append(np.zeros([len(regions[m]), ]))
            bound_reg = point_region[m][np.array(np.nonzero(bound_cones[m])[0], dtype=int)]
            bound_regions[m][bound_reg] = 1
            unbound_inds = np.nonzero(bound_regions[m] == 0)[0]
            unbound_cone_inds = np.nonzero(np.isin(point_region[m], unbound_inds))[0]        
            remove_inds = []
            while len(remove_inds) < diff_num_cones:
                ind_to_remove = np.round(np.random.randint(0,unbound_cone_inds.shape[0]-1))
                if ind_to_remove not in remove_inds:
                    remove_inds.append(ind_to_remove)
            temp.append(np.delete(coord[m], unbound_cone_inds[remove_inds], 0))
    else:
        temp = coord
    coord = np.array(temp)
    return coord


#EXPLODE X AND Y
def explode_xy(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0])
        yl.append(xy[i][1])
    return xl,yl


def vector_zeroPad(vector_to_pad, num_preceding, num_following):
    if not (len(vector_to_pad.shape) == 1
            or ((len(vector_to_pad.shape) == 2)
            and (vector_to_pad.shape[0] == 1
                 or vector_to_pad[1] == 1))):
        raise Exception('calc.vector_zeroPad received a vector to be padded of inappropriate size')

    prefix = np.zeros([num_preceding, ])
    suffix = np.zeros([num_following, ])

    padded_vector = np.append(np.append(prefix, vector_to_pad), suffix)

    return padded_vector


def mapStringToLocal(proc_vars, local, data_to_set={}):
    """
    set dictonary keys/value pairs based on a list of strings corresponding to local variables 

    Parameters
    ----------
    proc_vars : 
    local : 
    data_to_set :

    Returns
    -------
    data_to_set : 
    """
    for var in proc_vars:
        data_to_set[var] = local[var]
    return data_to_set

def mapStringToNan(string, data_to_set={}):
    for s in string:
        data_to_set[s] = np.asarray([nan])
    return data_to_set


def indsNotInList(check_list, ref_list):
    """ 
    find the indices of a list where the nth value is not contained
    in another list

    Parameters
    ----------
    check_list : list
    ref_list : list

    Returns
    -------
    pop_inds : list of int
    

    """
    pop_inds = []
    for ind, name in enumerate(check_list):
        if name not in ref_list:
            pop_inds.append(ind)
    return pop_inds


def removeListInds(edit_list, pop_inds):
    """
    remove more than one index from a list at once

    Parameters
    ----------
    edit_list : list
    pop_inds : list of int

    Returns
    -------
    edit_list : list

    """
    subtract = 0

    if edit_list and pop_inds:
        for ind in pop_inds:
            poppable = ind - subtract
            edit_list.pop(poppable)
            subtract = subtract + 1

    return edit_list
