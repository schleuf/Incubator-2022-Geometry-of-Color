from cmath import nan
import random
import numpy as np
import mosaic_topog.calc as calc
import mosaic_topog.show as show
import pandas as pan
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

def updateEccentricity(xlsx_path, outputs):
    try:
        mosaic =  [bytes(z).decode('utf8') for z in outputs['mosaic_meta__mosaic']]
    except:
        try:
            mosaic =  outputs['mosaic_meta__mosaic']
        except:
            print('dude i dont know why it doesnt want to take these mosaics')
            
    xlsx = pan.read_excel(xlsx_path)
    subj = xlsx['Observer']
    ang = xlsx['Meridian']
    ecc1 = xlsx['Eccentricity']
    ecc2 = xlsx['Actual Eccentricity']
    ecc_updated = np.empty([len(mosaic), ])
    ecc_updated[:] = np.nan

    for m_ind, mos in enumerate(mosaic):
#         print(mos)
        for x_ind in np.arange(0, subj.shape[0]):
            str_e = str(ecc1[x_ind])
            if str_e[len(str_e)-2:len(str_e)] == '.0':
                e = str_e[0:len(str_e)-2]
            else:
                e = str_e
            if subj[x_ind] == 'SS':
                s = 'AO008R'
            if subj[x_ind] == 'RS':
                s = 'AO001R'
                
            test = s + '_' + ang[x_ind] + '_' + e
#             print('      ' + test)
            if mos == test:
#                 print('                 ' + 'MATCH')
                ecc_updated[m_ind] =  ecc2[x_ind]

    return ecc_updated

def reformat_stat_hists_for_plot(bin_edges, hist_mean, hist_std):
    # print(bin_edges.shape[0])
    hist_x = np.empty(bin_edges.shape[0]*2)
    hist_y = np.empty(bin_edges.shape[0]*2)
    hist_y_plus = np.empty(bin_edges.shape[0]*2)
    hist_y_minus = np.empty(bin_edges.shape[0]*2)
   
    hist_x[:] = np.nan
    hist_y[:] = np.nan
    hist_y_plus[:] = np.nan
    hist_y_plus[:] = np.nan

    nan_inds = np.nonzero([np.isnan(hist_mean[v]) for v in np.arange(0, hist_mean.shape[0])])[0]

    if nan_inds.shape[0] > 0:
        last_non_nan_ind = (nan_inds[nan_inds.shape[0]-1]+1) * 2

    else:
        last_non_nan_ind = (hist_mean.shape[0]) * 2

    for ind, bin in enumerate(np.arange(0, hist_x.shape[0])):
        hist_x[ind] = bin_edges[int(np.floor(ind/2))]
        # print(int(np.floor(ind/2)))
        
        if ind == 0 or ind == last_non_nan_ind+1:

            hist_y[ind] = 0
            hist_y_plus[ind] = 0
            hist_y_minus[ind] = 0

        elif ind < last_non_nan_ind+1 and ind > 0: 
            
            hist_y[ind] = hist_mean[int(np.floor((ind-1)/2))]
            if len(hist_std.shape) > 1:
                hist_y_plus[ind] = hist_std[0, int(np.floor((ind-1)/2))]
                hist_y_minus[ind] = hist_std[1, int(np.floor((ind-1)/2))]
            else:
                hist_y_plus[ind] = hist_mean[int(np.floor((ind-1)/2))] + hist_std[int(np.floor((ind-1)/2))]
                hist_y_minus[ind] = hist_mean[int(np.floor((ind-1)/2))] - hist_std[int(np.floor((ind-1)/2))]


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
        try:
            process_ind = np.nonzero([z == process for z in sim_to_gen])[0][0]
            for ind, sim in enumerate(sim_to_gen):
                if ind == process_ind:
                    sim_ind = ind 
            numSim = int(num_sim[sim_ind])
        except: 
            numSim = 0  
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
