from cmath import nan
import random
import numpy as np
import mosaic_topog.calc as calc
import mosaic_topog.show as show
import pandas as pan
import pytest
from numpy.random import default_rng
import matplotlib.pyplot as plt


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
def cropData(img, coord, cropCenter):

    og_halfX = int(img.shape[1]/2)
    og_halfY = int(img.shape[0]/2)
    crop_halfX = int(cropCenter[0]/2)
    crop_halfY =  int(cropCenter[1]/2)
    
    crop_lim = [[og_halfX - crop_halfX, og_halfX + crop_halfX],
                [og_halfY - crop_halfY, og_halfY + crop_halfY]]

    crop_img = img[crop_lim[1][0]:crop_lim[1][1], 
                   crop_lim[0][0]:crop_lim[0][1]]
      
    crop_coord = coord
    
    lessX  = coord[:,0] < crop_lim[0][0]
    moreX = coord[:,0] > crop_lim[0][1]
    lessY = coord[:,1] < crop_lim[1][0]
    moreY = coord[:,1] > crop_lim[1][1]

    crop_coord[lessX, :] = np.nan
    crop_coord[moreX, :] = np.nan
    crop_coord[lessY, :] = np.nan
    crop_coord[moreY,:] = np.nan
#     print(crop_coord)   
    

    crop_coord[:,0] = crop_coord[:,0] - crop_lim[0][0]
    crop_coord[:,1] = crop_coord[:,1] - crop_lim[1][0]
    
    
    return crop_img, crop_coord





def initialize_nested_lists(depth, width):
    if depth <= 0:
        return None  # or return [] if you want to end with lists
    return [initialize_nested_lists(depth - 1, width) for _ in range(width)]


def attemptQs(dat, perc, lab = ''):
    try:
        qs = np.quantile(dat, perc)
    except:
        qs = NaN([len(perc),])
        # print(lab)
        # print('could not get quantiles, probably because empty, non-flat, of NaN-containing data set')
    return qs


def printValByType(value, pref='    '):
    if isinstance(value, dict):
        printDict(value, pref = pref + '    ')
    elif isinstance(value, str):
        print(pref + value)
    elif isinstance(value, list):
        printList(value, pref = pref + '    ')
    elif isinstance(value,  np.bytes_):
        data = bytes(value).decode("utf8")
        print(pref + data)
    elif isinstance(value,  bytes):
        data = bytes(value).decode("utf8")
        print(pref + data)
    else:
        try:
            print(pref + str(value.shape))
        except:
            print(type(value))
            print('problems printing list element....')


def printList(thelist, pref = '    '):
    print(pref + str(len(thelist)))
    for value in thelist:
        printValByType(value, pref + '    ')
    
def printDict(dictionary, pref = '    '):
    # GPT code. not tested yet. 
    for key, value in dictionary.items():
        print(pref + key)
        printValByType(value, pref + '   ')


def makeItNaN(toNaN):
    NaNs = []
    for n in toNaN:
        NaNs.append(np.nan)
    return NaNs


def scatterBitsByTit(tit):
    # angle_colors = ['#44AA99', '#117733', '#999933', '#DDCC77']
    angle_colors = ['#1976D2', '#03A9F4', '#4DD0E1', '#B2DFDB']
    angles = ['temporal', 'inferior', 'nasal', 'superior']

    if 'AO001R' in tit:
        marker='o'
    if 'AO008R' in tit:
        marker='^'
    if 'JWcrop' in tit:
        marker = 's'
    if 'M5crop' in tit:
        marker = '*'

    for a_ind, a in enumerate(angles):
    
        if a in tit:
            color = angle_colors[a_ind]
            # print(color)
    
    if not color:
        color = 'k'
        
    return marker, color


def removeCones(coord, num2rem):
    coord = np.array(coord)
    x = coord[:, 0]
    y = coord[:, 1]
    
    rng = default_rng()
    random_inds = rng.choice(coord.shape[0], size=num2rem, replace=False).astype('int64')
    less_coord = np.empty([coord.shape[0]-num2rem, 2])
    less_coord[:] = np.nan
    
    if less_coord.shape[0] > 0:
        less_coord[:, 0] = np.delete(x, random_inds)
        less_coord[:, 1] = np.delete(y, random_inds)
    return less_coord


def modCoord(this_S, cone_mod, this_A = np.array([np.nan])):
    if cone_mod < 0:
        return(removeCones(this_S, -1 * cone_mod))
    elif cone_mod > 0:
        return(addCones(this_S, cone_mod, this_A))
    else:
        return this_S
    

def addCones(sub_coord, num2add, all_coord):
    more_coord=NaN([sub_coord.shape[0] + num2add, sub_coord.shape[1]])
    more_coord[0:sub_coord.shape[0],:] = sub_coord
    
    A_ind2_S = all_coord_match_inds(all_coord, sub_coord)
    non_sub_ind = np.nonzero([z not in A_ind2_S for z in np.arange(0,all_coord.shape[0])])[0].astype('int64')
    
    rng = default_rng()
    random_inds = rng.choice(non_sub_ind, size=num2add, replace=False).astype('int64')
    more_coord[sub_coord.shape[0]:sub_coord.shape[0] + num2add,:] = all_coord[random_inds,:]
    
    return more_coord


def NaN(shape):
    nans = np.empty(shape)
    nans[:] = np.nan
    return nans


def deNaN(vect):
    vect = np.array(vect).flatten()
    non_nan = np.nonzero([not np.isnan(z) for z in vect])[0].astype('int64')
    new_vect = vect[non_nan]
    return new_vect


def forceAspect(ax,aspect=1):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.set_aspect(abs((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))/aspect)


def all_coord_match_inds(A_coord, this_coord):
    A_ind2_S = np.empty([this_coord.shape[0],])
    A_ind2_S[:] = np.nan
    badmatch = False
    for c_ind in np.arange(0, this_coord.shape[0]):
        # find ind of matching cone in all-cone coord
        c = this_coord[c_ind,:]
        if not np.isnan(c[0]):
            try:
                [A_coord[z,0] == c[0] and A_coord[z,1] == c[1] for z in np.arange(A_coord.shape[0])]
                match = np.nonzero([A_coord[z,0] == c[0] and A_coord[z,1] == c[1] for z in np.arange(A_coord.shape[0])])[0][0]
                A_ind2_S[c_ind] = match
            except:
                badmatch = True
                print('WARNING could not find match to S-cone coordinate ' + str(c_ind))
                print(c_ind)
                print(c)
                print(np.nonzero([A_coord[z,0] == c[0] and A_coord[z,1] == c[1] for z in np.arange(A_coord.shape[0])])[0])
    if badmatch:
        print('SOMETHING WRONG WITH THIS SET OF COORDINATES')
        # fig, ax = plt.subplots()
        # ax.scatter(A_coord[:,0], A_coord[:, 1], marker = 'x')
        # ax.scatter(this_coord[:,0], this_coord[:,1], marker = '+')
    # else:
    #     print('ALL GOOD IN THE HOOD')
    non_nan = np.nonzero([not np.isnan(z) for z in A_ind2_S])[0]
    A_ind2_S = A_ind2_S[non_nan]
    A_ind2_S = A_ind2_S.astype('int64')
    
    return A_ind2_S


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
            elif subj[x_ind] == 'RS':
                s = 'AO001R'
            elif subj[x_ind] == 'JWcrop':
                s = 'JWcrop'
            elif subj[x_ind] == 'M5crop':
                s = 'M5crop'
                
            test = s + '_' + ang[x_ind] + '_' + e
#             print('      ' + test)
            if mos == test:
#                 print('                 ' + 'MATCH')
                ecc_updated[m_ind] =  ecc2[x_ind]

    return ecc_updated



def getCF_DpP(xlsx_path, outputs):
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
    CF = xlsx['Degrees per Pixel']
    CF_DpP = NaN([len(mosaic), ])

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
            elif subj[x_ind] == 'RS':
                s = 'AO001R'
            elif subj[x_ind] == 'JWcrop':
                s = 'JWcrop'
            elif subj[x_ind] == 'M5crop':
                s = 'M5crop'
                
            test = s + '_' + ang[x_ind] + '_' + e
#             print('      ' + test)
            if mos == test:
#                 print('                 ' + 'MATCH')
                CF_DpP[m_ind] =  CF[x_ind]

    return CF_DpP




def cum_hist_to_line_plottables(bin_edges, hist):
    # print(bin_edges.shape[0])
    hist_x = np.empty(bin_edges.shape[0]*2)
    hist_y = np.empty(bin_edges.shape[0]*2)
   
    hist_x[:] = np.nan
    hist_y[:] = np.nan

    nan_inds = np.nonzero([np.isnan(hist[v]) for v in np.arange(0, hist.shape[0])])[0]

    if nan_inds.shape[0] > 0:
        last_non_nan_ind = (nan_inds[nan_inds.shape[0]-1]+1) * 2

    else:
        last_non_nan_ind = (hist.shape[0]) * 2

    for ind, b in enumerate(np.arange(0, hist_x.shape[0])):
        hist_x[ind] = bin_edges[int(np.floor(ind/2))]
        # print(int(np.floor(ind/2)))
        
        if ind == 0:
            hist_y[ind] = 0

        elif ind > last_non_nan_ind:
            hist_y[ind] = 1

        elif ind < last_non_nan_ind+1 and ind > 0: 
            hist_y[ind] = hist[int(np.floor((ind-1)/2))]

    return hist_x, hist_y






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
