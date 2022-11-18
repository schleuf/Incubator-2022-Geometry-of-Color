from cmath import nan
import random
import numpy as np
import mosaic_topog.calc as calc
import cv2

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


def randCol():
    rand_col = np.array([random.randint(0, 255),
                         random.randint(0, 255),
                         random.randint(0, 255)])
    rand_col = rand_col/255
    return rand_col


def trim_random_edge_points(coord, img_x, img_y, num_cones, buffer=0):
    # copied voronoi functionality from smp, bad form, needs to be a function
    img = np.zeros([int(img_x), int(img_y), 3])
    subdiv = cv2.Subdiv2D([0, 0, int(img_x), int(img_y)])

    points = []
    for p in np.arange(0, coord.shape[0]):
        points.append([coord[p, 0], coord[p, 1]])

    for p in np.arange(0, len(coord)):
        try:
            subdiv.insert(points[p])
        except:
            print('could not add coord to voronoi analysis: ' )
            print(np.array([coord[p, 0],coord[p, 1]],'int'))
    
    calc.delaunay(img, subdiv, (0, 0, 255))

    img_voronoi = np.zeros(img.shape, dtype=img.dtype)
    [facets, centers, bound, voronoi_area, voronoi_area_regularity,
     num_neighbor, num_neighbor_regularity] = calc.voronoi(img_voronoi, subdiv, buffer)
    
    temp = coord
    diff_num_cones = coord.shape[1] - num_cones
    print(diff_num_cones)
    if diff_num_cones > 0:
        unbound_inds = np.nonzero(bound==0)[0]
        print(unbound_inds)
        remove_inds = []
        num_current = temp.shape[0]
        while len(remove_inds) < diff_num_cones:
            ind_to_remove = np.round(np.random.randint(0,unbound_inds.shape[0]-1))
            if not ind_to_remove in remove_inds:
                remove_inds.append(ind_to_remove)
        print('inds to remove: ')
        print(ind_to_remove)

        temp[unbound_inds[ind_to_remove], :] = []
        print(len(temp))
    return temp


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
