from cmath import nan
import numpy as np

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
