from cmath import nan
import numpy as np
from scipy import spatial

# Functions
# ---------
# dist_matrices
# Monte_Carlo_uniform

 
def corr(test_vector, ref_vector):
    return test_vector / ref_vector - 1


def annulusArea(annulus_edge):
    """
    get the areas of concentric annuli

    Parameters
    ----------
    annulus_edges : list of float
        vector
    Returns
    -------
    annulus_area : list of float
        vector, one element smaller than annulus edges

    """
    annulus_area = []
    for ind, edge in enumerate(annulus_edge):
        if ind < annulus_edge.size-1:
            annulus_area.append(np.pi * (np.power(annulus_edge[ind+1], 2)
                                - np.power(edge, 2)))

    annulus_area = np.array(annulus_area)

    return annulus_area


def dist_matrices(coords, dist_self=-1):
    """
     Get all intercone distances of a mosaic

    Parameters
    ----------
    coord : str
        2D np.array of ints or floats corresponding to cone coordinates in a
        single mosaic. rows = cones, columns = [x,y]
    dims : {1, 2, 3}, default 3
        2 returns square matrix, 1 returns vector, 3 returns both
    self : {-1, 0}, default -1
        sets distance between a cone and itself as -1 or 0

    Returns
    -------
    np.array
        all intercone distances of the input cone coordinates, as a square
        matrix or vector (returns both if dims==3)

    """
    # if we have multiple cones...
    if len(coords.shape) == 2 and coords.shape[1] == 2:
        # 2D array of every intercone distance in the mosaic
        dists = spatial.distance_matrix(coords, coords)
    else:
        raise Exception('bad input for coords to dist_matrices()')

    # replace distance of cones to themselves with -1 if dim == -1
    # flags to handle a situation with 1 or no cones
    if dist_self == -1:
        for ind, cone in enumerate(coords[:, 0]):
            dists[ind, ind] = -1

    return dists


def distHist(dists, bin_width):
    # vectorize the matrix of distances
    dists = np.sort(np.reshape(dists, -1))

    # remove any -1s if present (indicate distance from self, if flagged to mark these in dist_matrices)
    dists = np.delete(dists, np.where(dists == -1))

    # calculate bin stuff
    # bin_edges = np.arange(0, np.ceil(max(dists) + bin_width), bin_width)
    num_bins = int(np.ceil((max(dists))/bin_width))
    bins = np.arange(0, num_bins * bin_width, step = bin_width)

    hist, bin_edges = np.histogram(dists, bins=bins)

    return hist, bin_edges


def setFirstAndSecondSpacifiedCone(coord, seed_ind, dists):
    set_cones = [seed_ind]
    set_coord = np.ones([2, 2]) * -1
    set_coord[0, :] = coord[seed_ind, :]

    furthest_cone = np.argmax(dists[seed_ind, :])
    set_cones.append(furthest_cone)
    set_coord[1, :] = coord[furthest_cone, :]
    
    return [set_cones, set_coord]


def removeVal(vals, val2remove):
    vals_removed = vals
    for val in val2remove:
        vals_removed = np.delete(vals_removed, np.nonzero(vals_removed == val))
    return vals_removed


def setThirdOnwardSpacifiedCone(coord, avail, set_cones, set_coord, next_cone_ind, dists):
    dists = np.where(dists > 0, dists, np.inf)
    # create a 2D array where each row is the current set of cone indices making up the
    # spacified mosaic followed by a hypothetical next cone-placement, with a row for 
    # every cone that is available to be set
    v1 = np.tile(np.array(set_cones), (avail.shape[0], 1))
    v2 = np.expand_dims(avail, 1)
    hyp_sp_sets = np.hstack((v1, v2))
    
    # identify the cone that, when added to the currently set spacified cones, minimizes 
    # the std of cone nearest neighbors
    nearest_neighbor_stds = []
    for hyp_set in hyp_sp_sets: 
        hyp_dists = dists[hyp_set, :][:, hyp_set]
        nearest_neighbor_stds.append(np.std(np.amin(hyp_dists, axis=0)))
    spaciest_cone = avail[np.argmin(np.array(nearest_neighbor_stds))]
    
    # set the next cone in the spacified mosaic data
    set_cones.append(spaciest_cone)
    set_coord[next_cone_ind][:] = coord[spaciest_cone][:]
    avail = removeVal(avail,[spaciest_cone])

    return [set_cones, set_coord, avail]


def spacified(num_coord, all_coord, num_sp):
    """
    asdf
    """
    if num_coord > 1 and all_coord.shape[0] >= num_coord:
        
        sp_coord = np.empty([num_sp, num_coord, 2], dtype=float)

        # get matrix of intercone distances
        dists = dist_matrices(all_coord)

        for sp in np.arange(0, num_sp):  # looop through mosaics to make
            avail_cones = np.arange(0, all_coord.shape[0])

            set_coord = np.ones([num_coord, 2]) * -1

            seed_ind = sp

            [set_cones, set_coord[0:2, :]] = setFirstAndSecondSpacifiedCone(all_coord, seed_ind, dists)

            avail_cones = removeVal(avail_cones, set_cones)

            for a in np.arange(0, num_coord):
                [set_cones, set_coord, avail_cones] = setThirdOnwardSpacifiedCone(all_coord, avail_cones, set_cones, set_coord, a, dists)
            sp_coord[sp, :, :] = set_coord
    else:
        sp_coord = nan
         
    return sp_coord


def monteCarlo_uniform(num_coord, num_mc, xlim, ylim):
    """
        generate array of monte carlo shuffled points distributed uniformly
        within a 2D range

        Parameters
        ----------
        num_mc : int
            number of Monte Carlo distributions to generate
        num_coords : int
            number of points to use in each Monte Carlo
        xlim : list of int
            2-element list, xlim[0] and xlim[1] are the lower and upper bounds
            of the uniform distribution (inclusive)
        ylim : list of int
            2-element list, xlim[0] and xlim[1] are the lower and upper bounds
            of the uniform distribution (inclusive)
        Returns
        -------
        np.array
            Monte Carlo coordinate matrix of shape [num_coords, 2, num_mc].
            1st column x, 2nd column y.
    """
    # [0] = cone index, [1] = (x,y), [2] = Monte Carlo trial
    mc_coord = np.zeros([num_mc, num_coord, 2])

    # randomly sample coordinates from a uniform distribution of the space
    for mc in np.arange(0, num_mc):
        mc_coord[mc, :, 0] = np.random.uniform(low=xlim[0], high=xlim[1],
                                               size=(num_coord))
        mc_coord[mc, :, 1] = np.random.uniform(low=ylim[0], high=ylim[1],
                                               size=(num_coord))
    return mc_coord


def monteCarlo_coneLocked(num_coord, all_coord, num_mc):
    mc_coord = np.zeros([num_mc, num_coord, 2])
    for mc in np.arange(0, num_mc):
        temp_x = all_coord[:,0]
        temp_y = all_coord[:,1]

        for c in np.arange(0, num_coord):
            # get random int between 1 and length temp 
            r = np.random.randint(low=0, high=len(temp_x), size=1)
            mc_coord[mc, c, 0] = temp_x[r]
            mc_coord[mc, c, 1] = temp_y[r]
            temp_x = np.delete(temp_x, r, 0)
            temp_y = np.delete(temp_y, r, 0)

    return mc_coord
