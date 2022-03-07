import numpy as np
from scipy import spatial
import mosaic_topog.show as show

# Functions
# ---------
# dist_matrices
# Monte_Carlo_uniform


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
    bin_edges = np.arange(1, int(max(dists) + bin_width), bin_width)

    # get histogram
    hist = np.histogram(dists, bins=bin_edges)[0]

    return hist, bin_edges


def MonteCarlo(num_coords, num_mc, mc_type, xlim, ylim):
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
    mc_coords = np.zeros([num_coords, 2, num_mc])

    # randomly sample coordinates from a uniform distribution of the space
    mc_coords[:, 0, :] = np.random.uniform(low=xlim[0], high=xlim[1],
                                           size=(num_coords, num_mc))
    mc_coords[:, 1, :] = np.random.uniform(low=ylim[0], high=ylim[1],
                                           size=(num_coords, num_mc))

    return(mc_coords)
    