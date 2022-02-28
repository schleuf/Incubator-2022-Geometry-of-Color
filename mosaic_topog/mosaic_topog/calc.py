import numpy as np
from scipy import spatial

# Functions
# ---------
# dist_matrices
# Monte_Carlo_uniform


def dist_matrices(coords, dims=3, self=-1):
    """
     Get all intercone distances of a mosaic

    Parameters
    ----------
    coord : str
        2D np.array of ints or floats corresponding to cone coordinates in a
        single mosaic. rows = cones, columns = [x,y]
    dims : {1, 2}, default 1
        2 returns square matrix, 1 returns vector
    self : {-1, 0}, default -1
        sets distance between a cone and itself as -1 or 0

    Returns
    -------
    np.array
        all intercone distances of the input cone coordinates, as a square
        matrix or vector.

    """
    print(coords.shape)
    # if we have multiple cones...
    if len(coords.shape) == 2 and coords.shape[1] == 2:
        # 2D array of every intercone distance in the mosaic
        dist_square = spatial.distance_matrix(coords, coords)
        one_cone = False
    # if there is only one cone...
    elif len(coords.shape) == 1 and coords.shape[0] == 2:
        dist_square = [0]
        one_cone = True
    elif coords.shape[0] == 0:
        if dims == 1 or dims == 2:
            return []
        elif dims == 3:
            return[]
    else:
        raise Exception('bad input for coords to dist_matrices()')

    # replace distance of cones to themselves with -1 if dim == -1
    # flags to handle a situation with 1 or no cones
    if self == -1:
        if not one_cone:
            iter = np.arange(0, coords.shape[0])
        else:
            iter = np.array([0])
        for cone in iter:
            if not one_cone:
                dist_square[cone, cone] = -1
            else:
                dist_square[cone] = -1

    # return the square matrix, a vector, or both
    if dims == 2:
        return dist_square
    elif (dims == 1 or dims == 3) and not one_cone:
        # vectorize the square distance matrix
        temp = np.reshape(dist_square, -1)
        # sort the vector
        dist_vect = np.sort(temp)
        if dims == 1:
            return dist_vect
    if dims == 3:
        return dist_square, dist_vect


def MonteCarlo_uniform(num_mc, num_coords, xlim, ylim):
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
