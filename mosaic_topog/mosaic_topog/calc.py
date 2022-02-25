import numpy as np
from scipy import spatial

# Functions
# ---------
# dist_matrices
# Monte_Carlo_uniform


def dist_matrices(coords, dims=1, self=-1):
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

    # [cut] numzeros = coords.shape[0]

    # [cut] if len(coords.shape) == 2:    # 1 COORDINATE ARRAY

    # 2D array of every intercone distance in the mosaic
    if coords.shape == 2:
        dist_square = spatial.distance_matrix(coords, coords)
    elif coords.shape == 1:
        if coords.shape[0] == 0:
            dist_square = []
        else:
            dist_square = [0]

    if self == -1:
        for cone, blah in enumerate(coords[:, 0]):
            dist_square[cone, cone] = -1

    if dims == 2:
        return dist_square

    elif dims == 1:
        # vectorize the square distance matrix
        temp = np.reshape(dist_square, -1)

        # sort the vector
        dist_vect = np.sort(temp)

        return dist_vect

    # [cut]
        # findzeros = np.nonzero(dist_vect == 0)[0].size

        # if numzeros != findzeros:
        #     print('ERROR: bad logic about distance = 0. lreal num zeros is ',
        #           findzeros, ', expected ', numzeros)

        # dist_vect = dist_vect[numzeros+1:]    

    # elif len(coords.shape) == 3:    # >1 COORDINATE ARRAY 

    #     print(coords.shape)
    #     #initialize 3D array for stack of square distance matrices and 2D array for columns of sorted cone distances
    #     dist_square = np.zeros([coords.shape[0],coords.shape[0],coords.shape[2]])
    #     dist_vect = np.zeros([pow(coords.shape[0],2),coords.shape[2]])

    #     for z in np.arange(0,coords.shape[2]):

    #         # fill w stack of 2D arrays of intercone distances
    #         dist_square[:,:,z] = spatial.distance_matrix(coords[:,:,z],coords[:,:,z])

    #         #turn ^ into 2D array with every column being an array of distances
    #         temp = np.reshape(dist_square[:,:,z], -1)

    #         #sort the vectors
    #         dist_vect[:,z] = np.sort(temp)
    
    #     #remove zeros
    #     findzeros = np.nonzero(dist_vect[:,z]==0)[0].size
    #     if numzeros != findzeros:
    #         print('ERROR: bad logic about distance = 0. real # zeros is ',
    #               findzeros , ', expected ', numzeros)
    #     dist_vect = dist_vect[numzeros+1:,:]

    # else: #if there are zero cones or only one in this data set, doesn't make
    #     #sense to return anything

    #     dist_square = []
    #     dist_vect = []
    #     if not coords.size == 0 and not coords.size == 2:
    #         print('HEY unexpected shape for coordinate data, you should look into that')
    #         print('it''s shaped like this: ' + str(coords.shape))


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
