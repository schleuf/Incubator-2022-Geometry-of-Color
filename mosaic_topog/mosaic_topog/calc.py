from cmath import nan
import numpy as np
from scipy import spatial

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
    # bin_edges = np.arange(0, np.ceil(max(dists) + bin_width), bin_width)
    num_bins = int(np.ceil((max(dists))/bin_width))
    bins = np.arange(0, num_bins * bin_width, step = bin_width)

    hist, bin_edges = np.histogram(dists, bins=bins)

    return hist, bin_edges


def spacified(num_coord, all_coord, num_sp):
    """
    """
    if num_coord > 0:
        sp_coord = np.empty([num_coord, 2, num_sp], dtype=float)
        seed_inds = []

        # get matrix of intercone distances
        dists = dist_matrices(all_coord)[0]

        for sp in np.arange(0, num_sp):  # looop through mosaics to make
        
            temp_inds = np.arange(0, all_coord.shape[0])

            # set first cone
            seedable_inds = temp_inds[np.nonzero(np.isin(temp_inds, seed_inds) == False)]
            coord = np.ones([num_coord, 2]) * -1
            seed_ind = seedable_inds[np.random.randint(0, high=len(seedable_inds), size=1)]
            coord[0] = all_coord[seed_ind][0]
            seed_inds.append(seed_ind)
            np.delete(temp_inds, seed_ind)

            if num_coord > 1:
                for c in np.arange(1, num_coord):
                    if c == 1:  # set 2nd cone as far as possible from the 1st cone
                        print(dists)
                        print(dists.shape)
                        furthest_cone = np.argmax(dists[seed_ind])
                        coord[1] = all_coord[furthest_cone]
                        np.delete(temp_inds, np.nonzero(temp_inds == furthest_cone))

                    else:  # set each additional cone at the location that minimizes the std of nearest neighbor distance
                        spaciest_cone = -1
                        min_std = dists[seed_ind, furthest_cone]
                        
                        for t in temp_inds:
                            # test adding each cone to the list, finding the std of the list of nearest neighbor distances
                            #  when the cone is added.  select the cone that minimizes the std of nearest neighbor distance. 
                            temp_sp = np.ones([c+1, 2])*-1
                            temp_sp[0:c] = coord[0:c][:]
                            temp_sp[c] = all_coord[t]
                            temp_dists = dist_matrices(temp_sp)[0]
                            temp_dists[np.nonzero(temp_dists == -1)] = nan
                            print(dists)
                            min_dists = np.amin(temp_dists, axis=0)
                            print(min_dists)
                            std_min_dists = np.std(min_dists)
                            print(std_min_dists)

                            if std_min_dists < min_std:
                                min_std = std_min_dists
                                spaciest_cone = t

                        coord[c, :] = all_coord[t, :]
                        np.delete(temp_inds, np.nonzero(temp_inds == t))

            sp_coord[:, :, sp] = coord
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
