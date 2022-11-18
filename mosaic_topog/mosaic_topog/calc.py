from cmath import nan
from pyexpat.errors import XML_ERROR_NO_ELEMENTS
import numpy as np
import random
from scipy import spatial
import matplotlib.pyplot as plt
import mosaic_topog.utilities as util
import mosaic_topog.show as show
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# Functions
# ---------
# dist_matrices
# Monte_Carlo_uniform

def slope_intercept(line):
    x1 = line[0][0]
    x2 = line[1][0]
    y1 = line[0][1]
    y2 = line[1][1]
    slope = (y2-y1)/(x2-x1)
    
    y_intercept = y1 - (slope*x1)
    return slope, y_intercept


def point_on_line(x, y, m, b):
    if (y == ((m * x) + b)):
        return True
    return False


def points_on_line(points, m, c):
    on_line = []
    for i, p in enumerate(points):
        if point_on_line(p[0], p[1], m, c):
            on_line.append(i)
    return on_line


def voronoi_region_metrics(bound, regions, vertices):
    num_neighbor = np.empty([len(regions), ])
    voronoi_area = np.empty([len(regions), ])
    num_neighbor[:] = np.nan
    voronoi_area[:] = np.nan

    # facets_e = util.explode_xy(np.unique(fac, axis=0))
    for r, reg in enumerate(regions):
        if bound[r]:
            cell_verts = np.empty([len(reg),2])
            for v, vert in enumerate(reg):
                cell_verts[v,:] = vertices[v]
            cell = Polygon(vertices[reg])
            voronoi_area[r] = cell.area
            num_neighbor[r] = int(np.unique(reg).shape[0])
    return voronoi_area, num_neighbor

def get_bound_voronoi_cells(vertices, regions,ridge_vertices):
    unbound_vert = []
    unbound_reg = []

    for vpair in ridge_vertices:
        if vpair[0] < 0:
            unbound_vert.append(vpair[1])
        elif vpair[1] < 0:
            unbound_vert.append(vpair[0])

    bound = np.ones([len(regions), ])
    for r, reg in enumerate(regions):
        if len(reg) == 0:
            bound[r] = 0
        for v in reg:
            if v in unbound_vert:
                unbound_reg.append(r)
    unbound_reg = np.unique(unbound_reg)
    bound[unbound_reg] = 0

    return bound


def voronoi(coord, buffer):

    vor = Voronoi(coord)
    vertices = vor.vertices
    regions = vor.regions
    ridge_vertices = vor.ridge_vertices
    voronoi_plot_2d(vor)
    show.scatt(vertices,'')

    bound = get_bound_voronoi_cells(vertices,regions,ridge_vertices)

    voronoi_area, num_neighbor = voronoi_region_metrics(bound, regions, vertices)

    if np.nanstd(voronoi_area) == 0:
        voronoi_area_mean = np.nan
        voronoi_area_std = np.nan
        voronoi_area_regularity = np.nan
    else:
        voronoi_area_mean = np.nanmean(voronoi_area[np.nonzero(bound)])
        voronoi_area_std = np.nanstd(voronoi_area[np.nonzero(bound)])
        voronoi_area_regularity = voronoi_area_mean/voronoi_area_std

    if np.nanstd(num_neighbor) == 0:
        num_neighbor_mean = np.nan
        num_neighbor_std = np.nan
        num_neighbor_regularity = np.nan
    else:
        num_neighbor_mean = np.nanmean(num_neighbor[np.nonzero(bound)])
        num_neighbor_std = np.nanstd(num_neighbor[np.nonzero(bound)])
        num_neighbor_regularity = num_neighbor_mean/num_neighbor_std

    return (regions, vertices, bound, 
            voronoi_area, voronoi_area_mean, voronoi_area_std, voronoi_area_regularity,
            num_neighbor, num_neighbor_mean, num_neighbor_std, num_neighbor_regularity)


def delaunay(img, subdiv, delaunay_colour):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_colour, 1)
            cv2.line(img, pt2, pt3, delaunay_colour, 1)
            cv2.line(img, pt3, pt1, delaunay_colour, 1)


def rectContains(rectangle, point):
    if (point[0] < rectangle[0] or
        point[1] < rectangle[1] or
        point[0] > rectangle[2] or
        point[0] > rectangle[3]): 
        return False
    return True


def shoelace_area(x_list, y_list):

    a1, a2 = 0, 0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]

    l=abs(a1-a2)/2
    # print('area')
    # print(l)
    # print('')
    return l

def hex_radius(density):
    radius = np.sqrt(np.sqrt(4/3)/density)
    return radius


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


def rectgrid(hex_radius, x_dim, y_dim):
    """
    generate array of hexagonally arranged points within a 2D range

    Parameters
    ----------
    hex_radius : float
        number of hexagonally spaced distributions to generate 
        (all the same if jitter = 0)
    x_len : int or float
        2-element list, xlim[0] and xlim[1] are the lower and upper bounds
        of the uniform distribution (inclusive)
    y_len : int or float
        2-element list, xlim[0] and xlim[1] are the lower and upper bounds
        of the uniform distribution (inclusive)
    jitter : 
    Returns
    -------
    np.array
        Monte Carlo coordinate matrix of shape [num_coords, 2, num_mc].
        1st column x, 2nd column y.
    """
    hex_radius = np.float_(hex_radius)

    x_len = x_dim[1] - x_dim[0]
    y_len = y_dim[1] - y_dim[0]
    
    #generate rectangular grid such that:
    #   points are contained in the same dimensions as the cone image
    #   x-spacing = the maximum spacing for cones of this density
    #   y-spacing = sin(60)*x-spacing (so that the distance between
    #               all points will be equal when every other row is
    #               displaced to go from rectangular -> hexagonal packing)
    x_rectgrid_spacing = hex_radius
    y_rectgrid_spacing = x_rectgrid_spacing
    xv, yv = np.meshgrid(np.arange(0, x_len, x_rectgrid_spacing),
                         np.arange(0, y_len, y_rectgrid_spacing),
                         sparse=False, indexing='xy')
    num_cones_placed = xv.shape[0] * xv.shape[1]

    # initialize output vars
    coord = np.empty([1, num_cones_placed, 2])
    coord[:] = np.nan

    # flatten the hexagonal spacing vectors, send to standard coordinate array
    x_vect = xv.flatten()
    y_vect = yv.flatten()
    coord[0, :, 0] = x_vect
    coord[0, :, 1] = y_vect

    return [coord]

def hexgrid(num2gen, hex_radius, x_dim, y_dim, jitter=0):
    """
    generate array of hexagonally arranged points within a 2D range

    Parameters
    ----------
    hex_radius : float
        number of hexagonally spaced distributions to generate 
        (all the same if jitter = 0)
    x_len : int or float
        2-element list, xlim[0] and xlim[1] are the lower and upper bounds
        of the uniform distribution (inclusive)
    y_len : int or float
        2-element list, xlim[0] and xlim[1] are the lower and upper bounds
        of the uniform distribution (inclusive)
    jitter : 
    Returns
    -------
    np.array
        Monte Carlo coordinate matrix of shape [num_coords, 2, num_mc].
        1st column x, 2nd column y.
    """
    hex_radius = np.float_(hex_radius)

    x_len = x_dim[1] - x_dim[0]
    y_len = y_dim[1] - y_dim[0]
    
    #generate rectangular grid such that:
    #   points are contained in the same dimensions as the cone image
    #   x-spacing = the maximum spacing for cones of this density
    #   y-spacing = sin(60)*x-spacing (so that the distance between
    #               all points will be equal when every other row is
    #               displaced to go from rectangular -> hexagonal packing)
    x_rectgrid_spacing = hex_radius
    y_rectgrid_spacing = x_rectgrid_spacing * (np.sqrt(3) / 2)  # SIN(60°)
    xv, yv = np.meshgrid(np.arange(0, x_len, x_rectgrid_spacing),
                         np.arange(0, y_len, y_rectgrid_spacing),
                         sparse=False, indexing='xy')
    num_cones_placed = xv.shape[0] * xv.shape[1]

    # initialize output vars
    coord = np.empty([num2gen, num_cones_placed, 2])
    coord[:] = np.nan

    if jitter:
        jitter_x_all = np.empty([num2gen, 1])
        jitter_x_all[:] = np.nan
        jitter_y_all = np.empty([num2gen, 1])
        jitter_y_all[:] = np.nan
    else:
        jitter_x_all = np.nan
        jitter_y_all = np.nan

    # jitter the hexagonal grid for as many mosaics as we are to make
    # *** i think that it would be better to generate a larger grid and have
    #     it randomly crop the image area from it
    for sp in np.arange(0, num2gen):
        if jitter:
            [xv, yv, jitter_x_all[sp], jitter_y_all[sp]] = rectGridJitter(xv, yv, -.5, x_rectgrid_spacing, y_rectgrid_spacing)
        
        # translate every other row by half the x-spacing (rect -> hex)
        xv[::2, :] += x_rectgrid_spacing/2

        # # view
        # fig_width = 21
        # fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        # ax.scatter(xv, yv)
        # ax.set_aspect('equal')

        # flatten the hexagonal spacing vectors, send to standard coordinate array
        x_vect = xv.flatten()
        y_vect = yv.flatten()
        coord[sp, :, 0] = x_vect
        coord[sp, :, 1] = y_vect

    return [coord, jitter_x_all, jitter_y_all]


def rectGridJitter(xv, yv, rand_modifier, x_rectgrid_spacing, y_rectgrid_spacing):
    
    # get randomized amounts of jitter in the x and y direction
    jitter_x = np.random.rand()
    jitter_x = (jitter_x + rand_modifier) * x_rectgrid_spacing
    jitter_y = np.random.rand()
    jitter_y = (jitter_y + rand_modifier) * y_rectgrid_spacing

    # apply jitter to the rectangular grid coordinates
    xv = xv + jitter_x
    yv = yv + jitter_y

    return xv, yv, jitter_x, jitter_y

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


def setThirdOnwardSpacifiedCone(coord, avail, set_cones, set_coord, next_cone_ind, dists, num_neighbors_to_average, std_of_blank_of_nearest_neighbor_distances):
    dists = np.where(dists > 0, dists, np.inf)
    # creatgooge a 2D array where each row is the current set of cone indices making up the
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
        sorted_neighbor_indices = np.argsort(hyp_dists, axis=1)
        num_averaged = np.min([num_neighbors_to_average, hyp_dists.shape[0]])
        distances_to_average = np.empty((hyp_dists.shape[0], num_averaged))
        distances_to_average[:] = np.nan

        inds_to_grab = sorted_neighbor_indices[:,0:num_averaged]

        for ind,hyp_set in enumerate(hyp_dists):
            distances_to_average[ind,:] = hyp_dists[ind, inds_to_grab[ind,:]]

        if std_of_blank_of_nearest_neighbor_distances == 'sum':
            
            nearest_neighbor_metric = np.sum(distances_to_average, axis = 1)
            
        elif std_of_blank_of_nearest_neighbor_distances == 'average':
            nearest_neighbor_metric = np.mean(distances_to_average, axis = 1)
        else:
            print('improper entry for std_of_blank_of_nearest_neighbor_distances, must be sum or average')
        
        if num_averaged == 1 and not (nearest_neighbor_metric == distances_to_average).all:
            print('SOMETHING IS WRONG WITH THE AVERAGING FOR NEAREST CONE DISTANCE')

        nearest_neighbor_stds.append(np.std(nearest_neighbor_metric, axis=0))

    spaciest_cone = avail[np.argmin(np.array(nearest_neighbor_stds))]
    
    # set the next cone in the spacified mosaic data
    set_cones.append(spaciest_cone)
    set_coord[next_cone_ind][:] = coord[spaciest_cone][:]
    avail = removeVal(avail, [spaciest_cone])

    return [set_cones, set_coord, avail]


def spacifyByNearestNeighbors(num_coord, all_coord, num_sp='all', num_neighbors_to_average=1, std_of_blank_of_nearest_neighbor_distances='average'):
    """
    asdf
    """
    if num_coord > 1 and all_coord.shape[0] >= num_coord:
        
        if num_sp == 'all':
            num2gen = all_coord.shape[0]
        else:
            num2gen = num_sp

        sp_coord = np.empty([num2gen, num_coord, 2], dtype=float)

        # get matrix of intercone distances
        dists = dist_matrices(all_coord)

        #randomize seed list
        seed_inds = np.arange(0, all_coord.shape[0])
        np.random.shuffle(seed_inds)
        
        for sp in np.arange(0, num_sp):  # looop through mosaics to make
            avail_cones = np.arange(0, all_coord.shape[0])

            set_coord = np.ones([num_coord, 2]) * -1

            seed_ind = seed_inds[sp]

            [set_cones, set_coord[0:2, :]] = setFirstAndSecondSpacifiedCone(all_coord, seed_ind, dists)

            avail_cones = removeVal(avail_cones, set_cones)

            for a in np.arange(0, num_coord):
                [set_cones, set_coord, avail_cones] = setThirdOnwardSpacifiedCone(all_coord, avail_cones, set_cones, set_coord, a, dists, num_neighbors_to_average, std_of_blank_of_nearest_neighbor_distances)
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


# ------------------------Functions from https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/


def draw_point(img, p, color) :
    cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )
 

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
 
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
 
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
 
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)
 
# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for i in xrange(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
 
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)
