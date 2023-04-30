from cmath import nan
from pyexpat.errors import XML_ERROR_NO_ELEMENTS
import numpy as np
import random
from scipy import spatial
import matplotlib.pyplot as plt
import mosaic_topog.utilities as util
import mosaic_topog.show as show
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d, distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import sys
import math

eps = sys.float_info.epsilon

# Functions
# ---------
# dist_matrices
# Monte_Carlo_uniform

def poisson_interval(k, alpha=0.05): 
    """
    copied from https://stackoverflow.com/questions/14813530/poisson-confidence-interval-with-numpy
    uses chisquared info to get the poisson interval. Uses scipy.stats 
    (imports in function). 
    """
    from scipy.stats import chi2
    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
    return low, high


def rotate_around_point(xy, radians, origin=(0, 0)):
    """
    FROM https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def in_box(towers, bounding_box):
    """
    from https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
    """
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))


def line_intersection(x1, y1, x2, y2):
    line1 = [[x1[0], y1[0]], [x1[1], y1[1]]]
    [s1, i1] = slope_intercept(line1)
    line2 = [[x2[0], y2[0]], [x2[1], y2[1]]]
    [s2, i2] = slope_intercept(line2)

    #solve for intercept
    x = (i2-i1)/(s1-s2)

    # print('X')
    # print(x)

    try1 = np.around(x*s1 + i1, 3)
    try2 = np.around(x*s2 + i2, 3)

    # print('Y1')
    # print(try1)
    # print('Y2')
    # print(try2)

    if try1 == try2:
        return [x, try1]
    else:
        return [np.nan, np.nan]


def slope_intercept(line):
    x1 = line[0][0]
    x2 = line[1][0]
    y1 = line[0][1]
    y2 = line[1][1]
    slope = (y2-y1)/(x2-x1)    
    y_intercept = y1 - (slope * x1)
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


def voronoi_region_metrics(bound, regions, vertices, point_region):
    num_mos = len(vertices)
    max_vertlen = np.nanmax([len(vertices[v]) for v in np.arange(0, num_mos)])
    max_reglen = np.nanmax([len(regions[r]) for r in np.arange(0, num_mos)])

    num_neighbor = np.empty([num_mos, max_reglen])
    voronoi_area  = np.empty([num_mos, max_reglen])
    voronoi_area_mean = np.empty([num_mos, ])
    voronoi_area_std = np.empty([num_mos, ])
    voronoi_area_regularity = np.empty([num_mos, ])
    num_neighbor_mean = np.empty([num_mos, ])
    num_neighbor_std = np.empty([num_mos, ])
    num_neighbor_regularity = np.empty([num_mos, ])

    num_neighbor[:] = np.nan
    voronoi_area[:] = np.nan
    voronoi_area_mean[:] = np.nan
    voronoi_area_std[:] = np.nan
    voronoi_area_regularity[:] = np.nan
    num_neighbor_mean[:] = np.nan
    num_neighbor_std[:] = np.nan
    num_neighbor_regularity[:] = np.nan

    max_neighbors = 0

    for m in np.arange(0, num_mos):
        for r, reg in enumerate(regions[m]):
            if bool(bound[m][r]):
                if len(reg) > 2:
                    cell_verts = np.empty([len(reg), 2])
                    for v, vert in enumerate(reg):
                        cell_verts[v, :] = vertices[m][vert]
                    poly = Polygon(cell_verts)
                    voronoi_area[m, r] = poly.area
                    num_neighbor[m, r] = int(np.unique(reg).shape[0])
        
        voronoi_area_mean[m] = (np.nanmean(voronoi_area[m, :]))
        voronoi_area_std[m] = (np.nanstd(voronoi_area[m, :]))     

        if np.nanstd(voronoi_area[m]) == 0:
            voronoi_area_regularity[m] = np.nan
        else:
            voronoi_area_regularity[m] = (voronoi_area_mean[m]/voronoi_area_std[m])

        num_neighbor_mean[m] = (np.nanmean(num_neighbor[m]))
        num_neighbor_std[m] = (np.nanstd(num_neighbor[m]))

        if np.nanstd(num_neighbor[m]) == 0:
            num_neighbor_regularity[m] = np.nan
        else:
            num_neighbor_regularity[m] = (num_neighbor_mean[m]/num_neighbor_std[m])
     
        max_neighbors = int(np.nanmax([max_neighbors, np.nanmax(num_neighbor[m])]))


    return [voronoi_area, voronoi_area_mean,
            voronoi_area_std, voronoi_area_regularity,
            num_neighbor, num_neighbor_mean,
            num_neighbor_std, num_neighbor_regularity]


def getVoronoiNeighbors(coord, vertices, regions, ridge_vertices, ridge_points, point_region, bound_regions, bound_cones):
    
    num_mos = len(vertices)
    max_vertlen = np.nanmax([len(vertices[v]) for v in np.arange(0, num_mos)])
    max_reglen = np.nanmax([len(regions[r]) for r in np.arange(0, num_mos)])

    max_neighbors = 0

    for m in np.arange(0, num_mos):
        for r, reg in enumerate(regions[m]):
            if bool(bound_regions[m][r]):
                max_neighbors = int(np.nanmax([max_neighbors, np.nanmax(len(reg))]))
  
    neighbors_cones = np.empty([num_mos, coord.shape[1], max_neighbors])
    neighbors_cones[:] = np.nan

    for m in np.arange(0, num_mos):
        # kw = {}
        # kw['figsize'] = 15
        # ax = show.plotKwargs(kw, '')
        # ax = show.scatt(np.squeeze(coord[m,:,:]),'', ax=ax, plot_col='r')
        this_coord = coord[m, :, :]
        # for r, ridge in enumerate(ridge_vertices[m]):
        #     if ridge[0] >= 0 and ridge[1] >= 0:
        #         v1 = vertices[m][ridge[0]]
        #         v2 = vertices[m][ridge[1]]
        #         col = util.randCol()
        #         x = np.array([v1[0], v2[0]])
        #         y = np.array([v1[1], v2[1]])
        #         c1 = int(ridge_points[m][r][0])
        #         c2 = int(ridge_points[m][r][1])

        #         if (np.all(x >= np.nanmin(this_coord[:,0])) and np.all(x <= np.nanmax(this_coord[:,0])) and
        #             np.all(y >= np.nanmin(this_coord[:,1])) and np.all(y <= np.nanmax(this_coord[:,1]))):
                    
        #             ax = show.line(x, y, '', plot_col = 'g', ax = ax)
        #             if bound_cones[m][c1] and bound_cones[m][c2]:
        #                 col = util.randCol()
        #                 ax = show.line(x, y, '', ax=ax, plot_col=col)
                        
        #                 x = np.array([this_coord[c1, 0], this_coord[c2, 0]])
        #                 y = np.array([this_coord[c1, 1], this_coord[c2, 1]])
        #                 # ax = show.line(x, y, 'ridges and ridge points', ax=ax, plot_col=col, linewidth=2)

        # ax2 = show.scatt(np.squeeze(this_coord), '')
        
        for c in np.arange(0, this_coord.shape[0]):
            cone = this_coord[c, :]
            # col = util.randCol()
            # ax2 = show.scatt(np.squeeze(cone), '', ax=ax2, plot_col=col, linewidth=.5)
            neighbs = []
            for r, ridge in enumerate(ridge_points[m]):
                if c in ridge and bound_cones[m][c]:
                    if ridge[0] == c:
                        neighbs.append(ridge[1])
                    elif ridge[1] == c:
                        neighbs.append(ridge[0])
            
            neighbors_cones[m, c, 0:len(neighbs)] = np.array(neighbs)
            
            # for n, neighb in enumerate(neighbs):
            #     if n < len(neighbs)-2:
            #         x = np.array([coord[m, neighb, 0], coord[m, neighbs[n+1], 0]])
            #         y = np.array([coord[m, neighb, 1], coord[m, neighbs[n+1], 1]])
            #         ax2 = show.line(x, y, 'neighbor scribbles', ax=ax2, plot_col=col)
            #     else:
            #         x = np.array([coord[m, neighb, 0], coord[m, neighbs[0], 0]])
            #         y = np.array([coord[m, neighb, 1], coord[m, neighbs[0], 1]])
            #         ax2 = show.line(x, y, 'neighbor scribbles', ax=ax2, plot_col=col)

    return neighbors_cones


def get_bound_voronoi_cells(coord, img_x, img_y):
    """
    adapted from https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
    """

    num_mos = coord.shape[0]
    bound_cones = np.ones([num_mos, coord.shape[1]])
    bound_cones = np.array(bound_cones, dtype=int)
    # print('IMAGE DIMENSIONS')
    # print(img_x)
    # print(img_y)
    bounding_box = np.array([img_x[0], img_x[1], img_y[0], img_y[1]])
    dist_cones = dist_matrices(np.squeeze(coord[0, :, :]), dist_self=np.nan)
    dist_mins = np.nanmin(dist_cones, axis=0)
    avg_dist_min = np.nanmean(dist_mins)
    buffer = avg_dist_min/5

    for m in np.arange(0, num_mos):
        points_center = coord[m, :, :]

        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_center[:, 0] - bounding_box[0])

        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_center[:, 0])

        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_center[:, 1] - bounding_box[2])
        
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_center[:, 1])
        points = np.append(points_center,
                           np.append(np.append(points_left,
                                               points_right,
                                               axis=0),
                                     np.append(points_down,
                                               points_up,
                                               axis=0),
                                     axis=0),
                           axis=0)
        
        #Compute Voronoi
        # fig = plt.figure()
        # ax = fig.gca()
        # ax = show.scatt(points_center,'',plot_col='w',ax=ax)
        # ax = show.line([bounding_box[0], bounding_box[1]],[bounding_box[2], bounding_box[2]], '', ax=ax, plot_col='w') #bottom
        # ax = show.line([bounding_box[0], bounding_box[1]],[bounding_box[3], bounding_box[3]], '', ax=ax, plot_col='w') #top
        # ax = show.line([bounding_box[0], bounding_box[0]],[bounding_box[2], bounding_box[3]], '', ax=ax, plot_col='w') #left
        # ax = show.line([bounding_box[1], bounding_box[1]],[bounding_box[2], bounding_box[3]], '', ax=ax, plot_col='w') #right
        # ax = show.scatt(points_left,'',plot_col='r',ax=ax)
        # ax = show.scatt(points_right,'',plot_col='y',ax=ax)
        # ax = show.scatt(points_up,'',plot_col='b',ax=ax)
        # ax = show.scatt(points_down,'',plot_col='g',ax=ax)
        non_nan = np.array(np.nonzero(~np.isnan(points[:,0]))[0], dtype=int)

        vor = Voronoi(points[non_nan])

        unbound_inds = []
        for c in np.arange(0, points_center.shape[0]):
            r = vor.point_region[c]
            cx = points_center[c, 0]
            cy = points_center[c, 1]
            # ax.plot(cx, cy, 'ro')

            for v in vor.regions[r]:
                x = vor.vertices[v][0]
                y = vor.vertices[v][1]

                if ((x <= bounding_box[0] + buffer) or 
                    (x >= bounding_box[1] - buffer) or
                    (y <= bounding_box[2] + buffer) or
                    (y >= bounding_box[3] - buffer) or 
                    (np.isinf(x) or np.isinf(y))):
                    # print('FAIL')
                    # print(x)
                    # print(y)
                    # print('')
                    unbound_inds.append(c)
                # else:
                    # print('PASS')
                    # print(x)
                    # print(y)
                    # print('')
                    # ax.plot(x,y,'go')

        unbound_inds = np.unique(unbound_inds)
        bound_cones[m, unbound_inds] = 0
        
        # ax.plot(points_center[np.nonzero(bound_cones[m,:])[0], 0], points_center[np.nonzero(bound_cones[m,:])[0], 1], 'ko')

        # ax.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'k-')
        # ax.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'k-')
        # ax.plot([bounding_box[0], bounding_box[0]], [bounding_box[2], bounding_box[3]], 'k-')
        # ax.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'k-')
        # plt.xlim(img_x)
        # plt.ylim(img_y)
    return bound_cones
    

def voronoi_innards(coord):
    coord_2D = np.squeeze(coord)
    non_nan = np.array(np.nonzero(~np.isnan(coord_2D[:,0]))[0], dtype=int)
    vor = Voronoi(coord_2D[non_nan, :])
    vertices = (vor.vertices)
    regions = (vor.regions)
    ridge_vertices = (vor.ridge_vertices)
    ridge_points = (vor.ridge_points)
    point_region = (vor.point_region)

    return (regions, vertices, ridge_vertices, ridge_points, point_region)


def voronoi(coord):
    if len(coord.shape) == 2:
        coord = np.expand_dims(coord, axis=0)
    regions = []
    vertices = []
    ridge_vertices = []
    point_region = []
    ridge_points = []

    for i in np.arange(0, coord.shape[0]):
        regions.append([])
        vertices.append([])
        ridge_vertices.append([])
        point_region.append([])
        ridge_points.append([])
        [regions[i], vertices[i],
            ridge_vertices[i], ridge_points[i], point_region[i]] = voronoi_innards(coord[i, :, :])

    return (regions, vertices, ridge_vertices, ridge_points, point_region)


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
    if dist_self == -1 or np.isnan(dist_self):
        for ind, cone in enumerate(coords[:, 0]):
            dists[ind, ind] = dist_self

    return dists


def distHist(dists, bin_width, offset_bin = False):
    # vectorize the matrix of distances
    dists = np.sort(np.reshape(dists, -1))

    # remove any -1s if present (indicate distance from self, if flagged to mark these in dist_matrices)
    dists = np.delete(dists, np.where(np.isnan(dists)))

    # calculate bin stuff
    # bin_edges = np.arange(0, np.ceil(max(dists) + bin_width), bin_width)
    if offset_bin:
        offset = bin_width/2
    else:
        offset = 0

    num_bins = int(np.ceil((max(dists)) / bin_width))
    bins = np.arange(0 + offset, (num_bins * bin_width) + offset, step = bin_width)

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


def genGrid(hex_radius, hex_x, hex_y):
    x_rectgrid_spacing = hex_radius
    y_rectgrid_spacing = x_rectgrid_spacing * (np.sqrt(3) / 2)  # SIN(60°)
    xv, yv = np.meshgrid(np.arange(hex_x[0], hex_x[1], x_rectgrid_spacing),
                         np.arange(hex_y[0], hex_y[1], y_rectgrid_spacing),
                         sparse=False, indexing='xy')
    xv = xv.round()
    yv = yv.round()

    num_cones_placed = xv.shape[0] * xv.shape[1]
    # translate every other row by half the x-spacing (rect -> hex)
    xv[::2, :] += x_rectgrid_spacing/2

    # flatten the hexagonal spacing vectors, send to standard coordinate array
    x_vect = xv.flatten()
    y_vect = yv.flatten()

    return x_vect, y_vect


def hexgrid(num2gen, hex_radius, x_dim, y_dim, randomize=False, target_num_cones=0):
    """
    generate array of hexagonally arranged points within a 2D range
    such that:
    points are contained in the same dimensions as the cone image
    x-spacing = the maximum spacing for cones of this density
    y-spacing = sin(60)*x-spacing (so that the distance between
                   all points will be equal when every other row is
                   displaced to go from rectangular -> hexagonal packing)
    Parameters
    ----------
    num2gen: 
        number of hexagonally spaced distributions to generate 
    hex_radius : float
        spacing of the hexagonal array
    x_len : int or float
        2-element list, xlim[0] and xlim[1] are the lower and upper bounds
        of the uniform distribution (inclusive)
    y_len : int or float
        2-element list, xlim[0] and xlim[1] are the lower and upper bounds
        of the uniform distribution (inclusive)
    Returns
    -------
    np.array
        hexgrid coordinate matrix of shape [num_generated, num_coords, 2].
        2nd column x, 3rd column y.
    """

    if randomize: # requires a larger initial grid to handle translation & rotation

        # calc the size of the hexgrid to generate
        x_len = x_dim[1] - x_dim[0]
        y_len = y_dim[1] - y_dim[0]
        jitter = hex_radius
        img_diagonal = np.sqrt(np.power(x_len, 2) + np.power(y_len, 2))
        jitt_diagonal = 2 * np.sqrt(2 * np.power(jitter,2))
        hex_width = img_diagonal + jitt_diagonal

        # calc the offset of the hexgrid to generate
        diff_x = hex_width - x_len
        diff_y = hex_width - y_len
        offset_x = -1 * (diff_x/2)
        offset_y = -1 * (diff_y/2)

        #set the gridsize
        hex_x = [offset_x, offset_x + hex_width -1]
        hex_y = [offset_y, offset_y + hex_width -1]
    else: 
        hex_x = x_dim
        hex_y = y_dim
    
    initial_hex_radius = np.float_(hex_radius)

    # initialize output vars
    coord_list = []

    max_hex = 0

    hex_radii_used = []

    num_cones_final = []
    
    # plt.scatter(x_vect, y_vect)

    for sp in np.arange(0, num2gen):
        density_optimized = False
        overshot_target = False
        gen_tests = []
        num_cones_gen = []
        hex_radii_tested = []
        test_radius = initial_hex_radius
        delta_radius = 0
        count = 0
        while not density_optimized:
            count = count + 1
            test_radius = test_radius + delta_radius
            # print('test_radius')
            # print(test_radius)
            hex_radii_tested.append(test_radius)
            x_vect, y_vect = genGrid(test_radius, hex_x, hex_y)
            temp_coord = np.empty([1, x_vect.shape[0], 2])
            temp_coord[0, :, 0] = x_vect
            temp_coord[0, :, 1] = y_vect

            if randomize: 
                if count == 1:
                    # jitter the hexgrid
                    temp_coord[0, :, :], jitt_x, jitt_y = jitter_grid(np.squeeze(temp_coord[0, :, :]), hex_radius)

                    # rotate the hexgrid
                    temp_coord[0, :, :] , deg_rot = rotate_grid(np.squeeze(temp_coord[0, :, :]), [x_dim[1]/2, y_dim[1]/2])

                else:
                    temp_coord[0, :, :], jitt_x, jitt_y = jitter_grid(np.squeeze(temp_coord[0, :, :]), hex_radius, jitt_x, jitt_y)

                    # rotate the hexgrid
                    temp_coord[0, :, :] , deg_rot = rotate_grid(np.squeeze(temp_coord[0, :, :]), [x_dim[1]/2, y_dim[1]/2], deg_rot)

                # find hexgrid points within the image bounds
                inds_in_img_bounds = in_box(np.squeeze(temp_coord[0, :, :]), [x_dim[0], x_dim[1], y_dim[0], y_dim[1]])
                bound_hex = temp_coord[0, inds_in_img_bounds, :]
                crop_coords = bound_hex

            else:
                crop_coords = temp_coord[0, :, :]

            gen_tests.append(crop_coords)

            #redo if there aren't enough points in the bounds
            num_cones = crop_coords.shape[0]
            num_cones_gen.append(num_cones)
            # print('num cones')
            # print(num_cones)
            if num_cones == target_num_cones:
                density_optimized = True
                keep_coords = gen_tests[len(gen_tests)-1]
                hex_radii_used.append(test_radius)

            elif num_cones < target_num_cones:
                # print('TOO FEW CONES')
                if delta_radius == .01:
                    # print('CROSSED')
                    overshot_target = True
                    density_optimized = True
                    # diff_this = target_num_cones - num_cones
                    # diff_prev = target_num_cones - num_cones_gen[len(gen_tests)-2]
                    # if diff_this > diff_prev:
                    #     keep_coords = gen_tests[len(gen_tests)-2]
                    #     hex_radii_used.append(hex_radii_tested[len(gen_tests)-2])
                    # elif diff_this < diff_prev: 
                    #     keep_coords = gen_tests[len(gen_tests)-1]
                    #     hex_radii_used.append(test_radius)
                    # elif diff_this == diff_prev:
                    rand_1_or_2 = np.random.randint(1,3)
                    keep_coords =gen_tests[len(gen_tests)-rand_1_or_2]
                    hex_radii_used.append(hex_radii_tested[len(gen_tests)-rand_1_or_2])
                else:
                    delta_radius = -.01

            elif num_cones > target_num_cones:
                # print('TOO MANY CONES')
                if delta_radius == -.01:
                    # print('CROSSED')
                    overshot_target = True
                    density_optimized = True
                    diff_this = target_num_cones - num_cones
                    diff_prev = target_num_cones - num_cones_gen[len(gen_tests)-2]
                    # if diff_this > diff_prev:
                    #     keep_coords = gen_tests[len(gen_tests)-2]
                    #     hex_radii_used.append(hex_radii_tested[len(gen_tests)-2])
                    # elif diff_this < diff_prev: 
                    #     keep_coords = gen_tests[len(gen_tests)-1]
                    #     hex_radii_used.append(test_radius)
                    # elif diff_this == diff_prev:
                    rand_1_or_2 = np.random.randint(1,3)
                    keep_coords =gen_tests[len(gen_tests)-rand_1_or_2]
                    hex_radii_used.append(hex_radii_tested[len(gen_tests)-rand_1_or_2])

                else:
                    delta_radius = .01
        # print('')

        coord_list.append(keep_coords)

        num_cones_final.append(coord_list[len(coord_list)-1].shape[0])
        max_hex = np.amax([max_hex, coord_list[sp].shape[0]])

    coord = np.empty([num2gen, max_hex, 2])
    for sp in np.arange(0, num2gen):
        coord[sp, 0:coord_list[sp].shape[0], :] = coord_list[sp]

        # # # view
        # fig_width = 10
        # fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        # ax.scatter(np.squeeze(coord[sp,:,0]), np.squeeze(coord[sp,:,1]))
        # ax.set_aspect('equal')

    return coord, num_cones_final, hex_radii_used


# def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
#     #  copied from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
#     """Rotate a point around a given point.
    
#     I call this the "high performance" version since we're caching some
#     values that are needed >1 time. It's less readable than the previous
#     function but it's faster.
#     """
#     x, y = xy
#     offset_x, offset_y = origin
#     adjusted_x = (x - offset_x)
#     adjusted_y = (y - offset_y)
#     cos_rad = math.cos(radians)
#     sin_rad = math.sin(radians)
#     qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
#     qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

#     return qx, qy


# def rectGridJitter(xv, yv, rand_modifier, x_rectgrid_spacing, y_rectgrid_spacing):
    
#     # get randomized amounts of jitter in the x and y direction
#     jitter_x = np.random.rand()
#     jitter_x = (jitter_x + rand_modifier) * x_rectgrid_spacing
#     jitter_y = np.random.rand()
#     jitter_y = (jitter_y + rand_modifier) * y_rectgrid_spacing

#     # apply jitter to the rectangular grid coordinates
#     xv = xv + jitter_x
#     yv = yv + jitter_y

#     return xv, yv, jitter_x, jitter_y


# def setFirstAndSecondSpacifiedCone(coord, seed_ind, dists):
#     set_cones = [seed_ind]
#     set_coord = np.ones([2, 2]) * -1
#     set_coord[0, :] = coord[seed_ind, :]

#     furthest_cone = np.argmax(dists[seed_ind, :])
#     set_cones.append(furthest_cone)
#     set_coord[1, :] = coord[furthest_cone, :]
    
#     return [set_cones, set_coord]


def removeVal(vals, val2remove):
    vals_removed = vals
    for val in val2remove:
        vals_removed = np.delete(vals_removed, np.nonzero(vals_removed == val))
    return vals_removed


# def setThirdOnwardSpacifiedCone(coord, avail, set_cones, set_coord, next_cone_ind, dists):
#     dists = np.where(dists > 0, dists, np.inf)
#     # creatgooge a 2D array where each row is the current set of cone indices making up the
#     # spacified mosaic followed by a hypothetical next cone-placement, with a row for 
#     # every cone that is available to be set
#     v1 = np.tile(np.array(set_cones), (avail.shape[0], 1))
#     v2 = np.expand_dims(avail, 1)
#     hyp_sp_sets = np.hstack((v1, v2))
    
#     # identify the cone that, when added to the currently set spacified cones, minimizes 
#     # the std of cone nearest neighbors
#     nearest_neighbor_stds = []
#     #max_neighbors = np.nanmax(num_neighbor[np.nonzero(bound_regions)])
#     for hyp_set in hyp_sp_sets: 
#         #print(coord[hyp_set,:])
#         vor = Voronoi(coord[hyp_set, :])
#         neighbs = np.empty([hyp_set.shape[0], 20])
#         neighbs[:] = np.nan
#         for c_ind, c in enumerate(hyp_set):
#             neighb = []
#             for pr in vor.ridge_points:
#                 if pr[0] == c_ind:
#                     neighb.append(pr[1])
#                 elif pr[1] == c_ind:
#                     neighb.append(pr[0])
#             print(neighb)
#             print(hyp_set[neighb])
#             ax = show.scatt(coord[np.array(hyp_set[neighbs], dtype=int),:],'neighbors', ax=ax)
#             break
#             neighbs[c_ind, 0:len(neighb)] = neighb

#         hyp_dists = dists[hyp_set, :][:, hyp_set]
#         #sorted_neighbor_indices = np.argsort(hyp_dists, axis=1)
#         distances_to_average = np.empty((hyp_dists.shape[0], 20))
#         distances_to_average[:] = np.nan  

#         for ind, hyp_d in enumerate(hyp_dists):
#             inds_to_grab = np.array(neighbs[ind, np.nonzero(~np.isnan(neighbs[ind,:]))[0]], dtype=int)
#             distances_to_average[ind, 0:len(inds_to_grab)] = hyp_dists[ind, inds_to_grab]
        
#         nearest_neighbor_metric = np.nanmean(distances_to_average, axis = 1)
        
#         nearest_neighbor_stds.append(np.std(nearest_neighbor_metric, axis=0))
#     print(nearest_neighbor_metric)
#     print(nearest_neighbor_stds)
#     spaciest_cone = avail[np.argmin(np.array(nearest_neighbor_stds))]
    
#     # set the next cone in the spacified mosaic data
#     set_cones.append(spaciest_cone)
#     set_coord[next_cone_ind][:] = coord[spaciest_cone][:]
#     avail = removeVal(avail, [spaciest_cone])

#     return [set_cones, set_coord, avail]


def jitter_grid(coord, max_jitter, jitt_x = np.nan, jitt_y = np.nan):
    
    if np.isnan(jitt_x) or np.isnan(jitt_y):
        jitt_x = (np.random.rand(1) * 2 * max_jitter) - max_jitter
        jitt_y = (np.random.rand(1) * 2 * max_jitter) - max_jitter
    
    coord[:, 0] = coord[:, 0] + jitt_x
    coord[:, 1] = coord[:, 1] + jitt_y

    return coord, jitt_x, jitt_y


def rotate_grid(coord, origin, deg_rot = np.nan):

    if np.isnan(deg_rot):
        deg_rot = np.random.rand(1) * 2 * math.pi   

    rot_x, rot_y = rotate_around_point(coord[:, :], deg_rot, origin)
    coord[:, 0] = rot_x
    coord[:, 1] = rot_y

    return coord, deg_rot


def coneLocked_hexgrid_mask(all_coord, num2gen, cones2place, x_dim, y_dim, hex_radius):
    spaced_coord = []
    max_coord = 0
    num_cones_final = []
    hex_radii_used = []

    for mos in np.arange(0, num2gen):
        num_cones_final.append([])
        hex_radii_used.append([])
        hex_coord, num_cones_final[mos], hex_radii_used[mos] = hexgrid(1, hex_radius, x_dim, y_dim, randomize = True, target_num_cones = cones2place)
        
        # get intercone distance histogram for cones versus bound hexgrid points
        dist_mat = distance.cdist(np.squeeze(hex_coord), all_coord, 'euclidean')

        # ax = show.scatt(all_coord, 'test grid', plot_col='y')
        # ax = show.scatt(np.squeeze(hex_coord), 'test grid', ax=ax)

        # for every bound hexgrid point, identify its closest cone and add to the spacified coordinates
        min_dist_cone_inds = np.argmin(dist_mat, axis=1)
        spaced_coord.append(all_coord[min_dist_cone_inds, :])

        max_coord = np.nanmax([max_coord, spaced_coord[mos].shape[0]])

        # ax = show.scatt(all_coord, 'test grid', plot_col='y', s=3)
        # ax = show.scatt(np.squeeze(hex_coord), 'test grid', ax=ax)
        # ax = show.scatt(spaced_coord[mos], 'test_grid', ax=ax, plot_col = 'r', s=.5)
        # ax = show.line([x_dim[0], x_dim[0]], [y_dim[0], y_dim[1]], '', ax=ax, plot_col = 'r')
        # ax = show.line([x_dim[1], x_dim[1]], [y_dim[0], y_dim[1]],'', ax=ax, plot_col = 'r')
        # ax = show.line([x_dim[0], x_dim[1]], [y_dim[0], y_dim[0]],'', ax=ax, plot_col = 'r')
        # ax = show.line([x_dim[0], x_dim[1]], [y_dim[1], y_dim[1]],'', ax=ax, plot_col = 'r')

    temp = np.empty([num2gen, max_coord, 2])
    temp[:] = np.nan
    for mos in np.arange(0, num2gen):
        temp[mos, 0:spaced_coord[mos].shape[0], :] = spaced_coord[mos]
    spaced_coord = temp
    num_cones_final = np.array(num_cones_final)
    hex_radii_used = np.array(hex_radii_used)
    return spaced_coord, num_cones_final, hex_radii_used
        
    

        
# def spacifyByNearestNeighbors(num_coord, all_coord, num2gen):
#     """
#     asdf
#     """
#     if num_coord > 1 and all_coord.shape[0] >= num_coord:
        
#         sp_coord = np.empty([num2gen, num_coord, 2], dtype=float)

#         # get matrix of intercone distances
#         dists = dist_matrices(all_coord)

#         #randomize seed list
#         seed_inds = np.arange(0, all_coord.shape[0])
#         np.random.shuffle(seed_inds)
        
#         for sp in np.arange(0, num2gen):  # looop through mosaics to make
#             avail_cones = np.arange(0, all_coord.shape[0])

#             set_coord = np.ones([num_coord, 2]) * -1

#             seed_ind = seed_inds[sp]

#             [set_cones, set_coord[0:2, :]] = setFirstAndSecondSpacifiedCone(all_coord, seed_ind, dists)

#             avail_cones = removeVal(avail_cones, set_cones)
#             ax = show.plotKwargs({},'')
#             for a in np.arange(0, num_coord):
#                 print(str(a) + '...')
#                 [set_cones, set_coord, avail_cones] = setThirdOnwardSpacifiedCone(all_coord, avail_cones, set_cones, set_coord, a, dists)
#                 ax = show.scatt(set_coord,'spacifying', ax=ax)
#                 ax.draw
#             sp_coord[sp, :, :] = set_coord
#     else:
#         sp_coord = nan
         
#     return sp_coord


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



def dmin(all_coord, num2gen, num2place, max_dist, prob_rej_type) :
    dmin_coord = np.zeros([num2gen, num2place, 2]) 

    for mos in np.arange(0, num2gen):

        #these coordinate lists will be depleted as cones are set for this mosaic
        temp_x = all_coord[:,0]
        temp_y = all_coord[:,1]
        
        placed = []
        enough_placed = False

        # ax = show.plotKwargs({}, [])
        # ax = show.scatt(all_coord, 'sim ' + str(mos), s=60, ax=ax)
        # print('    working on dmin mosaic ' + str(mos))
        tries = 0
        sim_failed = False

        while not enough_placed:
            tries = tries + 1
            # get random int between 1 and the number of cones available to be placed 
            r = np.random.randint(low=0, high=len(temp_x), size=1)

            # candidate coordinates for cone placement
            candidate_x = temp_x[r]
            candidate_y = temp_y[r]
        
            # print(candidate_x)
            # print(candidate_y)
            # print('')

            # ax = show.scatt(np.concatenate([candidate_x, candidate_y], axis=0), 'sim ' + str(mos), ax=ax, plot_col = 'y', s = 40)
            # for every cone that has already been set, test the pobability that
            # a cone would be found here under the dmin model 
    
            test_cone_passed = True
            for ind, p in enumerate(placed):
                px = p[0]
                py = p[1]
                dist = np.sqrt((candidate_x - px)**2 + (candidate_y - py)**2)

                if prob_rej_type == 'all_or_none':

                    if dist < max_dist:
                        prob_placement = 0
                    else:
                        prob_placement = 1

                elif prob_rej_type == 'inverse_distance_squared':

                    if dist < max_dist:
                        prob_placement = 1/((max_dist-dist)**2)
                    else:
                        prob_placement = 1
                

                # draw a random number between 0 and 1.  
                r2 = np.random.rand(1)

                # if r is less than probability of placement for any already-placed
                # cone, the candidate cone fails the test.  
                if r2[0] > prob_placement:
                    test_cone_passed = False

            # if the candidate placement has passed all tests with previously-set
            # cones, place a cone here
            if test_cone_passed:
                placed.append(np.concatenate([candidate_x, candidate_y], axis=0))
                temp_x = np.delete(temp_x, r, 0)
                temp_y = np.delete(temp_y, r, 0)
                # ax = show.scatt(np.concatenate([candidate_x, candidate_y], axis=0), 'sim ' + str(mos), ax = ax, plot_col = 'r', size = 20)
                # print('        placed cone ' + str(len(placed)) + '/' + str(num2place) + ': ' + str(candidate_x) + ', ' + str(candidate_y))
                tries = 0
                
            if len(placed) == num2place:
                enough_placed = True

            if tries > 1000:
                print('1000 tries to set cone, dmin simulation aborted')
                sim_failed = True
                break

        if not sim_failed:
            temp_coord = np.empty([num2place,2])
            for p in np.arange(0,num2place):
                temp_coord[p,:] = placed[p]

            dmin_coord[mos,:,:] = temp_coord
        else: 
            dmin_coord = np.nan

    return dmin_coord


        

# # ------------------------Functions from https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/


# def draw_point(img, p, color) :
#     cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )
 

# # Draw delaunay triangles
# def draw_delaunay(img, subdiv, delaunay_color ) :
 
#     triangleList = subdiv.getTriangleList();
#     size = img.shape
#     r = (0, 0, size[1], size[0])
 
#     for t in triangleList :
 
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
 
#         if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
 
#             cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
#             cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
#             cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)
 
# # Draw voronoi diagram
# def draw_voronoi(img, subdiv) :
 
#     ( facets, centers) = subdiv.getVoronoiFacetList([])
 
#     for i in xrange(0,len(facets)) :
#         ifacet_arr = []
#         for f in facets[i] :
#             ifacet_arr.append(f)
 
#         ifacet = np.array(ifacet_arr, np.int)
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
#         cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0);
#         ifacets = np.array([ifacet])
#         cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
#         cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)
