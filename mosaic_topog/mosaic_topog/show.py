import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
from scipy.spatial import voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# --------------------------data viewing and saving functions--------------------------

## --------------------------------SMP VIEWING FUNCTIONS--------------------------------------


# def viewIntraconeDistHists(save_names, prefix, save_things=False, save_path=''):
#     print('beep')
#     for fl in save_names:
#         # get intracone distance histogram data and plotting parameters from the save file
#         with h5py.File(fl, 'r') as file:  # context manager 
#             coord = file['input_data']['cone_coord'][()]
#             hist = file[prefix + 'intracone_dist']['hist_mean'][()]
#             bin_edge = file[prefix + 'intracone_dist']['bin_edge'][()]
#             mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
#             conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
#             coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
#             conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
#             bin_width = file['input_data']['bin_width'][()]
#         num_cone = coord.shape[0]
#         id = mosaic + '_' + conetype
#         if not np.isnan(hist[0]):

#             # set up inputs to plot
#             xlab = 'distance, ' + coord_unit
#             ylab = 'bin count (binsize = ' + str(bin_edge[1]-bin_edge[0])
#             tit = 'intracone distance (' + str(num_cone) + " cones)"
#             x = bin_edge[1:]-(bin_width/2)

#             # view histogram
            
#             ax = line(x, hist, id, plot_col=conetype_color, title=tit, xlabel=xlab, ylabel=ylab)

#             ax.figure

#             if save_things:
#                 savnm = save_path + id + '.png'
#                 plt.savefig(savnm)
            
#         else:
#             print(id + ' contains < 2 cones, skipping... ')
    

# def viewMonteCarloStats(save_name, mc_type, scale_std=1, save_things=False, save_path=''):
#     for fl in save_name:
#         # get intracone distance histogram data and plotting parameters from the save file
#         with h5py.File(fl, 'r') as file:  # context manager
#             coord = file['input_data']['cone_coord'][()]
#             mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
#             conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
#             coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
#             conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
#             num_mc = file['input_data']['num_mc'][()]
#             bin_edge = file['monteCarlo_' + mc_type + '_intracone_dist']['bin_edge'][()]
#             mean_hist = file['monteCarlo_' + mc_type + '_intracone_dist']['hist_mean'][()]
#             std_hist = file['monteCarlo_' + mc_type + '_intracone_dist']['hist_std'][()]
#             bin_width = file['input_data']['bin_width'][()]
#         num_cone = coord.shape[0]
#         id_str = mosaic + '_' + conetype
#         if not np.isnan(mean_hist[0]):

#             # set up inputs to plot
#             xlab = 'distance, ' + coord_unit
#             ylab = 'bin count (binsize = ' + str(bin_edge[1]-bin_edge[0])
#             tit = 'MCU intracone distance (' + str(num_cone) + " cones, " + str(num_mc) + " MCUs)"
#             x = x = bin_edge[1:]-(bin_width/2)

#             ax = shadyStats(x, mean_hist, std_hist, id_str, scale_std=scale_std,
#                             plot_col=conetype_color, title=tit, xlabel=xlab,
#                             ylabel=ylab)

#             ax.figure

#             if save_things:
#                 savnm = save_path + id_str + '.png'
#                 plt.savefig(savnm)

            
        # saving images
        # .png if it doesn't need to be gorgeous and scaleable
        # .pdf otherwise, or eps, something vectorized 
        # numpy does parallelization under the hood

        # manually setting up parallel in python kinda sucks
        #   mpi is one approach


def view2PC(save_name, scale_std=2, showNearestCone=False, save_things=False, save_path=''):
    for fl in save_name:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            num_mc = file['input_data']['num_mc'][()]
            bin_width = file['input_data']['bin_width'][()]
            bin_edge = file['two_point_correlation']['bin_edge'][()]
            corred = file['two_point_correlation']['corred'][()]
            all_cone_mean_nearest = file['two_point_correlation']['all_cone_mean_nearest'][()]
            to_be_corr_colors = [bytes(n).decode('utf8') for n in file['input_data']['to_be_corr_colors'][()]]
            to_be_corr = [bytes(n).decode('utf8') for n in file['input_data']['to_be_corr'][()]]
            hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]

        # *** shouldn't need to get this this way, save it in meta data
        num_cone = coord.shape[0]
        id_str = mosaic + '_' + conetype

        fig, ax = plt.subplots(1, 1, figsize = [10,10])
        
        maxY = 0

        for ind, corr_set in enumerate(corred):
            if not np.isnan(corr_set[0].all()):
                if corr_set[0].shape[0] > 1:
                    hist_mean = corr_set[0]
                    hist_std = corr_set[1]
                    plot_label = to_be_corr[ind]
                    plot_col = to_be_corr_colors[ind]

                    maxY = np.nanmax([maxY,np.nanmax(hist_mean)])

                    # set up inputs to plot
                    xlab = 'distance, ' + coord_unit
                    ylab = 'bin count (binsize = ' + str(bin_width)

                    # *** SS fix this to pull the string from inputs
                    tit = 'intracone dists normed by MCU (' + str(num_cone) + " cones, " + str(num_mc) + " MCs)"
                    x = bin_edge[1:]-(bin_width/2)

                    half_cone_rad = all_cone_mean_nearest / 2
                    cone_rad_x = np.arange(half_cone_rad, half_cone_rad + (5 * all_cone_mean_nearest + 1), step=all_cone_mean_nearest)
                    lin_extent = 1.5

                    # if showNearestCone:
                    #     for lin in cone_rad_x:
                    #         if lin == cone_rad_x[0]:
                    #             ax = show.line([lin, lin], [-1 * lin_extent, lin_extent], id='cone-dist', ax=ax, plot_col='olive')
                    #         else:
                    #             ax = show.line([lin, lin], [-1 * lin_extent, lin_extent], id='cone-dist', ax=ax, plot_col='olive')

                    ax = line([hex_radius, hex_radius], [-1 * lin_extent, lin_extent], id='hex_radius', ax=ax, plot_col='maroon')
                    
                    ax = shadyStats(x, hist_mean, hist_std, id_str, ax = ax, scale_std=scale_std,
                                        plot_col = plot_col, label = plot_label)
        
            else:
                    print('no')
        
        print(maxY)
        if showNearestCone:
            plt.xlim([0, half_cone_rad + 5 * all_cone_mean_nearest + 1])
            plt.ylim([-1.5,maxY+2]) #plt.ylim([-0, lin_extent])

        ax.figure
        ax.legend()

        if save_things:
            savnm = save_path + id_str + '.png'
            plt.savefig(savnm)

            
def viewIntraconeDist(mos_type, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, scale_std=2,
            mosaic_data=True, marker='.', label=None, **kwargs):
    for fl in save_name:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            
            if mos_type == 'measured':
                coord = file['input_data']['cone_coord'][()]
                coord = np.expand_dims(coord, axis=0)
            else:
                coord = file[mos_type]['coord'][()]  

            bin_edge = file[mos_type + '_intracone_dist']['bin_edge'][()]
            mean_hist = file[mos_type + '_intracone_dist']['hist_mean'][()]
            std_hist = file[mos_type + '_intracone_dist']['hist_std'][()]
            bin_width = file['input_data']['bin_width'][()]

        num_mos = coord.shape[0]
        num_cone = coord.shape[1]
        id_str = mosaic + '_' + conetype
        if not np.isnan(mean_hist[0]):

            # set up inputs to plot
            xlab = 'distance, ' + coord_unit
            ylab = 'bin count (binsize = ' + str(bin_edge[1]-bin_edge[0])
            tit = mos_type + ' intracone distance (' + str(num_cone) + " cones, " + str(num_mos) + " mosaics)"
            x = x = bin_edge[1:]-(bin_width/2)

            ax = shadyStats(x, mean_hist, std_hist, id_str, scale_std=scale_std,
                            plot_col=conetype_color, title=tit, xlabel=xlab,
                            ylabel=ylab)

            ax.figure

            if save_things:
                savnm = save_path + id_str + '.png'
                plt.savefig(savnm)
                 

def viewVoronoiHistogram(mos_type, metric, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, 
            mosaic_data=True, marker='.', label=None, **kwargs):
    for fl in save_name:
        print(fl)
        # get spacified coordinate data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
           
            bound_regions = file[mos_type+'_voronoi']['bound_regions'][()][z_dim]
            metric_data = file[mos_type+'_voronoi'][metric][()][z_dim]
            metric_mean = file[mos_type+'_voronoi'][metric+'_mean'][()][z_dim]
            metric_std = file[mos_type+'_voronoi'][metric+'_std'][()][z_dim]
            metric_regularity = file[mos_type+'_voronoi'][metric+'_regularity'][()][z_dim]

        #print(metric_data[np.nonzero(bound_regions)])
        print('coord_unit: ' + coord_unit)
        print(metric + ' mean: ' + str(metric_mean))
        print(metric + ' std: ' + str(metric_std))
        print(metric + ' regularity: ' + str(metric_regularity))
        ax = getAx(kwargs)
        counts, bins = np.histogram(metric_data[np.nonzero(bound_regions)])
        ax.stairs(counts, bins)
        if metric_std > 1:
            plt.xlim([metric_mean - (2 * metric_std), metric_mean + (2 * metric_std)])
        else:
            plt.xlim([metric_mean - 5, metric_mean + 5])
        plt.xlabel(metric)
        plt.ylabel('count per bin')
        ax.figure

        if save_things:
            savnm = save_path + mosaic + '_' + str(z_dim) + '_' + conetype + '.png'
            plt.savefig(savnm)


def viewVoronoiDiagram(mos_type, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, 
            mosaic_data=True, marker='.', label=None, **kwargs):
    for fl in save_name:
        print(fl)
        # get spacified coordinate data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            if mos_type == 'measured':
                coord = file['input_data']['cone_coord'][()]
                coord = np.expand_dims(coord, axis=0)
            else:
                coord = file[mos_type]['coord'][()]
            
            regions = file[mos_type+'_voronoi']['regions'][()]
            vertices = file[mos_type+'_voronoi']['vertices'][()]
            num_neighbor = file[mos_type+'_voronoi']['num_neighbor'][()]
            bound_regions = file[mos_type+'_voronoi']['bound_regions'][()]
            voronoi_area = file[mos_type+'_voronoi']['voronoi_area'][()]
            convert_coord_unit = file['input_data']['convert_coord_unit'][()]
            
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            hex_radius = file[mos_type+'_voronoi']['hex_radius'][()]
            density = file[mos_type+'_voronoi']['density'][()]

            if convert_coord_unit:
                density_unit = bytes(file['input_data']['density_unit'][()]).decode("utf8")
            else:
                density_unit = coord_unit

    ax = getAx(kwargs)

    for i in range(0, len(regions[z_dim])):
        if bound_regions[z_dim][i]:
            if int(num_neighbor[z_dim][i]) == 3:
                colour = [255, 0, 0]
            elif int(num_neighbor[z_dim][i]) == 4:
                colour = [255, 100, 0]
            elif int(num_neighbor[z_dim][i]) == 5:
                colour = [255, 255, 0]
            elif int(num_neighbor[z_dim][i]) == 6:
                colour = [0, 255, 0]
            elif int(num_neighbor[z_dim][i]) == 7:
                colour = [0, 255, 255]
            elif int(num_neighbor[z_dim][i]) == 8:
                colour = [0, 0, 255]
            elif int(num_neighbor[z_dim][i]) == 9:
                colour = [100, 0, 255]
            elif int(num_neighbor[z_dim][i]) == 10:
                colour = [255, 0, 255]
            else:
                colour = [255, 255, 255]
            for ind, c in enumerate(colour): 
                colour[ind] = c / 255
            
            vert = regions[z_dim][i][np.nonzero(~np.isnan(regions[z_dim][i]))]
            polygon = vertices[z_dim][np.array(vert, dtype=int)]
            ax.fill(*zip(*polygon), facecolor = colour, edgecolor='k')

    ax = scatt(np.squeeze(coord[z_dim, :, :]),'bound_voronoi_cells', ax=ax, mosaic_data=True)
    ax.figure
    plt.xlabel(coord_unit)
    plt.ylabel(coord_unit)

    print('num bound cells: ' + str(sum(bound_regions[z_dim])))
    print('total voronoi area: ' + str(sum(voronoi_area[z_dim][np.nonzero(bound_regions[z_dim])])) + density_unit + '^2')
    print('voronoi density: ' + str(density) + ' points per ' + density_unit + '^2')
    print('hex radius calc from voronoi: ' + str(hex_radius) + ' ' + coord_unit)

    if save_things:
        savnm = save_path + mosaic + '_bound_cells_' + str(z_dim) + '_' + conetype + '.png'
        plt.savefig(savnm)


def viewMosaic(mos_type, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, 
            mosaic_data=True, marker='.', label=None, **kwargs):

    for fl in save_name:
        print(fl)
        # get spacified coordinate data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")

            if mos_type == 'measured':
                coord = file['input_data']['cone_coord'][()]
            else:
                try:
                    coord = file[mos_type]['coord'][()]
                except:
                    raise Exception('bad mosaic type sent to viewMosaic: ' + mos_type)

        if not np.isnan(coord[0]).any():
            print('num points: ' + str(coord.shape[0]))
            if len(coord.shape) == 3:
                num_mos = coord.shape[0]
                num_cone = coord.shape[1]
            elif len(coord.shape) == 2:
                num_mos = 1
                num_cone = coord.shape[0]
            
            for mos in np.arange(0,num_mos):
                id_str = mos_type + '_(' + str(mos+1) + '//' + str(num_mos) + ')_' + mosaic + '_(' + str(num_cone) + ' cones)'
                xlab = coord_unit
                ylab = coord_unit
                if len(coord.shape) == 3:
                    this_coord = np.zeros([num_cone, 2])
                    this_coord[:, :] = coord[mos, :, :]
                else: 
                    this_coord = coord

                ax = scatt(this_coord, id_str, plot_col=conetype_color, xlabel=xlab, ylabel=ylab, mosaic_data = mosaic_data, z_dim=z_dim, marker=marker, label=label)

                ax.figure

                if save_things:
                    savnm = save_path + mosaic + '_' + str(mos) + '_' + conetype + '.png'
                    plt.savefig(savnm)

        else:
            print('no coords for for "' + fl + '," skipping')



# ---------------------------------plotting functions----------------------------------


def getAx(kwargs):
    """
    Checks whether user input their own subplot, 
    generates one if they didn't

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    ax : AxesSubplot

    """
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    elif 'figsize' in kwargs.keys():
        fig, ax = plt.subplots(1, 1, figsize=[kwargs['figsize'],
                            kwargs['figsize']])
    else:
        fig, ax = plt.subplots(1, 1)
    return ax


def plotKwargs(kwargs, id):
    ax = getAx(kwargs)

    if 'title' in kwargs.keys():
        ax.set_title(id + ': ' + kwargs['title'])
    else:
        ax.set_title(id)

    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'])

    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'])

    return ax


def plotOnROI(img, coords, cone_types, id, colors, **kwargs):
    """
    # plot the ROI image, then outline all classed cones in yellow and fill in
    # with their respective cone type
    Parameters
    ----------
    img :
        ROI image loaded in by pyplot
    coords : dict of np.array
        cone coordinates to be plotted on the image
    ids : list of str
        conetypes to be plotted, which are also the keys to the coords
        dictionary, eg. ['all', 'L', 'M', 'S']
    colors : dict of str
        each str corresponds to a color, e.g ['y', 'r', 'g'] that will be used
        to plot the conetype of the same index in ids
    axes : AxesSubplot, optional

    Returns
    -------
    ax : AxesSubplot

    """
    ax = plotKwargs(kwargs, id)

    # overlay cone coordinates on the image
    ax.imshow(img)

    for ind, cone_type in enumerate(cone_types):

        # outline all cones in the 'all' mosaic
        if cone_type == 'all':
            csize = 30
            cface = 'none'
            cedge = colors[ind]

        else:  # plot all cones in solid circles based on their classification
            csize = 10
            cface = colors[ind]
            cedge = 'none'

        if coords[cone_type].size > 0:  # catch for an empty coordinate array

            if coords[cone_type].size > 2:
                xcoords = coords[cone_type][:, 0]
                ycoords = coords[cone_type][:, 1]

            else:  # catch for a coordinate array w only one element
                xcoords = coords[cone_type][0]
                ycoords = coords[cone_type][1]

            ax.scatter(x=xcoords, y=ycoords, s=csize, facecolors=cface,
                    edgecolors=cedge)

    return ax


def quad_fig(size):
    """
    initialize 2x2 figure.  input: size = [x,y] in inches
    """

    fig, ((ax, ax1),(ax2, ax3)) = plt.subplots(2, 2)
    axes = [ax,ax1,ax2,ax3]
    fig.set_size_inches(size[0], size[1])
    fig.tight_layout()

    return axes,fig


def scatt(coords, id, plot_col='w', bckg_col='k', z_dim=0, mosaic_data=True, marker='.', label=None, **kwargs):
    """
    2D scatter plot

    Parameters
    ----------
    coords : np.array
    unit : str
    id : str
    plot_col : str, default = 'w'
    bckg_col : str, default = 'k'
    z_dim : int, default = 0
    axes : AxesSubplot, optional

    Returns
    -------
    ax : AxesSubplot

    """
    # Handle any other
    ax = plotKwargs(kwargs, id)

    if len(coords.shape) == 1:
        scatter_x = coords[0]
        scatter_y = coords[1]

    elif len(coords.shape) == 2:  # 2D COORDINATE ARRAY
        scatter_x = coords[:, 0]
        scatter_y = coords[:, 1]

    else:  # 3D COORDINATE ARRAY
        scatter_x = coords[:, 0, z_dim]
        scatter_y = coords[:, 1, z_dim]

    ax.set_facecolor(bckg_col)
    ax.scatter(x=scatter_x, y=scatter_y,
            s=30,
            facecolors=plot_col,
            marker=marker,
            edgecolors='none')
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    if mosaic_data:
        ax.set_aspect('equal')

    return ax


def histo(hist_data, bin_edges, id, x_dim=1, plot_col='w',
        bckg_col='k', **kwargs):
    """

    Parameters
    ----------
    hist_data : np.array
    bin_edges : np.array
    id : str
    plot_col : str, default = 'w'
    bckg_col : str, default = 'k'
    z_dim : int, default = 0
    axes : AxesSubplot, optional
    figsize : int, optional
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional

    Returns
    -------
    ax : AxesSubplot

    """
    ax = plotKwargs(kwargs, id)

    # plot histogram of intercone distances for each mosaic
    if len(hist_data.shape) == 1:
        hist_data = hist_data
    else:
        hist_data = hist_data[:, x_dim]

    ax.set_facecolor(bckg_col)
    ax.hist(hist_data,
            bins=bin_edges,
            color=plot_col)

    return ax


def line(x, y, id, plot_col='w', bckg_col='k', linestyle='-', marker="", markersize=1, linewidth=1, **kwargs):
    """
    Plot a line

    Parameters
    ----------
    x : np.array
    y : np.array
    id : str
    plot_col : str, default = 'w'
    bckg_col : str, default = 'k'
    z_dim : int, default = 0
    axes : AxesSubplot, optional
    figsize : int, optional
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional

    Returns
    -------
    ax : AxesSubplot
    """
    ax = plotKwargs(kwargs, id)
    ax.set_facecolor(bckg_col)

    ax.plot(x, y, color=plot_col, linestyle=linestyle, linewidth = linewidth, marker=marker, markersize=markersize)

    return ax


def shadyStats(x, mean, std, id, scale_std=1, plot_col='w',
            bckg_col='k', label = '', **kwargs):
    """
    plot the mean of function with the std shaded around it

    Parameters
    ----------
    x : np.array
        1D x-values, same length as data and std
    mean : np.array
        1D mean of the data
    std : np.array
        1D std of the data
    scale_std : int, default = 1
    id : str
    axes : AxesSubplot
    figsize : int
    title_col : str
    bckg_col : str
    plot_col : str

    Returns
    -------
    ax : AxesSubplot

    """

    err_high = mean+(std*scale_std)
    err_low = mean-(std*scale_std)

    ax = plotKwargs(kwargs, id)
    ax.set_facecolor(bckg_col)
    ax.plot(x, mean, color=plot_col, linewidth = 6, label = label)
    ax.fill_between(x, err_low, err_high, color=plot_col, alpha=.7)

    return ax

# CV2

def drawPoint(img, p, colour):
    cv2.circle(img, p, 6, colour, cv2.FILLED, 8, 0)
