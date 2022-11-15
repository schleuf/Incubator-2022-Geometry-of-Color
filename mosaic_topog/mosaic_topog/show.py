import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import cv2

# --------------------------data viewing and saving functions--------------------------

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
            bound = file[mos_type+'_voronoi']['bound'][()]

            metric_data = file[mos_type+'_voronoi'][metric][()]
            metric_regularity = file[mos_type+'_voronoi'][metric+'_regularity'][()]

        ax = getAx(kwargs)
        ax.hist(metric_data[np.nonzero(bound)])
        ax.figure
        print(metric + ' regularity: ' + str(metric_regularity))

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
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            facets = file[mos_type+'_voronoi']['facets'][()]
            centers = file[mos_type+'_voronoi']['centers'][()]
            bound = file[mos_type+'_voronoi']['bound'][()]
            img_x = int(file['input_data']['img_x'][()])
            img_y = int(file['input_data']['img_y'][()])

    # convert facets to list of lists
    temp_f = []
    for f in np.arange(0, int(np.nanmax(facets[:,0]))+1): # first column indicates which cell this is a voronoi facet of
        rows = np.nonzero(facets[:,0]==f)
        temp_f.append(facets[rows,1:3])

    facets = temp_f

    img = np.zeros([img_x, img_y, 3])    

    for i in range(0, len(facets)):

        ifacet_arr = []
        for f in facets[i][0]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        if not bound[i]:
            colour = [150, 150, 150]
        elif len(ifacet_arr) == 3:
            colour = [255, 0, 0]
        elif len(ifacet_arr) == 4:
            colour = [255, 100, 0]
        elif len(ifacet_arr) == 5:
            colour = [255, 255, 0]
        elif len(ifacet_arr) == 6:
            colour = [0, 255, 0]
        elif len(ifacet_arr) == 7:
            colour = [0, 255, 255]
        elif len(ifacet_arr) == 8:
            colour = [0, 0, 255]
        elif len(ifacet_arr) == 9:
            colour = [100, 0, 255]
        elif len(ifacet_arr) == 10:
            colour = [255, 0, 255]
        else:
            colour = [255, 255, 255]
        for ind, c in enumerate(colour): 
            colour[ind] = c / 255

        cv2.fillConvexPoly(img, ifacet, colour, 8, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 255), 1)
        cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 1 + round(img.shape[1]/100), (0, 0, 0), cv2.FILLED)

    ax = getAx(kwargs)
    ax.imshow(img)
    ax.figure

    if save_things:
        savnm = save_path + mosaic + '_' + str(z_dim) + '_' + conetype + '.png'
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
