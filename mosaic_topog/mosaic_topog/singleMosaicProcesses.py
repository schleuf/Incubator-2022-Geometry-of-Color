from cmath import nan
from tkinter.tix import MAX
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import cv2

import mosaic_topog.flsyst as flsyst
import mosaic_topog.calc as calc
import mosaic_topog.show as show
import mosaic_topog.utilities as util
from py import process

import random


## --------------------------------SECONDARY ANALYSIS FUNCTIONS--------------------------------------


def two_point_correlation_process(param, sav_cfg):
    proc = 'two_point_correlation'
    proc_vars = sav_cfg[proc]['variables']

    sav_fl = param['sav_fl']
    bin_width = param['bin_width']
    corr_by = param['corr_by']
    to_be_corr = param['to_be_corr']

    # pull the data that is requested for corr_by and to_be_corr

    maxbins = 0 

    with h5py.File(sav_fl, 'r') as file:
        corr_by_hist = file[corr_by + '_intracone_dist']['hist_mean'][()]
        bin_edge = file[corr_by + '_intracone_dist']['bin_edge'][()]
        all_coord = file['input_data']['cone_coord'][()]
        maxbins = np.amax([corr_by_hist.shape[0], maxbins])

        to_be_corr_hists = []
        for vect in to_be_corr:
            proc_to_get = vect + '_intracone_dist'
            try:
                all_coord = file[vect]['all_coord'][()]
            except:
                print('no all_coord key found in ' + vect)
            to_be_corr_hists.append([file[proc_to_get]['hist_mean'][()], file[proc_to_get]['hist_std'][()]])
            maxbins = np.amax([file[proc_to_get]['hist_mean'][()].shape[0], maxbins])

    corr_by_hist = util.vector_zeroPad(corr_by_hist, 0, maxbins - corr_by_hist.shape[0])
    for ind1, to_be_corr in enumerate(to_be_corr_hists):
        for ind2, vect in enumerate(to_be_corr):
            to_be_corr_hists[ind1][ind2] = util.vector_zeroPad(to_be_corr_hists[ind1][ind2], 0, maxbins-to_be_corr_hists[ind1][ind2].shape[0])

    if not np.isnan(corr_by_hist).any():
        while bin_edge.shape[0] <= maxbins:
            bin_edge = np.append(bin_edge, max(bin_edge)+bin_width)

        # Hey Sierra this section shouldn't need to be heeeere ***
        # get average nearest cone in the overall mosaic
        all_cone_dist = calc.dist_matrices(all_coord)
        print(all_cone_dist.size)

        # get avg and std of nearest cone distance in the mosaic
        nearest_dist = []
        for cone in np.arange(0, all_cone_dist.shape[0]):
            # get row for this cone's distance to every other cone
            row = all_cone_dist[cone, :]
            # find the index where the distance = -1 if it exists - this is self
            # and shouldn't be included
            row = np.delete(row, np.nonzero(row == -1))
            # get the minimum value in the row
            nearest_dist.append(row.min())

        all_cone_mean_nearest = np.mean(np.array(nearest_dist))
        all_cone_std_nearest = np.std(np.array(nearest_dist))

        corred = []
        for to_be_corr_set in to_be_corr_hists:
            corred_set = []
            for ind, vect in enumerate(to_be_corr_set):
                if not all(x == 0 for x in vect):
                    print('points in intrapoint distance hist: ' + str(np.nansum(vect)))
                    if ind == 0: #mean
                        corred_set.append(calc.corr(vect, corr_by_hist))
                    elif ind == 1: 
                        corred_set.append(vect/corr_by_hist)
                else: 
                    temp = np.empty((len(vect)))
                    temp[:] = np.NaN
                    corred_set.append(temp)
            corred.append(np.float64(corred_set))
        corred = np.float64(corred)
        
        data_to_set = util.mapStringToLocal(proc_vars, locals())
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


## --------------------------------PRIMARY ANALYSIS FUNCTIONS--------------------------------------


def voronoi_process(param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------


    """
    proc = 'voronoi'
    proc_vars = sav_cfg[proc]['variables']

    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        convert_coord_unit = file['input_data']['convert_coord_unit'][()]  
        coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
        img = file['input_data']['cone_img'][()]
        img_y = [0, img.shape[0]]
        img_x = [0, img.shape[1]]
        if convert_coord_unit:
            coord_conversion_factor = file['input_data']['coord_conversion_factor'][()]
            if type(coord_conversion_factor) == str:
                coord_conversion_factor = eval(coord_conversion_factor)
            img_x[1] = float(img_x[1]) * coord_conversion_factor
            img_y[1] = float(img_y[1]) * coord_conversion_factor
            density_unit = bytes(file['input_data']['density_unit'][()]).decode("utf8")
            density_conversion_factor = file['input_data']['density_conversion_factor'][()]
            if type(density_conversion_factor) == str:
                density_conversion_factor = eval(density_conversion_factor)
        else:
            density_unit = coord_unit

    coord, PD_string = getDataForPrimaryProcess(sav_fl)


    for ind, point_data in enumerate(coord):
        if len(point_data.shape) == 2:
            print('ack 2D data!!!')
        
        print('     Running voronois for ' + str(point_data.shape[0]) + " " + PD_string[ind] + ' mosaics...') 
        # ax = show.scatt(np.squeeze(point_data), 'points to be voronoid')

        [regions, vertices, ridge_vertices, ridge_points, point_region] = calc.voronoi(point_data)
        bound_cones = calc.get_bound_voronoi_cells(point_data, img_x, img_y)
       
        bound_regions = []
        for m in np.arange(point_data.shape[0]):
            ax = show.plotKwargs({}, '')
            ax = show.scatt(point_data[m, :, :], 'bound region check', ax=ax)
            bound_regions.append(np.zeros([len(regions[m]), ]))
            bound_reg = point_region[m][np.array(np.nonzero(bound_cones[m])[0], dtype=int)]
            bound_regions[m][bound_reg] = 1
            # for b in bound_reg:
            #     verts = regions[m][b]
            #     poly = vertices[m][verts]
            #     ax.fill(*zip(*poly), facecolor='r', edgecolor='k')
        
        neighbors_cones = calc.getVoronoiNeighbors(point_data, vertices, regions, ridge_vertices, ridge_points, point_region, bound_regions, bound_cones)

        [voronoi_area, voronoi_area_mean,
        voronoi_area_std, voronoi_area_regularity,
        num_neighbor, num_neighbor_mean, cone_neighbors,
        num_neighbor_std, num_neighbor_regularity]  = calc.voronoi_region_metrics(bound_regions, regions, vertices, point_region)

        density = np.empty([point_data.shape[0],])
        hex_radius = np.empty([point_data.shape[0],])

        maxnum = int(np.nanmax([np.nanmax(num_neighbor[s]) for s in np.arange(0, len(num_neighbor))]))
        temp_reg = np.empty([point_data.shape[0], len(regions[0]), maxnum])
        temp_reg[:] = np.nan
        temp_vert = np.empty([point_data.shape[0],
                    int(np.nanmax([len(vertices[m]) for m in np.arange(0, len(vertices))])),
                    2])
        
        for m in np.arange(0, point_data.shape[0]):
            numbound = np.sum(bound_regions[m])

            sumarea = np.sum(np.squeeze(voronoi_area[m, np.nonzero(bound_regions[m])]))
            density[m] = numbound/sumarea
            hex_radius[m] = calc.hex_radius(density[m])

            if convert_coord_unit:
                density = density * density_conversion_factor

            # convert list of lists to numpy arrays that can be saved in the h5py
            for r, reg in enumerate(regions[m]):
                if bound_regions[m][r]:
                    temp_reg[m, r, 0:len(reg)] = reg

            temp_vert[m, 0:len(vertices[m]), :] = np.array(vertices[m])
        regions = temp_reg
        vertices = temp_vert

        point_region = np.array(point_region)
        bound_regions = np.array(bound_regions)
        bound_cones = np.array(bound_cones)

        data_to_set = util.mapStringToLocal(proc_vars, locals())

        flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set,
                                      prefix=PD_string[ind])


def intracone_dist_common(coord, bin_width, dist_area_norm):
    """
    intracone_dist code shared for true and mc coordinate processes


    """

    # get intracone distances
    dist = calc.dist_matrices(coord)

    # get avg and std of nearest cone distance in the mosaic
    nearest_dist = []
    for cone in np.arange(0, dist.shape[0]):
        # get row for this cone's distance to every other cone
        row = dist[cone, :]
        # find the index where the distance = -1 if it exists - this is self
        # and shouldn't be included
        row = np.delete(row, np.nonzero(row == -1))
        # get the minimum value in the row
        nearest_dist.append(row.min())

    mean_nearest = np.mean(np.array(nearest_dist))
    std_nearest = np.std(np.array(nearest_dist))

    hist, bin_edge = calc.distHist(dist, bin_width)

    annulus_area = calc.annulusArea(bin_edge)

    if dist_area_norm:
        # normalize cone counts in each bin by the area of each annulus from
        # which cones were counted
        for ind, bin in enumerate(hist):
            hist[ind] = bin/annulus_area[ind]

    return dist, mean_nearest, std_nearest, hist, bin_edge, annulus_area


def getDataForPrimaryProcess(sav_fl):
   # get any needed info from the save file
    # this handling of the different types of point-data shouldn't be within the intracone
    with h5py.File(sav_fl, 'r') as file:
        all_coord = file['input_data']['cone_coord'][()]
        coord = []
        PD_string = []
        coord.append(np.reshape(all_coord, [1] + list(all_coord.shape)))

        simulated = []
        for key in file:
            take = 0
            if key == 'monteCarlo_uniform':
                take = 1
            elif key == 'monteCarlo_coneLocked':
                take = 1 
            elif key == 'coneLocked_spacify_by_nearest_neighbors':
                take = 1
            elif key == 'hexgrid_by_density':
                take = 1
            
            if take:
                simulated.append(key)

        for sim in simulated:
            coord.append(file[sim]['coord'][()])
            # print(file[sim]['coord'][()])

    for ind, point_data in enumerate(coord):

        # to store the outputs of this process
        data_to_set = {}

        if ind == 0:
            PD_string.append('measured' + '_')
        else:
            PD_string.append(simulated[ind-1] + '_')

    return coord, PD_string


def intracone_dist_process(param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------


    """
    proc = 'intracone_dist'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']
    bin_width = param['bin_width']
    dist_area_norm = param['dist_area_norm']
    sim_to_gen = param['sim_to_gen']

    coord, PD_string = getDataForPrimaryProcess(sav_fl)
    
    for ind, point_data in enumerate(coord):
        # if this is a valid coordinate dataset for this process...
        if len(point_data.shape) == 3:
            num_mosaic = point_data.shape[0]
            points_per_mos = point_data.shape[1]
            dist = np.zeros((num_mosaic, points_per_mos, points_per_mos))
            mean_nearest = np.zeros(num_mosaic)
            std_nearest = np.zeros(num_mosaic)
            hist = np.empty(num_mosaic, dtype=np.ndarray)
            max_hist_bin = 0
            print('     Running intracone distances on ' + str(num_mosaic) + " " + PD_string[ind] + ' mosaics...') 
            for mos in np.arange(0, num_mosaic):
                this_coord = point_data[mos, :, :]
                dist[mos, :, :], mean_nearest[mos], std_nearest[mos], hist[mos], bin_edge, annulus_area = intracone_dist_common(this_coord.squeeze(), bin_width, dist_area_norm)
                if hist[mos].shape[0] > max_hist_bin:
                    max_hist_bin = hist[mos].shape[0]

            # this isjust to convert the returned histograms into a rectangular array
            # (this can't be done in advance because of...slight variability in the number of bins returned? why?)
            hist_mat = np.zeros([num_mosaic, max_hist_bin])
            for mos in np.arange(0, num_mosaic):
                hist_mat[mos, 0:hist[mos].shape[0]] = hist[mos]

            hist = hist_mat

            while len(bin_edge) < max_hist_bin + 1:
                bin_edge = np.append(bin_edge, np.max(bin_edge)+bin_width)

            hist_mean = np.mean(hist_mat, axis=0)
            hist_std = np.std(hist_mat, axis=0)

            data_to_set = util.mapStringToLocal(proc_vars, locals())

        else:  # otherwise set these values to NaNs
            data_to_set = util.mapStringToNan(proc_vars)

        flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set, prefix=PD_string[ind])

        
## -------------------------- SIMULATION PROCESSES ---------------------------------

        
def hexgrid_by_density_process(param, sav_cfg):
    proc = 'hexgrid_by_density'
    proc_vars = sav_cfg['hexgrid_by_density']['variables']

    # First load in the inputs that are needed for this function
    sav_fl = param['sav_fl']
    num2gen = util.numSim(proc, param['num_sim'], param['sim_to_gen'])

    with h5py.File(sav_fl, 'r') as file:
        num_cones = file['basic_stats']['num_cones'][()]
        img_x = file['input_data']['img_x'][()]
        img_y = file['input_data']['img_y'][()]
        sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")

        # print('rectangular density')
        # print(file['basic_stats']['rectangular_cone_density'][()])
        # print('voronoi density')
        # print(file['measured_voronoi']['density'][()])
        # print('rectangular hex_radius')
        # print(file['basic_stats']['hex_radius_of_this_density'][()])
        # print('voronoi hex_radius')
        # print(file['measured_voronoi']['hex_radius'][()])

        if sim_hexgrid_by == 'rectangular':
            print(' creating hex radius by rectangluar density')
            hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]

        elif sim_hexgrid_by == 'voronoi':
            print('creating hex radius by voronoi density')
            hex_radius = file['measured_voronoi']['hex_radius'][()]
        else:
            print('bad input for sim_hexgrid_by, needs to be rectangular or voronoi')
            print(sim_hexgrid_by)

    if num_cones > 1:  # otherwise why bother
        coord = calc.hexgrid(num2gen,
                             hex_radius,
                             [0, img_x],
                             [0, img_y])
        
        ax1 = show.scatt(coord[0,:,:], '')
        ax2 = show.scatt(coord[1,:,:], '')

        num_cones_placed = coord.shape[1]
        cone_density = num_cones_placed / (img_x * img_y)

        #before = show.scatt(np.squeeze(coord[0, :, :]), 'before')
        coord = util.trim_random_edge_points(coord, num_cones, [0, img_x], [0, img_y])
        #after = show.scatt(np.squeeze(coord[0, :, :]), 'after')

        data_to_set = util.mapStringToLocal(proc_vars, locals())
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def coneLocked_spacify_by_nearest_neighbors_process(param, sav_cfg):
    """
    """
    # get any needed info from the save file
    proc = 'coneLocked_spacify_by_nearest_neighbors'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        og_coord = file['input_data']['cone_coord'][()]
        img = file['input_data']['cone_img'][()]
        sim_to_gen = file['input_data']['sim_to_gen'][()]
    num_sim = param['num_sim']

    num2gen = util.numSim(proc, num_sim, sim_to_gen)

    if len(og_coord.shape) == 2 and og_coord.shape[1] == 2:
        num_coord = og_coord.shape[0]

        all_coord = flsyst.getAllConeCoord(sav_fl, param['mosaic'])

        if all_coord.shape[0] == og_coord.shape[0]:
            print('skipped unnecessary expensive spacifying of all_coord data')
            coord = np.tile(all_coord, (num2gen, 1, 1))
        else:
            coord = calc.spacifyByNearestNeighbors(num_coord, all_coord, num2gen)
            for m in coord.shape[0]:
                ax = show.scatt(np.squeeze(coord[m,:,:],'coneLocked spacified'))
        data_to_set = util.mapStringToLocal(proc_vars, locals())
        
    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def monteCarlo_process(param, sav_cfg, mc_type):
    # get any needed info from the save file
    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        all_coord = file['input_data']['cone_coord'][()]
        num2gen = util.numSim('monteCarlo_' + mc_type, param['num_sim'], param['sim_to_gen'])
        img_x = file['input_data']['img_y'][()]
        img_y = file['input_data']['img_y'][()]

    proc = 'monteCarlo_' + mc_type
    proc_vars = sav_cfg[proc]['variables']

    # check for expected dimensions of the coordinate variable
    if len(all_coord.shape) == 2 and all_coord.shape[1] == 2:
        data_to_set = {}
        num_coord = all_coord.shape[0]
        if mc_type == 'uniform':
            xlim = [0, img_x]
            ylim = [0, img_y]
            coord = calc.monteCarlo_uniform(num_coord, num2gen, xlim, ylim)
            data_to_set = util.mapStringToLocal(proc_vars, locals())
        elif mc_type == 'coneLocked':
            # look for all cone mosaic for this data
            mosaic = param['mosaic']
            save_path = os.path.dirname(sav_fl)
            all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5'
            # try:
            with h5py.File(all_coord_fl, 'r') as file:
                all_coord = file['input_data']['cone_coord'][()]
            coord = calc.monteCarlo_coneLocked(num_coord, all_coord, num2gen)
            data_to_set = util.mapStringToLocal(proc_vars, locals())
            # except:
            #     print('could not find "' + all_coord_fl + ", cannot create coneLocked monteCarlo, skipping...")
            #     data_to_set = util.mapStringToNan(proc_vars)
        
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def monteCarlo_uniform_process(param, sav_cfg):
    """
    Inputs
    ------
    
    just a directory to the function for monteCarlo
    processes that takes uniform and coneLocked as
    options
    """
    monteCarlo_process(param, sav_cfg, 'uniform')


def monteCarlo_coneLocked_process(param, sav_cfg):
    """
    Inputs
    ------
    
    just a directory to the function for monteCarlo
    processes that takes uniform and coneLocked as
    options
    """
    monteCarlo_process(param, sav_cfg, 'coneLocked')



## --------------------------------DEFAULT PROCESS FUNCTIONS--------------------------------------

def input_data_process(param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------
    """
    proc = 'input_data'
    proc_vars = sav_cfg[proc]['variables']
    data_to_set = {}

    coord = np.loadtxt(param['coord_fl'], delimiter=',')
    img = plt.imread(param['img_fl'])

    param['img_x'] = img.shape[1]
    param['img_y'] = img.shape[0]

    if param['convert_coord_unit']:
        param['coord_unit'] = param['convert_coord_unit_to']
        if type(param['coord_conversion_factor']) == str:
            param['coord_conversion_factor'] = eval(param['coord_conversion_factor'])
        if type(param['density_conversion_factor']) == str:
            param['density_conversion_factor'] = eval(param['density_conversion_factor'])
        coord = coord * param['coord_conversion_factor']
        # ax = show.scatt(coord, 'converted', xlabel=param['convert_coord_unit_to'], ylabel=param['convert_coord_unit_to'])
        param['img_x'] = param['img_x'] * param['coord_conversion_factor']
        param['img_y'] = param['img_y'] * param['coord_conversion_factor']

    for var in proc_vars:
        if var == 'cone_img':
            data_to_set[var] = img
            
        elif var == 'cone_coord':
            data_to_set[var] = coord
        else:
            data_to_set[var] = param[var]

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def mosaic_meta_process(param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------

    """
    proc = 'mosaic_meta'
    proc_vars = sav_cfg[proc]['variables']
    data_to_set = {}
    for var in proc_vars:
        data_to_set[var] = param[var]

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def basic_stats_process(param, sav_cfg):
    proc = 'basic_stats'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        coord = file['input_data']['cone_coord'][()]
        img_x = file['input_data']['img_x'][()]
        img_y = file['input_data']['img_y'][()]
        num_cones = coord.shape[0]
        convert_coord_unit = bytes(file['input_data']['convert_coord_unit'][()]).decode("utf8")
        if convert_coord_unit:
            density_unit = bytes(file['input_data']['density_unit'][()]).decode("utf8")
            density_conversion_factor = file['input_data']['density_conversion_factor'][()]
        else:
            density_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
    img_area = img_x * img_y 
    rectangular_cone_density = num_cones/img_area

    hex_radius_of_this_density = calc.hex_radius(rectangular_cone_density)

    if convert_coord_unit:
        rectangular_cone_density = rectangular_cone_density * density_conversion_factor

    data_to_set = util.mapStringToLocal(proc_vars, locals())
    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)

## -------------------------------PROCESS HIERARCHY FUNCTIONS-------------------------------------
def unpackThisParam(user_param, ind):
    # this function is dumb
    # make this less dumb

    index = user_param['coord_index']
    param = {}

    # HEY SIERRA if you start getting errors here when you add more mosaics,
    # eccentricities, something - note the difference in the way "conetype" and "conetype_color" are set
    param['coord_fl'] = user_param['coord_fl_name'][ind]
    param['img_fl'] = user_param['img_fl_name'][index['mosaic'][ind]]
    param['sav_fl'] = user_param['save_name'][ind]
    param['mosaic'] = user_param['mosaic'][index['mosaic'][ind]]
    param['subject'] = user_param['subject'][0][index['subject'][ind]]
    param['angle'] = user_param['angle'][0][index['angle'][ind]]
    param['eccentricity'] = user_param['eccentricity'][0][index['eccentricity'][ind]]
    param['conetype'] = user_param['conetype'][0][index['conetype'][ind]]
    param['conetype_color'] = user_param['conetype_color'][0][index['conetype'][ind]]

    param['coord_unit'] = user_param['coord_unit'][0]
    param['bin_width'] = user_param['bin_width'][0]
    param['dist_area_norm'] = user_param['dist_area_norm'][0]
    param['data_path'] = user_param['data_path'][0]
    param['save_path'] = user_param['save_path'][0]
    param['sim_to_gen'] = user_param['sim_to_gen'][0]
    param['num_sim'] = user_param['num_sim'][0]
    param['analyses_to_run'] = user_param['analyses_to_run'][0]
    param['corr_by'] = user_param['corr_by']
    param['to_be_corr'] = user_param['to_be_corr'][0]
    param['to_be_corr_colors'] = user_param['to_be_corr_colors'][0]

    param['convert_coord_unit'] = user_param['convert_coord_unit']
    param['convert_coord_unit_to'] = user_param['convert_coord_unit_to']
    param['coord_conversion_factor'] = user_param['coord_conversion_factor']
    param['density_unit'] = user_param['density_unit']
    param['density_conversion_factor'] = user_param['density_conversion_factor']
    param['sim_hexgrid_by'] = user_param['sim_hexgrid_by']
    
    return param


# this can be streamlined and made not single/multi specific if I add
# parameters to the yaml that define order of operations
def runSingleMosaicProcess(user_param, sav_cfg):
    """

    """

    # this needs to updated, needs to error if the layer doesn't exist
    for layer in sav_cfg['process_hierarchy']['content']:
        globals()[sav_cfg[layer]['process']](user_param, sav_cfg)
def default_processes_process(user_param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------
    """
    # identify mandatory processes
    # should consider making this a list in the yaml rather than 
    # a property of the individual files

    mand = []
    for key in sav_cfg.keys():
        if sav_cfg[key]['process_type'] == 'default':
            mand.append(key)
    print('mand: ')
    print(mand)

    # get all files to check
    fls = []
    processes = user_param['processes']
    for proc in processes:
        fls = np.union1d(fls, processes[proc])

    fls = fls.astype(int)

    for ind in fls:
        param = unpackThisParam(user_param, ind)
        if not os.path.exists(param['sav_fl']):
            with h5py.File(param['sav_fl'], 'w') as file:
                print('created file: ' + param['sav_fl'])
        for proc in mand:
            globals()[sav_cfg[proc]['process']](param, sav_cfg)


def gen_sim_process(user_param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------
    """
    sim_to_gen = user_param['sim_to_gen'][0]
    num_sim = user_param['num_sim'][0]
    processes = user_param['processes']
    for sim in sim_to_gen:
        num2gen = util.numSim(sim, num_sim, sim_to_gen)
        print('Generating ' + str(num2gen) + ' "' + sim + '"" simulations for ' + str(len(processes[sim])) + ' mosaic coordinate files...') 
        for ind in processes[sim]:
            param = unpackThisParam(user_param, ind)
            globals()[sav_cfg[sim]['process']](param, sav_cfg)


def getAnalysisTiers(sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------
    """
    analysis_proc = []
    analysis_tiers = []
    for key in sav_cfg.keys():
        if 'process_type' in sav_cfg[key]:
            if sav_cfg[key]['process_type'] == 'analysis':
                analysis_proc.append(key)
                analysis_tiers.append(sav_cfg[key]['analysis_tier'])
    max_tier = np.amax(analysis_tiers)
    analyses_by_tier = []
    for tier in np.arange(0, max_tier):
        analyses_by_tier.append(np.array(analysis_proc)[np.nonzero(analysis_tiers == tier+1)[0]])
    return analyses_by_tier


def primary_analyses_process(user_param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------

    """
    processes = user_param['processes']
    
    tiers = getAnalysisTiers(sav_cfg)
    
    # perform on data
    for proc in tiers[0]:
        if proc in user_param['analyses_to_run'][0]:
            for ind in processes[proc]:
                print('Running process "' + proc + '" on file' + str(ind+1) + '/' + str(len(processes[proc])) +'...') 
                param = unpackThisParam(user_param, ind)
                print('     ' + param['sav_fl'])
                globals()[sav_cfg[proc]['process']](param, sav_cfg)
                #print("     SIMULATED COORDINATES")
                # for sim in user_param['sim_to_gen'][0]:
                #     if sim == 'monteCarlo_uniform' or sim == 'monteCarlo_coneLocked':
                #         numsim = user_param['num_mc']
                #     elif sim == 'coneLocked_spacify_by_nearest_neighbors' or sim == 'hexgrid_by_density':
                #         numsim = user_param['num_sp']
                #     else:
                #         print('Error: invalid simulation entry')
                    # print('     Running process "' + proc + '" on' + str(numsim) + ' ' + sim + 'simulations...') 
                    # globals()[sav_cfg[sim]['process']](param, sav_cfg)
        else:
            print('didnt run ' + proc)


def secondary_analyses_process(user_param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------
    """
    processes = user_param['processes']
    tiers = getAnalysisTiers(sav_cfg)

    # perform on data
    for proc in tiers[1]:
        if proc in user_param['analyses_to_run'][0]:
            print('Running process "' + proc + '" on ' + str(len(processes[proc])) + ' mosaic coordinate files...') 
            for ind in processes[proc]:
                param = unpackThisParam(user_param, ind)
                globals()[sav_cfg[proc]['process']](param, sav_cfg)


