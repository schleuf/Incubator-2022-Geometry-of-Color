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
from shapely.geometry.polygon import Polygon
from scipy import spatial

import random


## --------------------------------SECONDARY ANALYSIS FUNCTIONS--------------------------------------

def metrics_of_2PC_process(param, sav_cfg):
    """ assumes that the data has been run on measured data and my 4 simulated populations"""
    proc = 'metrics_of_2PC'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']
    corr_by = param['corr_by']
    to_be_corr = param['to_be_corr']
    bin_width = param['bin_width']

    coord, PD_string = getDataForPrimaryProcess(sav_fl)

    if bin_width == -1:
        save_path = os.path.dirname(sav_fl)
        all_coord_fl = save_path + '\\' + param['mosaic'] + '_all.hdf5'

        try:
            with h5py.File(all_coord_fl, 'r') as file2:
                all_cone_mean_icd   = file2['measured_voronoi']['icd_mean'][()]
        except:
            print('could not pull mean nearest from ' + all_coord_fl)

        bin_width = all_cone_mean_icd

    with h5py.File(sav_fl, 'r') as file:
        corr_by_corr = file[corr_by + '_' + 'two_point_correlation']['corred'][()]
        max_bins = file[corr_by + '_' + 'two_point_correlation']['max_bins'][()]
        bin_edge = file[corr_by + '_' + 'two_point_correlation']['max_bin_edges'][()]
        sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
        if sim_hexgrid_by == 'rectangular':
            hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]
        elif sim_hexgrid_by == 'voronoi':
            hex_radius = file['measured_voronoi']['hex_radius'][()]
        else:
            print('ack!!! problem getting hex_radius in metrics_of_2PC_process')
        
    
    analysis_x_cutoff = int(np.argmin(np.abs((bin_edge - (2 * hex_radius)))))

    corr_by_corr = corr_by_corr[:, 0:analysis_x_cutoff]
    corr_by_mean = np.nanmean(corr_by_corr, axis=0)
    corr_by_std = np.nanstd(corr_by_corr, axis=0)

    for ind, PD in enumerate(PD_string):

        with h5py.File(sav_fl, 'r') as file:
            corred = file[PD + 'two_point_correlation']['corred'][()]

        crop_corr = corred[:, 0:analysis_x_cutoff]
        
        mean_corr = np.nanmean(crop_corr, axis=0)
        std_corr = np.nanstd(crop_corr, axis=0)  

        dearth_bins = []
        peak_bins = []
        first_peak_rad = np.empty([crop_corr.shape[0],])
        exclusion_bins = np.empty([crop_corr.shape[0],])
        exclusion_radius = np.empty([crop_corr.shape[0],])
        exclusion_area = np.empty([crop_corr.shape[0],])
        max_obed_exclusion_area = np.empty([crop_corr.shape[0],])
        exclusion_obed = np.empty([crop_corr.shape[0],])
        first_peak_rad[:] = np.nan
        exclusion_bins[:] = np.nan
        exclusion_radius[:] = np.nan
        exclusion_area[:] = np.nan
        max_obed_exclusion_area[:] = np.nan
        exclusion_obed[:] = np.nan
 
        for m in np.arange(0,crop_corr.shape[0]):
            dearth_bins.append(np.nonzero(crop_corr[m,:]  < corr_by_mean - (2 * corr_by_std))[0])

            peak_bins.append(np.nonzero(crop_corr[m,:] > corr_by_mean + (2 * corr_by_std))[0])

            first_peak_rad[m] = np.nan
            if len(peak_bins[m]) > 0:

                first_peak_rad[m] = bin_edge[peak_bins[m][0]+1]

            # ax = show.plotKwargs({'figsize':10}, '')

            # corr_by_x, corr_by_y, corr_by_y_plus, corr_by_y_minus = util.reformat_stat_hists_for_plot(bin_edge, corr_by_mean, corr_by_std*2)
            # ax = show.line(corr_by_x, corr_by_y, '', ax=ax, plot_col = 'firebrick')
            # ax.fill_between(corr_by_x, corr_by_y_plus, corr_by_y_minus, color='firebrick', alpha=.7)

            # hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bin_edge, crop_corr[m,:], np.zeros([crop_corr.shape[1],]))
            # ax = show.line(hist_x, hist_y, '', ax=ax, plot_col = 'w')
            # ax.fill_between(hist_x, hist_y_plus, hist_y_minus, color='royalblue', alpha=.7)

            # ax.scatter(bin_edge[dearth_bins[m]] + bin_width/2, crop_corr[m, dearth_bins[m]], color='g')
            # ax.scatter(bin_edge[peak_bins[m]] + bin_width/2, crop_corr[m, peak_bins[m]], color='y')


            if np.all((dearth_bins[m].shape[0] > 0) and (dearth_bins[m][0] == 0)):
                diff_dearth = np.diff(dearth_bins[m])

                diff1 = [d == 1 for d in diff_dearth.tolist()]
                
                zeros = np.nonzero([d == 0 for d in diff1])[0]

                if not np.any(zeros):
                    if len(diff1) == 0:
                        exclusion_bins[m] = 1
                    else:
                        exclusion_bins[m] = dearth_bins[m][np.nonzero(d==False for d in diff1)[0][0]] + 1
                else:
                    first_zero = zeros[0] + 1

                    exclusion_bins[m] = first_zero
            else:
                exclusion_bins[m] = 0

            exclusion_bins[m] = int(exclusion_bins[m])

            if exclusion_bins[m] > 0:
                exclusion_radius[m] = bin_edge[int(exclusion_bins[m])]-bin_edge[0]
            else: 
                exclusion_radius[m] = 0

            # ax = show.line([exclusion_radius, exclusion_radius], [-1, 1], '', plot_col = 'g', ax=ax)
            exclusion_area[m] = 0
            max_obed_exclusion_area[m] = 0
            # print('exclusionary radius)')
            # print(exclusion_radius[m])
            
            if (exclusion_radius[m] > 0):
                # print('exclusionary bins')
                # print(exclusion_bins[m])
                for b in np.arange(0, int(exclusion_bins[m])):

                    # print('            ' + str(exclusion_area[m]))
                    # print('                   ' + str((corr_by_mean[b] + (2 * corr_by_std))-crop_corr[m, b]))
                    # print(corr_by_mean[b] + (2 * corr_by_std[b]))
                    # print(crop_corr[m, b])
                    exclusion_area[m] = exclusion_area[m] + (bin_width * ((corr_by_mean[b] - (2 * corr_by_std[b]))-crop_corr[m, b]))
                    max_obed_exclusion_area[m] = max_obed_exclusion_area[m] + (bin_width * ((corr_by_mean[b] - (2 * corr_by_std[b]))-(-1)))
                    if max_obed_exclusion_area[m] > 0:
                        exclusion_obed[m] = exclusion_area[m] / max_obed_exclusion_area[m]
        

            #         ax.fill_between(bin_edge[b:b+2],
            #                         [crop_corr[m,b], crop_corr[m,b]],
            #                         [corr_by_mean[b] - (2 * corr_by_std[b]), crr_by_mean[b] - (2 * corr_by_std[b])], 
            #                         color='g', alpha=.5)
            # ax.set_title(PD + ' mosaic #' + str(m))
            # ax.set_xticks(bin_edge[0:analysis_x_cutoff])
            # ax.set_ylim([-1.5, 4])
        
        # print(PD)
        # print('radii')
        # print(exclusion_radius)
        # print('bins1')
        # print(exclusion_radius/bin_width)
        # print('bins2')
        # print(exclusion_bins)
        # print('EXCLUSION AREA')
        # print(exclusion_area)
        # print('')

        longest_dearth_list = np.amax([dearth_bins[x].shape[0] for x in np.arange(0,crop_corr.shape[0])])
        longest_peak_list= np.amax([x.shape[0] for x in peak_bins])

        temp_dearth = np.empty([crop_corr.shape[0], longest_dearth_list])
        temp_peak = np.empty([crop_corr.shape[0], longest_peak_list])
        temp_dearth[:] = np.nan
        temp_peak[:] = np.nan
        for m in np.arange(0,crop_corr.shape[0]):
            temp_dearth[m,0:dearth_bins[m].shape[0]] = dearth_bins[m]
            temp_peak[m,0:peak_bins[m].shape[0]] = peak_bins[m]

        dearth_bins = temp_dearth
        peak_bins = temp_dearth

        data_to_set = util.mapStringToLocal(proc_vars, locals())

        flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set, prefix=PD)


def two_point_correlation_process(param, sav_cfg):
    proc = 'two_point_correlation'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']
    bin_width = param['bin_width']
    to_be_corr = param['to_be_corr']
    sim_to_gen = param['sim_to_gen']
    corr_by = param['corr_by']

    #need this for the PD_string that will sort the 2PC results to each mosaic types save spot
    coord, PD_string = getDataForPrimaryProcess(sav_fl)

    dist_hists = []
    max_bins = 0
    with h5py.File(sav_fl, 'r') as file:
        
        for ind, PD in enumerate(PD_string):
            if PD == corr_by + '_':
                corr_by_ind = ind
                corr_by_mean = file[PD + 'intracone_dist']['hist_mean'][()]
                corr_by_std = file[PD + 'intracone_dist']['hist_std'][()]

            dist_hists.append(file[PD + 'intracone_dist']['hist_mat'][()])
            temp_edges = file[PD + 'intracone_dist']['bin_edge'][()]
            num_bins = temp_edges.shape[0]

            ax = show.plotKwargs({}, '')

            for m in np.arange(0, dist_hists[ind].shape[0]):
                plt.stairs(dist_hists[ind][m,:], temp_edges)

            if num_bins > max_bins:
                max_bins = num_bins
                max_bin_edges = temp_edges

    if corr_by_mean.shape[0] < max_bins:
        corr_by_mean = util.vector_zeroPad(corr_by_mean, 0, max_bins-corr_by_mean.shape[0]-1)
        corr_by_std = util.vector_zeroPad(corr_by_std, 0, max_bins-corr_by_std.shape[0]-1)

    if coord:
        hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(max_bin_edges, corr_by_mean, corr_by_std)

        ax = show.plotKwargs({}, '')
        ax = show.line(hist_x, hist_y, '', plot_col='w', ax=ax)
        ax.fill_between(hist_x, 
                        hist_y_plus, 
                        hist_y_minus, color='royalblue', alpha=.7)

        corr_by_hist = corr_by_mean

        # for each mosaic to be correlated against the msoaics
        for ind, dist_mat in enumerate(dist_hists):

            # preallocation for new 2PC matrix
            corred = np.empty([dist_mat.shape[0], max_bins-1])
            corred[:] = np.nan
            ax = show.plotKwargs({},'')

            for mos in np.arange(0, dist_mat.shape[0]):
                hist = dist_mat[mos,:]

                if hist.shape[0] < max_bins:
                    hist = util.vector_zeroPad(hist, 0, max_bins - hist.shape[0] - 1)

                # new 2PC calculation
                corred[mos,:] = calc.corr(hist, corr_by_hist)
                ax.stairs(corred[mos,:], max_bin_edges)

            corred = np.float64(corred)

            data_to_set = util.mapStringToLocal(proc_vars, locals())
            flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set, prefix=PD_string[ind])

    

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
        img_y = [0, file['input_data']['img_y'][()]]
        img_x = [0, file['input_data']['img_x'][()]]
        if convert_coord_unit:
            coord_conversion_factor = file['input_data']['coord_conversion_factor'][()]
            if type(coord_conversion_factor) == str:
                coord_conversion_factor = eval(coord_conversion_factor)
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
            # ax = show.plotKwargs({}, '')
            # ax = show.scatt(point_data[m, :, :], 'bound region check', ax=ax)
            bound_regions.append(np.zeros([len(regions[m]), ]))
            bound_reg = point_region[m][np.array(np.nonzero(bound_cones[m])[0], dtype=int)]
            bound_regions[m][bound_reg] = 1
            # for b in bound_reg:
            #     verts = regions[m][b]
            #     vert_copy = np.array(verts)
            #     if vert_copy.shape[0] > len(verts):
            #         print('an abundance of verts!')
            #     poly = vertices[m][verts]
            #     ax.fill(*zip(*poly), facecolor='r', edgecolor='k')

        neighbors_cones = calc.getVoronoiNeighbors(point_data, vertices, regions, ridge_vertices, ridge_points, point_region, bound_regions, bound_cones)
        icd = np.empty([point_data.shape[0], point_data.shape[1], neighbors_cones.shape[2]])
        icd_mean = np.empty([point_data.shape[0], ])
        icd_std = np.empty([point_data.shape[0], ])
        icd_regularity = np.empty([point_data.shape[0], ])
        icd[:] = np.nan
        icd_mean[:] = np.nan
        icd_std[:] = np.nan
        icd_regularity[:] = np.nan
        # it's possible for some cones in erratic mosaics to be bound but have no bound neighbors. get rid of those.
        for m in np.arange(point_data.shape[0]):
            dists = calc.dist_matrices(np.squeeze(point_data[m, :, :]))
            for nc in np.arange(0, neighbors_cones.shape[1]):
                not_nans = [~np.isnan(neighbors_cones[m, nc, x]) for x in np.arange(0, neighbors_cones.shape[2])]
                neighb_cones = np.array(neighbors_cones[m, nc, np.nonzero(not_nans)[0]], dtype=int)
                if ~np.any(np.nonzero(bound_cones[m][neighb_cones])[0]):
                    bound_cones[m][nc] = 0
                    r = point_region[m][nc]
                    bound_regions[m][r] = 0
                    neighbors_cones[m, nc, :] = np.nan
                else:
                    #get ICDs while we're here
                    icd[m, nc, np.array(np.nonzero(not_nans)[0], dtype=int)] = dists[nc, np.array(neighbors_cones[m, nc, np.nonzero(not_nans)[0]], dtype=int)]
            icd_mean[m] = np.nanmean(icd[m, :, :])
            icd_std[m] = np.nanstd(icd[m, :, :])
            icd_regularity[m] = icd_mean[m]/icd_std[m]
        # print('icd_mean')
        # print(icd_mean)
        # print('icd_std')
        # print(icd_std)
        # print('icd_regularity')
        # print(icd_regularity)
        # print('')
        # print('icd size')
        # print(icd.shape)
        # print(icd[0,:,:])
        [voronoi_area, voronoi_area_mean,
        voronoi_area_std, voronoi_area_regularity,
        num_neighbor, num_neighbor_mean,
        num_neighbor_std, num_neighbor_regularity]  = calc.voronoi_region_metrics(bound_regions, regions, vertices, point_region)

        # def convertRegion2Cone(metric, point_region):
        #     print('CONVERT REGION TO CONE')
        #     print(metric.shape)
        #     print(metric)
        #     print(point_region.shape)
        #     print(point_region)
        #     # temp = np.empty(np.squeeze(metric[m,:].shape))
        #     # temp[:] = np.nan
        #     # temp[:] = metric[m, point_region[m], :]


        # #convert voronoi region metrics to voronoi cone metrics
        # voronoi_area = convertRegion2Cone(voronoi_area, point_region)
        
        
        density = np.empty([point_data.shape[0],])
        hex_radius = np.empty([point_data.shape[0],])

        maxnum = int(np.nanmax([np.nanmax(num_neighbor[s]) for s in np.arange(0, len(num_neighbor))]))

        if not neighbors_cones.shape[2] == maxnum:
            print('THROW A HECKIN FIT NUM NEIGHBORS ARE COMING OUT DIFFERENT AAUUUGHGHGHH')

        temp_reg = np.empty([point_data.shape[0], 
                            int(np.nanmax([len(regions[m]) for m in np.arange(0, len(regions))])), 
                            maxnum])
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


def intracone_dist_common(coord, bin_width, dist_area_norm, offset_bin = False):
    """
    intracone_dist code shared for true and mc coordinate processes


    """
    # print('bin_width in intracone distance')
    # print(bin_width)
    # get intracone distances
    dist = calc.dist_matrices(coord, dist_self=np.nan)

    # get avg and std of nearest cone distance in the mosaic
    nearest_dist = []
    for cone in np.arange(0, dist.shape[0]):
        # get row for this cone's distance to every other cone
        row = dist[cone, :]
        # find the index where the distance = -1 if it exists - this is self
        # and shouldn't be included
        row = np.delete(row, np.nonzero(np.isnan(row))[0])
        # get the minimum value in the row
        nearest_dist.append(row.min())

    mean_nearest = np.mean(np.array(nearest_dist))
    std_nearest = np.std(np.array(nearest_dist))

    hist, bin_edge = calc.distHist(dist, bin_width, offset_bin)

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
            elif key == 'coneLocked_maxSpacing':
                take = 1
            elif key == 'hexgrid_by_density':
                take = 1
            
            if take:
                simulated.append(key)

        for sim in simulated:
            coord.append(file[sim]['coord'][()])

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

    if bin_width == -1:
         # look for all cone mosaic for this data
        save_path = os.path.dirname(sav_fl)
        all_coord_fl = save_path + '\\' + param['mosaic'] + '_all.hdf5'

        try:
            with h5py.File(all_coord_fl, 'r') as file:
                all_cone_mean_icd   = file['measured_voronoi']['icd_mean'][()]
        except:
            print('could not pull mean nearest from ' + all_coord_fl)
        bin_width = all_cone_mean_icd
        offset_bin = True
    else:
        offset_bin = False

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
                dist[mos, :, :], mean_nearest[mos], std_nearest[mos], hist[mos], bin_edge, annulus_area = intracone_dist_common(this_coord.squeeze(), bin_width, dist_area_norm, offset_bin)
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

            # print('hist std in the intracone dist process')
            # print(hist_std)

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
        id = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
        og_density = file['basic_stats']['rectangular_cone_density'][()]

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
        
        temp = []
        maxipad = 0
        for m in np.arange(0,num2gen):
            t, modded_hexradius, hexgrid_radius_decrease, hex_radius = calc.hexgrid(1,
                                                                hex_radius,
                                                                [0, img_x],
                                                                [0, img_y],
                                                                randomize=True,
                                                                min_cones=num_cones
                                                                )
            temp.append(t)
            maxipad = np.amax([maxipad, temp[m].shape[1]])

        coord = np.empty([num2gen,maxipad,2])
        coord[:] = np.nan

        for m in np.arange(0,num2gen):
            #non_nan = np.nonzero(np.isnan([temp[m][0,z,0] for z in np.arange(0,temp[m].shape[1])]))[0]
            coord[m, 0:temp[m].shape[1], :] = temp[m]
        # coord, modded_hexradius, hexgrid_radius_decrease, hex_radius = calc.hexgrid(num2gen,
        #                                                                 hex_radius,
        #                                                                 [0, img_x],
        #                                                                 [0, img_y],
        #                                                                 randomize=True,
        #                                                                 min_cones=num_cones
        #                                                                 )

        num_cones_placed = coord.shape[1]
        cone_density = num_cones_placed / (img_x * img_y)

        temp = np.empty([num2gen, num_cones, 2])
        for m in np.arange(0,num2gen):
            # print(['HEX COORD ' + str(m)])
            # print(coord[m, :, :])
            # print(np.nonzero([np.isnan(coord[m,a,0]) for a in np.arange(0,coord.shape[1])])[0].shape[0])
            non_nan = np.array(np.nonzero(~np.isnan(coord[m,:,0]))[0], dtype=int)
            # ax = show.plotKwargs({},'')
            # before = show.scatt(np.squeeze(coord[m, :, :]), id + ' ' + str(m) + ' before', ax=ax)
            temp[m,:,:] = util.trim_random_edge_points(coord[m, non_nan, :], num_cones, [0, img_x], [0, img_y])
            # ax1 = show.plotKwargs({},'')
            # after = show.scatt(np.squeeze(coord[m, :, :]), id + ' ' + str(m) + ' after', ax=ax1)

        coord = temp

        data_to_set = util.mapStringToLocal(proc_vars, locals())
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def coneLocked_maxSpacing_process(param, sav_cfg):
    proc = 'coneLocked_maxSpacing'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']   
    print(sav_fl) 
    with h5py.File(sav_fl, 'r') as file:
        og_coord = file['input_data']['cone_coord'][()]
        img_x = file['input_data']['img_x'][()]
        img_y = file['input_data']['img_y'][()]
        sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
        if sim_hexgrid_by == 'rectangular':
            hex_radius = file['input_data']['hex_radius_for_this_density'][()]
        elif sim_hexgrid_by == 'voronoi' :
            hex_radius = file['measured_voronoi']['hex_radius'][()]
    if og_coord.shape[0] > 1:
        num_sim = param['num_sim']
        sim_to_gen = param['sim_to_gen']
        x_dim = [0, img_x]
        y_dim = [0, img_y]
        # get all-cone-coordinates
        all_coord = flsyst.getAllConeCoord(sav_fl, param['mosaic'])
        # number of mosaics to generate
        num2gen = util.numSim('coneLocked_maxSpacing', num_sim, sim_to_gen)
        # cones per mosaic to generate
        cones2place = og_coord.shape[0]
        spaced_coord, modded_hex_radius, hex_radius_decrease, hex_radius = calc.coneLocked_hexgrid_mask(all_coord, num2gen, cones2place, x_dim, y_dim, hex_radius)
        
        # trim excess cones
        temp = np.empty([num2gen, cones2place, 2])
        ax = show.plotKwargs({},'')
        for mos in np.arange(0, num2gen):
            # print('num hexgrid pre trim')
            # print(spaced_coord[mos,:,:].shape)
            # print('num unique hexgrid')
            # print(np.unique(np.squeeze(spaced_coord[mos,:,:]), axis=0).shape)
            # non_nan = np.array(np.nonzero(~np.isnan(spaced_coord[mos,:,0]))[0], dtype=int)
            # ax0 = show.scatt(all_coord, 'trim test: before', plot_col = 'y')
            # ax0 = show.scatt(spaced_coord[mos,:,:], 'trim test: before', plot_col = 'r', ax=ax0)
            beep = np.unique(np.squeeze(spaced_coord[mos, non_nan, :]), axis=0)
            # print('num unique coords')
            # print(beep.shape)
            temp[mos, :, :] = util.trim_random_edge_points(spaced_coord[mos, non_nan, :], cones2place, x_dim, y_dim)
            # ax2 = show.scatt(all_coord, '', plot_col='y')
            # ax2 = show.scatt(temp[mos, :, :], 'trim test: after',plot_col='r', ax=ax2)
            # col = util.randCol()
            # #print(temp[mos,:,:].shape)
            # ax = show.scatt(temp[mos,:,:], 'conelocked spacified overlay', plot_col = col, ax=ax)
            
        coord = temp
        # print('size of spaced coordinates saved')
        # print(coord.shape)q
        # print('')
        data_to_set = util.mapStringToLocal(proc_vars, locals())
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


# def coneLocked_spacify_by_nearest_neighbors_process(param, sav_cfg):
#     """
#     """
#     # get any needed info from the save file
#     proc = 'coneLocked_spacify_by_nearest_neighbors'
#     proc_vars = sav_cfg[proc]['variables']
#     sav_fl = param['sav_fl']
#     with h5py.File(sav_fl, 'r') as file:
#         og_coord = file['input_data']['cone_coord'][()]
#         img = file['input_data']['cone_img'][()]
#         sim_to_gen = file['input_data']['sim_to_gen'][()]
#     num_sim = param['num_sim']

#     num2gen = util.numSim(proc, num_sim, sim_to_gen)

#     if len(og_coord.shape) == 2 and og_coord.shape[1] == 2:
#         num_coord = og_coord.shape[0]

#         all_coord = flsyst.getAllConeCoord(sav_fl, param['mosaic'])

#         if all_coord.shape[0] == og_coord.shape[0]:
#             print('skipped unnecessary expensive spacifying of all_coord data')
#             coord = np.tile(all_coord, (num2gen, 1, 1))
#         else:
#             coord = calc.spacifyByNearestNeighbors(num_coord, all_coord, num2gen)
#             for m in coord.shape[0]:
#                 ax = show.scatt(np.squeeze(coord[m,:,:],'coneLocked spacified'))
#         data_to_set = util.mapStringToLocal(proc_vars, locals())
        
#     flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def monteCarlo_process(param, sav_cfg, mc_type):
    # get any needed info from the save file
    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        all_coord = file['input_data']['cone_coord'][()]
        num2gen = util.numSim('monteCarlo_' + mc_type, param['num_sim'], param['sim_to_gen'])
        img_x = file['input_data']['img_x'][()]
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
    if len(tiers) >= 2:
        for proc in tiers[1]:
            if proc in user_param['analyses_to_run'][0]:
                print('Running process "' + proc + '" on ' + str(len(processes[proc])) + ' mosaic coordinate files...') 
                for ind in processes[proc]:
                    param = unpackThisParam(user_param, ind)
                    globals()[sav_cfg[proc]['process']](param, sav_cfg)


