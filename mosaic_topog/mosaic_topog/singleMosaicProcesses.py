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

import random


## --------------------------------SECONDARY ANALYSIS FUNCTIONS--------------------------------------

def metrics_of_2PC_process(param, sav_cfg):
    """ assumes that the data has been run on measured data and my 4 simulated popualtions"""
    proc = 'metrics_of_2PC'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']
    corr_by = param['corr_by']
    to_be_corr = param['to_be_corr']
    bin_width = param['bin_width']
    with h5py.File(sav_fl, 'r') as file:
        all_cone_mean_nearest = file['two_point_correlation']['all_cone_mean_nearest'][()]
        all_cone_std_nearest = file['two_point_correlation']['all_cone_std_nearest'][()]
        corred = file['two_point_correlation']['corred'][()]
        maxbins = file['two_point_correlation']['maxbins'][()]
        bin_edge = file['two_point_correlation']['bin_edge'][()]
        sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
        if sim_hexgrid_by == 'rectangular':
            hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]
        elif sim_hexgrid_by == 'voronoi':
            hex_radius = file['measured_voronoi']['hex_radius'][()]
        else:
            print('ack!!! problem getting hex_radius in metrics_of_2PC_process')

    analysis_x_cutoff = int(np.ceil((2 * hex_radius) / bin_width))   
    crop_corr = corred[:,:,0:analysis_x_cutoff]
    print('crop_corr')
    print(crop_corr)

    x = bin_edge[1:analysis_x_cutoff+1]-(bin_width/2)

    #M = measured
    M_ind = int(np.nonzero([d =='measured' for d in to_be_corr])[0])
    #Sr = spaced, restricted
    Sr_ind = int(np.nonzero([d =='coneLocked_maxSpacing' for d in to_be_corr])[0])
    #Rr = randomized, restricted
    Rr_ind = int(np.nonzero([d =='monteCarlo_coneLocked' for d in to_be_corr])[0])
    #Ru = randomized, unrestricted
    Ru_ind = int(np.nonzero([d =='monteCarlo_uniform' for d in to_be_corr])[0])
    
    #---------------------------------PEAKS----------------------------------------------

    x_peaks = np.array([np.nanargmax(crop_corr[d, 0, :]) for d in np.arange(0,crop_corr.shape[0])])
    ax = show.view2PC([sav_fl], scale_std=2, showNearestCone=False)
    plt.xlim([0, 2 * hex_radius])
    print('amax')
    print(crop_corr[Sr_ind, 0, :])
    print(np.nanmax(crop_corr[Sr_ind, 0, :]))
    plt.ylim([-1.5, np.nanmax(crop_corr[Sr_ind, 0, :])+1])

    # coneLocked maximally spaced peak
    Sr_x_peak_ind = x_peaks[Sr_ind]
    Sr_y_peak = crop_corr[Sr_ind, 0, Sr_x_peak_ind]
    Sr_std_peak = crop_corr[Sr_ind, 1, Sr_x_peak_ind]
    Rr_y_at_Sr_peak = crop_corr[Rr_ind, 0, Sr_x_peak_ind]
    Rr_std_at_Sr_peak = crop_corr[Rr_ind, 1, Sr_x_peak_ind]
    if Sr_y_peak - Sr_std_peak > Rr_y_at_Sr_peak + Rr_std_at_Sr_peak:
        Sr_x_peak = bin_edge[0] + Sr_x_peak_ind * bin_width + (bin_width/2)
        Sr_peak = np.array([Sr_x_peak, Sr_y_peak])
    else:
        Sr_peak = np.array([np.nan, np.nan])
    diff_Sr_peak_from_hex_radius = Sr_x_peak - hex_radius

    ax = show.scatt(Sr_peak, '', ax=ax, plot_col ='darkorange', s=600, marker='x', mosaic_data=False)
    
    # measured peak
    M_x_peak_ind = x_peaks[M_ind]
    M_y_peak = crop_corr[M_ind, 0, M_x_peak_ind]
    M_std_peak = crop_corr[M_ind, 1, M_x_peak_ind]
    Rr_y_at_M_peak = crop_corr[Rr_ind, 0, M_x_peak_ind]
    Rr_std_at_M_peak = crop_corr[Rr_ind, 1, M_x_peak_ind]
    if M_y_peak - M_std_peak > Rr_y_at_M_peak + Rr_std_at_M_peak:
        M_x_peak = bin_edge[0] + M_x_peak_ind * bin_width + (bin_width/2)
        M_peak = np.array([M_x_peak, M_y_peak])
    else: 
        M_peak = np.array([np.nan, np.nan])

    ax = show.scatt(M_peak, '', ax=ax, plot_col = 'white', s=600, marker='x', mosaic_data=False)
    
    #  random peak
    Rr_x_peak_ind = x_peaks[Rr_ind]
    Rr_y_peak = crop_corr[Rr_ind, 0, Rr_x_peak_ind]
    Rr_std_peak = crop_corr[Rr_ind, 1, Rr_x_peak_ind]
    Ru_y_at_Rr_peak = crop_corr[Ru_ind, 0, Rr_x_peak_ind]
    Ru_std_at_Rr_peak = crop_corr[Ru_ind, 1, Rr_x_peak_ind]
    if Rr_y_peak - Rr_std_peak > Ru_y_at_Rr_peak + Ru_std_at_Rr_peak:
        Rr_x_peak = bin_edge[0] + (Rr_x_peak_ind * bin_width) + (bin_width/2)
        Rr_peak = np.array([Rr_x_peak, Rr_y_peak])
    else: 
        Rr_peak = np.array([np.nan, np.nan])

    ax = show.scatt(Rr_peak, '', ax=ax, plot_col = 'royalblue', s=600, marker='x', mosaic_data=False)
    
    #-----------------------------MEASURED EXCLUSION RADIUS & AREA----------------------------------------------
    ax2 = show.view2PC([sav_fl], scale_std=2, showNearestCone=False)
    plt.xlim([0, 2 * hex_radius])
    plt.ylim([-1.5, np.nanmax(crop_corr[Sr_ind, 0, :])+1])

    # Rr_min_dist = distance where mean Rr deviates first deviates from -1
    lower_Rr = crop_corr[Rr_ind, 0, :] - (2 * crop_corr[Rr_ind, 1, :])
    Rr_above_neg1_ind = np.nonzero([p > -1 for p in lower_Rr])[0][0]
    Rr_above_neg1 = bin_edge[0] + (Rr_above_neg1_ind * bin_width) + (bin_width/2)
    ax2 = show.scatt(np.array([Rr_above_neg1, 0]), '', plot_col = 'r', ax=ax2, s=600, marker='x', mosaic_data=False)
    if Rr_above_neg1_ind > 0:
        line_inds = [Rr_above_neg1_ind - 1, Rr_above_neg1_ind]
        Rr_cross_neg1 = np.array(calc.line_intersection(x[line_inds], lower_Rr[line_inds], x[line_inds], [-1, -1]))
        if np.any(np.isnan(Rr_cross_neg1)):
            Rr_cross_neg1 = np.array([0,lower_Rr[0]])
    else:
        Rr_cross_neg1 = np.array([0,lower_Rr[0]])
    ax2 = show.scatt(np.array(Rr_cross_neg1), '', plot_col = 'y', ax=ax2, s=600, marker='x', mosaic_data=False)

    # M_meets_Rr_dist
    upper_M = crop_corr[M_ind, 0, :]
    M_meets_Rr_ind = (Rr_above_neg1_ind + 
                       np.nonzero([upper_M[x] >= lower_Rr[x] 
                                   for x in np.arange(Rr_above_neg1_ind,
                                                      upper_M.shape[0])])[0][0])
    M_meets_Rr = bin_edge[0] + (M_meets_Rr_ind * bin_width) + (bin_width/2)
    ax2 = show.scatt(np.array([M_meets_Rr, 0]), '', plot_col = 'r', ax=ax2, s=600, marker='x', mosaic_data=False)
    if M_meets_Rr > 0:
        line_inds = [M_meets_Rr_ind - 1, M_meets_Rr_ind]
        M_cross_Rr = np.array(calc.line_intersection(x[line_inds], upper_M[line_inds], x[line_inds], lower_Rr[line_inds]))
    else:
        M_cross_Rr = np.array([x[0],upper_M[0]])
    ax2 = show.scatt(np.array(M_cross_Rr), '', plot_col = 'y', ax=ax2, s=600, marker='x', mosaic_data=False)

    # M_exclusion_radius
    M_exclusion_radius = M_cross_Rr[0] - Rr_cross_neg1[0]
    ax2 = show.line([Rr_cross_neg1[0], M_cross_Rr[0]], [1, 1], '', ax=ax2, plot_col='b', linewidth=3)
    
    # M_exclusion_area
    len_radius = M_meets_Rr_ind - Rr_above_neg1_ind + 1
    points = np.empty([len_radius * 2, 2])
    points[:] = np.nan
    points[0, :] = Rr_cross_neg1
    print(Rr_cross_neg1)
    points[1:len_radius, 0] = x[Rr_above_neg1_ind:M_meets_Rr_ind]
    points[1:len_radius, 1] = lower_Rr[Rr_above_neg1_ind:M_meets_Rr_ind]
    points[len_radius, :] = M_cross_Rr
    points[len_radius+1:points.shape[0], 0] = np.flip(x[Rr_above_neg1_ind:M_meets_Rr_ind])
    points[len_radius+1:points.shape[0], 1] = np.flip(upper_M[Rr_above_neg1_ind:M_meets_Rr_ind])
    if points.shape[0] <= 2:
        M_exclusion_area = np.nan
    else:
        print('points')
        print(points)
        poly = Polygon(points)
        M_exclusion_area = poly.area
        ax2.fill(*zip(*points), facecolor = 'b', edgecolor='b')
    print('M_exclusion_area')
    print(M_exclusion_area)

    #-----------------------------SPACED RESTRICTED EXCLUSION RADIUS & AREA----------------------------------------------
    ax3 = show.view2PC([sav_fl], scale_std=2, showNearestCone=False)
    plt.xlim([0, 2 * hex_radius])
    plt.ylim([-1.5, np.amax(crop_corr[Sr_ind, :])+1])

    # Sr_meets_Rr_ind
    upper_Sr = crop_corr[Sr_ind, 0, :] + (2 * crop_corr[Sr_ind, 1, :])
    lower_Rr = crop_corr[Rr_ind, 0, :] - (2 * crop_corr[Rr_ind, 1, :])
    Sr_meets_Rr_ind = (Rr_above_neg1_ind + 
                       np.nonzero([upper_Sr[x] >= lower_Rr[x] 
                                   for x in np.arange(Rr_above_neg1_ind,
                                                      upper_Sr.shape[0])])[0][0])
    Sr_meets_Rr = bin_edge[0] + (Sr_meets_Rr_ind * bin_width) + (bin_width/2)
    ax3 = show.scatt(np.array([Sr_meets_Rr, 0]), '', plot_col = 'r', ax=ax3, s=600, marker='x', mosaic_data=False)
    if Sr_meets_Rr > 0:
        line_inds = [Sr_meets_Rr_ind - 1, Sr_meets_Rr_ind]
        Sr_cross_Rr = np.array(calc.line_intersection(x[line_inds], upper_Sr[line_inds], x[line_inds], lower_Rr[line_inds]))
    else:
        Sr_cross_Rr = np.array([x[0],upper_Sr[0]])
    ax3 = show.scatt(np.array(Sr_cross_Rr), '', plot_col = 'y', ax=ax3, s=600, marker='x', mosaic_data=False)

    # Sr_exclusion_radius
    Sr_exclusion_radius = Sr_cross_Rr[0] - Rr_cross_neg1[0]
    ax3 = show.line([Rr_cross_neg1[0], Sr_cross_Rr[0]], [.75, .75], '', ax=ax3, plot_col = 'b', linewidth=3)
    print('Sr_exclusion_radius')
    print(Sr_exclusion_radius)

    # # Sr_exclusion_area
    len_radius = Sr_meets_Rr_ind - Rr_above_neg1_ind + 1
    points = np.empty([len_radius * 2, 2])
    points[:] = np.nan
    points[0, :] = Rr_cross_neg1
    points[1:len_radius, 0] = x[Rr_above_neg1_ind:Sr_meets_Rr_ind]
    points[1:len_radius, 1] = lower_Rr[Rr_above_neg1_ind:Sr_meets_Rr_ind]
    points[len_radius, :] = Sr_cross_Rr
    points[len_radius+1:points.shape[0], 0] = np.flip(x[Rr_above_neg1_ind:Sr_meets_Rr_ind])
    points[len_radius+1:points.shape[0], 1] = np.flip(upper_Sr[Rr_above_neg1_ind:Sr_meets_Rr_ind])
    if points.shape[0] <= 2:
        Sr_exclusion_area = np.nan
    else:
        poly = Polygon(points)
        Sr_exclusion_area = poly.area
        ax3.fill(*zip(*points), facecolor = 'b', edgecolor='b')
    print('Sr_exclusion_area')
    print(Sr_exclusion_area)

    #-----------------------------EXCLUSIONARY OBEDIENCE---------------------------------------------
    ax4 = show.view2PC([sav_fl], scale_std=2, showNearestCone=False)
    plt.xlim([0, 2 * hex_radius])
    plt.ylim([-1.5, np.amax(crop_corr[Sr_ind, :])+1])

    print('xy of lower Rr where M would cross if fully obedient to the max')
    print(x[line_inds])
    print(lower_Rr[line_inds])
    if M_meets_Rr > 0:
        line_inds = [M_meets_Rr_ind-1, M_meets_Rr_ind]

        M_maxObed_cross_Rr = np.array(calc.line_intersection(x[line_inds], [-1, upper_M[M_meets_Rr_ind]], x[line_inds], lower_Rr[line_inds]))
    else:
        M_maxObed_cross_Rr = np.array([x[0], upper_M[0]])
    print('M_maxObed_cross_Rr')
    print(M_maxObed_cross_Rr)
    # ax4 = show.line(x[line_inds], [-1, upper_M[M_meets_Rr_ind]],'',plot_col='b',ax=ax4, linewidth=3)
    # ax4 = show.line(x[line_inds], lower_Rr[line_inds],'',plot_col='b',ax=ax4, linewidth=3)
    ax4 = show.scatt(np.array(M_maxObed_cross_Rr), '', plot_col = 'y', ax=ax4, s=400, marker='x', mosaic_data=False)
    ax4 = show.scatt(np.array(M_cross_Rr), '', plot_col = 'r', ax=ax4, s=400, marker='x', mosaic_data=False)

    len_radius = M_meets_Rr_ind - Rr_above_neg1_ind + 1
    points = np.empty([len_radius * 2, 2])
    points[:] = np.nan
    points[0, :] = Rr_cross_neg1
    points[1:len_radius, 0] = x[Rr_above_neg1_ind:M_meets_Rr_ind]
    points[1:len_radius, 1] = lower_Rr[Rr_above_neg1_ind:M_meets_Rr_ind]
    points[len_radius, :] = M_maxObed_cross_Rr
    points[len_radius+1:points.shape[0], 0] = np.flip(x[Rr_above_neg1_ind:M_meets_Rr_ind])
    points[len_radius+1:points.shape[0], 1] = -1
    if points.shape[0] <= 2:
        M_maxObed_exclusion_area = np.nan
    else:
        poly = Polygon(points)
        M_maxObed_exclusion_area = poly.area
        ax4.fill(*zip(*points), facecolor = 'r', edgecolor='r')
    print('M_maxObed_exclusion_area')
    print(M_maxObed_exclusion_area)

    exclusion_obedience = M_exclusion_area / M_maxObed_exclusion_area
    print('exclusion_obedience')
    print(exclusion_obedience)

    #-----------------------------asdl;kfj---------------------------------------------
    relative_structure_to_max = np.empty([2,])
    relative_structure_to_max[0] = M_exclusion_radius/Sr_exclusion_radius
    relative_structure_to_max[1] = M_exclusion_area/Sr_exclusion_area
    print('relative_structure_to_max')
    print(relative_structure_to_max)
    print('')

    data_to_set = util.mapStringToLocal(proc_vars, locals())

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def two_point_correlation_process(param, sav_cfg):
    proc = 'two_point_correlation'
    proc_vars = sav_cfg[proc]['variables']

    sav_fl = param['sav_fl']
    bin_width = param['bin_width']
    # print("bin_width")
    # print(bin_width)
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
        x = bin_edge[1:]-(bin_width/2)
        # Hey Sierra this section shouldn't need to be heeeere ***
        # get average nearest cone in the overall mosaic
        all_cone_dist = calc.dist_matrices(all_coord)

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
        ax = show.plotKwargs({},'')
        for to_be_corr_set in to_be_corr_hists:
            corred_set = []
            for ind, vect in enumerate(to_be_corr_set):
                if not all(x == 0 for x in vect):
                    # print('points in intrapoint distance hist: ' + str(np.nansum(vect)))
                    if ind == 0: #mean
                        corred_set.append(calc.corr(vect, corr_by_hist))
                    elif ind == 1: 
                        corred_set.append(vect/corr_by_hist)
                else: 
                    temp = np.empty((len(vect)))
                    temp[:] = np.NaN
                    corred_set.append(temp)
            
            corred.append(np.float64(corred_set))
            # print('hist_std for 2pc')
            # print(corred_set[1])
            ax = show.shadyStats(x, corred_set[0], corred_set[1], '', scale_std=2, ax=ax)
        plt.xlim([0, 31])
        plt.ylim([-2, 5])
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
    # print('bin_width in intracone distance')
    # print(bin_width)
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
    # print('mean_nearest')
    # print(mean_nearest)
    # print('std_nearest')
    # print(std_nearest)

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
        coord = calc.hexgrid(num2gen,
                             hex_radius,
                             [0, img_x],
                             [0, img_y])

        num_cones_placed = coord.shape[1]
        cone_density = num_cones_placed / (img_x * img_y)

        # before = show.scatt(np.squeeze(coord[0, :, :]), id +' before')
        coord = util.trim_random_edge_points(coord, num_cones, [0, img_x], [0, img_y])
        # after = show.scatt(np.squeeze(coord[0, :, :]), id + ' after')

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
        # ax = show.plotKwargs({},'')
        for mos in np.arange(0, num2gen):
            # print('num hexgrid pre trim')
            # print(spaced_coord[mos,:,:].shape)
            # print('num unique hexgrid')
            # print(np.unique(np.squeeze(spaced_coord[mos,:,:]), axis=0).shape)
            non_nan = np.array(np.nonzero(~np.isnan(spaced_coord[mos,:,0]))[0], dtype=int)
            # print('num spaced pre-trim')
            # print(non_nan.shape)
            # ax0 = show.scatt(all_coord, 'trim test: before', plot_col = 'y')
            # ax0 = show.scatt(spaced_coord[mos,:,:], 'trim test: before', plot_col = 'r', ax=ax0)
            # print('size pre-trimmed edge points')
            # print(spaced_coord[mos, non_nan, :].shape)
            beep = np.unique(np.squeeze(spaced_coord[mos, non_nan, :]), axis=0)
            # print('num unique coords')
            # print(beep.shape)
            temp[mos, :, :] = util.trim_random_edge_points(spaced_coord[mos, non_nan, :], cones2place, x_dim, y_dim)
            # ax2 = show.scatt(all_coord, '', plot_col='y')
            # ax2 = show.scatt(temp[mos, :, :], 'coneLocked maximally spaced',plot_col='r', ax=ax2)
            # col = util.randCol()
            # print(temp[mos,:,:].shape)
            # ax = show.scatt(temp[mos,:,:], 'conelocked spacified overlay', plot_col = col, ax=ax)
            

        coord = temp
        # print('size of spaced coordinates saved')
        # print(coord.shape)
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
    for proc in tiers[1]:
        if proc in user_param['analyses_to_run'][0]:
            print('Running process "' + proc + '" on ' + str(len(processes[proc])) + ' mosaic coordinate files...') 
            for ind in processes[proc]:
                param = unpackThisParam(user_param, ind)
                globals()[sav_cfg[proc]['process']](param, sav_cfg)


