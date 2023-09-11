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
# from py import process
# from shapely.geometry.polygon import Polygon
import scipy
# from scipy import spatial

import random


## --------------------------------SECONDARY ANALYSIS FUNCTIONS--------------------------------------
def degree_structure_process(param, sav_cfg):
    proc = 'degree_structure'
    proc_vars = sav_cfg[proc]['variables']
    proc_metrics = sav_cfg[proc]['metrics']
    sav_fl = param['sav_fl']
    mos_types = ['Ru', 'Rr', 'M', 'Sr', 'Su','Dm']
    mos_labels = ['monteCarlo_uniform', 'monteCarlo_coneLocked', 'measured', 'coneLocked_maxSpacing', 'hexgrid_by_density', 'dmin']

    metric_colors = ['rebeccapurple', 'dodgerblue', 'white', 'darkorange', 'firebrick','chartreuse']
    print(sav_fl)
    for met in proc_metrics:
        metric = met[1]
        mosaic = param['mosaic']
        # print([metric_key[x] == met for x in np.arange(0,len(metric_key))])
        data, metrics_taken, metric_key, mos_key = getDataForMosaicSetProcess(sav_fl, [met])

        min_max_per_mosaic = np.empty([2,6])
        min_max_per_mosaic[:] = np.nan

        flat_data = [[], [], [], [], [], []]
        for mi in np.arange(0, len(data)):
            for li, lab in enumerate(mos_labels):
                if mos_key[mi] == lab:
                    flat_data[mi] = data[mi].flatten()
                    min_max_per_mosaic[0,mi] = np.nanmin(data[mi].flatten())
                    min_max_per_mosaic[1,mi] = np.nanmax(data[mi].flatten())

        Ru_data = flat_data[0]
        Rr_data = flat_data[1]
        M_data = flat_data[2]
        Sr_data = flat_data[3]
        Su_data = flat_data[4]
        Dm_data = flat_data[5]

        min_val = np.nanmin(min_max_per_mosaic[0,:])
        max_val = np.nanmax(min_max_per_mosaic[1,:])

        # avoid the extremely high upper limit of regularity indices
        # print('metric')
        # print(met[1])
        met_split = met[1].split('_')
        if met_split[len(met_split)-1] == 'regularity':
            max_val = np.nanmin([max_val,100])

  
        if M_data.shape[0] > 1:
            #print([met[1] + ': apply FD to Ru'])
            non_nans = np.nonzero([not np.isnan(M_data[x]) for x in np.arange(0, M_data.size)])[0]
            n = non_nans.size
            temp_bin_edge = np.histogram_bin_edges(M_data[non_nans], bins='fd') 
        
        elif M_data.shape[0] == 1:
            #print([met[1] + ': apply FD to M'])
            non_nans = np.nonzero([not np.isnan(Rr_data[x]) for x in np.arange(0, Rr_data.size)])[0]
            n = non_nans.size
            # if met[1] == 'exclusion_radius':
            #     temp_bin_edge = [0,3]
            # elif met[1] == 'exclusion_area':
            #     temp_bin_edge = [0,.5]
            # elif met[1] == 'exclusion_obed':
            #     temp_bin_edge = [0,.05]

            temp_bin_edge = np.histogram_bin_edges(Rr_data[non_nans], bins='scott') 
            temp_bin_edge = temp_bin_edge

        else:
            print('uhm, ' + met[1] + ' is empty')
        
        bin_width = temp_bin_edge[1] - temp_bin_edge[0]
        round_min = np.floor(min_val)
        round_max = np.ceil(max_val)
        #print(met)
        hist_extents = [round_min, round_max + ((round_max-round_min) % bin_width)]
        # print(hist_extents)
        # print('')
        bin_edge = np.arange(hist_extents[0], hist_extents[1], bin_width)
        # print('BIN EDGE')
        # print(bin_edge.shape)
        metric_values = [Ru_data, Rr_data, M_data, Sr_data, Su_data, Dm_data]
        metric_hists = np.empty([6,bin_edge.shape[0]-1])
        metric_hists[:] = np.nan

        # ax = show.plotKwargs({}, "")
        for ind, vals in enumerate(metric_values):    
            # print(mos_types[ind])
            # if met == 'exclusion_factor':
                # print('EXCLUSION FACTORRR')
                # print(vals)
            if vals.shape[0] > 1:
                hist, bin_edge = np.histogram(vals, bin_edge)
                non_nan = np.nonzero(~np.isnan(vals))[0]
                hist = hist/non_nan.shape[0]
                metric_hists[ind, :] = hist
                # hist_x, hist_y, blah, bloo = util.reformat_stat_hists_for_plot(bin_edge, metric_hists[ind, :], np.zeros([metric_hists[ind, :].shape[0],]))
                # ax = show.line(hist_x, hist_y, mosaic + ' ' + met[1], plot_col = metric_colors[ind], linewidth = 2, ax = ax)
                
                #plt.stairs(metric_hist, bin_edge, color = metric_colors[])
            else:
                # print('VAL')
                # print(vals.shape)
                # plt.scatter(vals[0], 0, 50, 'w', 'o', 'filled')
                try:
                    metric_hists[ind, 0] = np.array([vals[0]])
                except:
                    print('WARNING: still that thing where histogram stuff in degree structure fails for overly discrete metric outputs')
    
        #print(np.concatenate([Rr_data,Sr_data]))
        max_constrained = np.nanmax(np.concatenate([Rr_data,Sr_data]))
        min_constrained = np.nanmin(np.concatenate([Rr_data,Sr_data]))
        max_unconstrained = np.nanmax(np.concatenate([Ru_data,Su_data]))
        min_unconstrained = np.nanmin(np.concatenate([Ru_data,Su_data]))

        cone_constrained_span = max_constrained - min_constrained
        unconstrained_span = max_unconstrained - min_unconstrained
        constraint_ratio = cone_constrained_span/unconstrained_span

        # ax = show.plotKwargs({}, "")
        # ax = show.line([min_constrained, max_constrained], [-.05, -.05], '', ax=ax, linewidth=2, plot_col = 'y')
        # ax = show.line([min_unconstrained, max_unconstrained], [-.1, -.1], met[1] + ' constraint_ratio: ' + str(constraint_ratio), ax=ax, linewidth=2, plot_col = 'g')

        constrained_inds = [1,2,3]

        # ax = show.plotKwargs({}, '')
        # for i in constrained_inds:
        #     hist = metric_hists[i,:]
            # print(mos_types[i])
            # print(np.nonzero(~np.isnan(hist))[0].shape)
            # print(np.nonzero(~np.isnan(hist))[0].shape[0] > 1)
            # if np.nonzero(~np.isnan(hist))[0].shape[0] > 1:
            #     hist_x, hist_y, blah, bloo = util.reformat_stat_hists_for_plot(bin_edge, hist, np.zeros([hist.shape[0],]))
            #     ax = show.line(hist_x, hist_y, mosaic + ' ' + met[1], plot_col = metric_colors[i], linewidth = 2, ax = ax)
                
                #plt.stairs(metric_hist, bin_edge, color = metric_colors[])
            # else:
            #     plt.scatter(hist[0], 0, 50, 'w', 'o', 'filled')
        # ax.set_xlim([min_constrained - bin_width, max_constrained + bin_width])
        #print(met)
        try: 
        
            KS_Rr_Sr = scipy.stats.kstest(Rr_data, Sr_data)
            KS_Rr_Sr_statistic = KS_Rr_Sr.statistic
            KS_Rr_Sr_pval = KS_Rr_Sr.pvalue
            #print(mosaic + ' ' + met[1])
            #print('KS_Rr_Sr: (stat) ' + str(KS_Rr_Sr.statistic) + ' (p) ' + str(KS_Rr_Sr.pvalue))
            #if KS_Rr_Sr.pvalue < .05:
            #    print('Rr & Sr SIGNIFICANTLY DIFFERENT')
            # print(met)

            Q_Rr = np.quantile(Rr_data[np.nonzero(~np.isnan(Rr_data))[0]], [.025, .975])
            Q_Sr = np.quantile(Sr_data[np.nonzero(~np.isnan(Sr_data))[0]], [.025, .975])
            percentiles_Rr = np.quantile(Rr_data[np.nonzero(~np.isnan(Rr_data))[0]], np.arange(.001,.999,.001))
            percentiles_Sr = np.quantile(Sr_data[np.nonzero(~np.isnan(Sr_data))[0]], np.arange(.001,.999,.001))
            Q_Ru = np.quantile(Ru_data[np.nonzero(~np.isnan(Ru_data))[0]], [.025, .975])
            #print('Su_data')
            #print(Su_data.shape)
            #if Su_data.shape[0] < 200:
            #    print(Su_data)
            #Q_Su = np.quantile(Su_data[np.nonzero(~np.isnan(Su_data))[0]], [.025, .975])
            Q_Dm = np.quantile(Dm_data[np.nonzero(~np.isnan(Dm_data))[0]], [.025, .975])
            percentiles_Rr = np.quantile(Ru_data[np.nonzero(~np.isnan(Ru_data))[0]], np.arange(.001,.999,.001))
            percentiles_Sr = np.quantile(Su_data[np.nonzero(~np.isnan(Su_data))[0]], np.arange(.001,.999,.001))
            percentiles_Dm = np.quantile(Dm_data[np.nonzero(~np.isnan(Dm_data))[0]], np.arange(.001,.999,.001))
            #print('Rr .95 confidence intervals: ' + str(Q_Sr))
            #print('Rr .95 confidence intervals: ' + str(Q_Sr))
            if M_data.shape[0] > 1:
                KS_M_Rr = scipy.stats.kstest(M_data, Rr_data)
                KS_M_Rr_statistic = KS_M_Rr.statistic
                KS_M_Rr_pval = KS_M_Rr.pvalue
                KS_M_Sr = scipy.stats.kstest(M_data, Sr_data)
                KS_M_Sr_statistic = KS_M_Sr.statistic
                KS_M_Sr_pval = KS_M_Sr.pvalue
                KS_M_Dm = scipy.stats.kstest(M_data, Dm_data)
                KS_M_Dm_statistic = KS_M_Dm.statistic
                KS_M_Dm_pval = KS_M_Dm.pvalue

            #    print('KS_M_Rr:  (stat) ' + str(KS_M_Rr.statistic) + ' (p) ' + str(KS_M_Rr.pvalue))
            #    print('KS_M_Sr:  (stat) ' + str(KS_M_Sr.statistic) + ' (p) ' + str(KS_M_Sr.pvalue))
            #    if KS_M_Rr.pvalue < .05:
            #        print('M & Rr SIGNIFICANTLY DIFFERENT')
            #    if KS_M_Sr.pvalue < .05:
            #        print('M & Sr SIGNIFICANTLY DIFFERENT')
                degree_structure = np.nan
                DS_Q_Rr = np.nan
                DS_Q_Sr = np.nan
                DS_Q_Dm = np.nan
                DS_Rr = np.nan
                DS_Sr = np.nan
                DS_Dm = np.nan
            else:
                KS_M_Rr = np.nan
                KS_M_Sr = np.nan
                KS_M_Dm = np.nan
                KS_M_Rr_statistic = np.nan
                KS_M_Rr_pval = np.nan
                KS_M_Sr_statistic = np.nan
                KS_M_Sr_pval = np.nan
                KS_M_Dm_statistic = np.nan
                KS_M_Dm_pval = np.nan
       
                max_val = np.nanmax(np.concatenate([Sr_data, Rr_data, Dm_data]))
                min_val = np.nanmin(np.concatenate([Sr_data, Rr_data, Dm_data]))

                degree_structure = ((M_data[0] - min_val) / (max_val-min_val))

                DS_Q_Rr = np.array([(Q_Rr[0] - min_val / (max_val-min_val)),
                                    (Q_Rr[1] - min_val / (max_val-min_val))])
                DS_Q_Sr = np.array([(Q_Sr[0] - min_val / (max_val-min_val)),
                                    (Q_Sr[1] - min_val / (max_val-min_val))])
                DS_Q_Dm = np.array([(Q_Dm[0] - min_val / (max_val-min_val)),
                                    (Q_Dm[1] - min_val / (max_val-min_val))])
                
                DS_Rr = np.array([(z - min_val) / (max_val-min_val) for z in Rr_data])
                DS_Sr = np.array([(z - min_val) / (max_val-min_val) for z in Sr_data])
                DS_Dm = np.array([(z - min_val) / (max_val-min_val) for z in Dm_data])

            data_to_set = util.mapStringToLocal(proc_vars, locals())
        except:
            print('Exception: Degree Structure not calculated for ' + met + '(likely because metric is empty, which may or may not be appropriate)')
            data_to_set = util.mapStringToNan(proc_vars)
    
        flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set, prefix='mosaic_set_' + metric + '_')


        
        
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
        corr_by_pi = file[corr_by + '_' + 'two_point_correlation']['corr_by_pi'][()]
        max_bins = file[corr_by + '_' + 'two_point_correlation']['max_bins'][()]
        bin_edge = file[corr_by + '_' + 'two_point_correlation']['max_bin_edges'][()]
        sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
        if sim_hexgrid_by == 'rectangular':
            hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]
        elif sim_hexgrid_by == 'voronoi':
            hex_radius = file['measured_voronoi']['hex_radius'][()]
        else:
            print('ack!!! problem getting hex_radius in metrics_of_2PC_process')
        
        emp_max_rad = np.nanmax(file['coneLocked_maxSpacing']['hex_radii_used'][()])
    
    analysis_x_cutoff = int(np.argmin(np.abs((bin_edge - (2 * hex_radius)))))

    corr_by_corr = corr_by_corr[:, 0:analysis_x_cutoff]
    corr_by_mean = np.nanmean(corr_by_corr, axis=0)
    corr_by_pi = corr_by_pi[:, 0:analysis_x_cutoff]

    # print(corr_by_mean)
    # print(corr_by_pi)
    #corr_by_std = np.nanstd(corr_by_corr, axis=0)

    for ind, PD in enumerate(PD_string):
        num_mosaic = coord[ind].shape[0]
        print('     Running metrics of 2PC on ' + str(num_mosaic) + " " + PD_string[ind] + ' mosaics...') 
        
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
        exclusion_factor = np.empty([crop_corr.shape[0],])
        # exclusion_area = np.empty([crop_corr.shape[0],])
        # max_obed_exclusion_area = np.empty([crop_corr.shape[0],])
        # exclusion_obed = np.empty([crop_corr.shape[0],])
        first_peak_rad[:] = np.nan
        exclusion_bins[:] = np.nan
        exclusion_radius[:] = np.nan
        exclusion_factor[:] = np.nan
        # exclusion_area[:] = np.nan
        # max_obed_exclusion_area[:] = np.nan
        # exclusion_obed[:] = np.nan
 
        printStuff = False
        for m in np.arange(0,crop_corr.shape[0]):
            dearth_bins.append(np.nonzero(crop_corr[m,:]  <  corr_by_pi[0,:])[0])
            if m == 0 and printStuff:
                print(dearth_bins)

            peak_bins.append(np.nonzero(crop_corr[m,:] > corr_by_pi[1,:])[0])

            first_peak_rad[m] = np.nan
            if len(peak_bins[m]) > 0:

                first_peak_rad[m] = bin_edge[peak_bins[m][0]+1]
        
            if m == 0 and printStuff:
                ax = show.plotKwargs({'figsize':10}, '')

                corr_by_x, corr_by_y, corr_by_y_plus, corr_by_y_minus = util.reformat_stat_hists_for_plot(bin_edge, corr_by_mean, corr_by_pi)
                ax = show.line(corr_by_x, corr_by_y, '', ax=ax, plot_col = 'firebrick')
                ax.fill_between(corr_by_x, corr_by_y_plus, corr_by_y_minus, color='firebrick', alpha=.7)

                hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bin_edge, crop_corr[m,:], np.zeros([crop_corr.shape[1],]))
                ax = show.line(hist_x, hist_y, '', ax=ax, plot_col = 'w')
                ax.fill_between(hist_x, hist_y_plus, hist_y_minus, color='royalblue', alpha=.7)

                ax.scatter(bin_edge[dearth_bins[m]] + bin_width/2, crop_corr[m, dearth_bins[m]], color='g')
                ax.scatter(bin_edge[peak_bins[m]] + bin_width/2, crop_corr[m, peak_bins[m]], color='y')


            if np.all((dearth_bins[m].shape[0] > 0) and (dearth_bins[m][0] == 0)):
                if dearth_bins[m].shape[0] == 1:
                    exclusion_bins[m] = 1
                    if m == 0 and printStuff:
                        print('option 1')
                else:
                    diff_dearth = np.diff(dearth_bins[m])
                    diff1 = [d == 1 for d in diff_dearth.tolist()]               
                    zeros = np.nonzero([d == 0 for d in diff1])[0]
                    if m == 0 and printStuff:
                        print(diff_dearth)
                        print(diff1)
                        print(zeros)
                    if not np.any(zeros):
                        if np.all([diff1[x] ==1 for x in np.arange(0,len(diff1))]):
                            exclusion_bins[m] = dearth_bins[m].shape[0]
                            # if m == 0 and printStuff:
                                # print('option 2')
                        else:
                            exclusion_bins[m] = 1
                            # if m == 0 and printStuff:
                                # print('option 3')
                    else:
                        first_zero = zeros[0] + 1
                        exclusion_bins[m] = first_zero
                        # if m == 0 and printStuff:
                        #     print('option 4')

            else:
                exclusion_bins[m] = 0
                # if m == 0 and printStuff:
                #     print('option 5')
            
            

            #exclusion_bins[m] = int(exclusion_bins[m])

            if exclusion_bins[m] > 0:

                exclusion_radius[m] = bin_edge[int(exclusion_bins[m])]

            else: 
                exclusion_radius[m] = 0
            if m == 0 and printStuff:
                print('exclusion_bins')
                print(exclusion_bins[m])
                print('exclusion_radius')
                print(exclusion_radius[m])

            if m == 0 and printStuff:
                ax = show.line([exclusion_radius, exclusion_radius], [-1, 1], '', plot_col = 'g', ax=ax)

            # print('exclusionary radius)')
            # print(exclusion_radius[m])
            # exclusion_area[m] = 0
            # max_obed_exclusion_area[m] = 0

            # print('EXCLUSION AREA')
            if exclusion_radius[m] > 0:
                
                # print('exclusionary bins')
                # print(exclusion_bins[m])
                for b in np.arange(0, int(exclusion_bins[m])):
                    # print('EXCLUsION FACTOR')
                    # print(exclusion_radius[m])
                    # print(hex_radius)
                    # print(exclusion_radius[m]/hex_radius)
                    try:
                        exclusion_factor[m] = (exclusion_radius[m]/emp_max_rad)**2
                        
                        #exclusion_factor[m] = exclusion_radius[m]/hex_radius[0]
                    except:
                        print('could not set exclusion factor for ')
     

                    
                    # print('            ' + str(exclusion_area[m]))
                    # print('                   ' + str((corr_by_mean[b] + (2 * corr_by_std))-crop_corr[m, b]))
                    # print(corr_by_mean[b] + (2 * corr_by_std[b]))
                    # print(crop_corr[m, b])
                    #print(((corr_by_mean[b] - (2 * corr_by_std[b]))-crop_corr[m, b]))
                    # Rumean_minus2std = (corr_by_pi[0,b])
                    # test = crop_corr[m, b]
                    # exclusion_area[m] = exclusion_area[m] + (bin_width * (Rumean_minus2std-test))
                    # max_obed_exclusion_area[m] = max_obed_exclusion_area[m] + (bin_width * (Rumean_minus2std-(-1)))
              
                    # print(exclusion_radius[m])
                    # print(exclusion_area[m])
                    # if m == 0 and printStuff:
                    #     ax.fill_between(bin_edge[b:b+2],
                    #                     [crop_corr[m,b], crop_corr[m,b]],
                    #                     [corr_by_pi[0,b], corr_by_pi[0,b]], 
                    #                     color='g', alpha=.5)
            else:
                exclusion_factor[m] = 0
            # if max_obed_exclusion_area[m] > 0:
            #     exclusion_obed[m] = exclusion_area[m] / max_obed_exclusion_area[m]
            # else: 
            #     max_obed_exclusion_area[m] = 0
        
            if m == 0 and printStuff:
                ax.set_title(PD + ' mosaic #' + str(m))
                ax.set_xticks(bin_edge[0:analysis_x_cutoff])
                ax.set_ylim([-1.5, 4])
        
        # print(PD)
        # print('radii')
        # print(exclusion_radius)
        # print('bins1')
        # print(exclusion_radius/bin_width)
        # # print('bins2')
        # # print(exclusion_bins)
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
    # bin_width = param['bin_width']
    to_be_corr = param['to_be_corr']
    sim_to_gen = param['sim_to_gen']
    corr_by = param['corr_by']

    #need this for the PD_string that will sort the 2PC results to each mosaic types save spot
    coord, PD_string = getDataForPrimaryProcess(sav_fl)

    dist_hists = []
    max_bins = 0
    with h5py.File(sav_fl, 'r') as file:
        xlim = [0, file['input_data']['img_x'][()]]
        ylim = [0, file['input_data']['img_y'][()]]
        for ind, PD in enumerate(PD_string):
            num_mosaic = coord[ind].shape[0]
            print('     Running two point correlation on ' + str(num_mosaic) + " " + PD_string[ind] + ' mosaics...') 

            # if PD == corr_by + '_':
            #     corr_by_ind = ind
            #     corr_by_mean = file[PD + 'intracone_dist']['hist_mean'][()]
            #     corr_by_std = file[PD + 'intracone_dist']['hist_std'][()]
            #     corr_by_pi = file[PD + 'intracone_dist']['poisson_intervals'][()]
            
            dist_hists.append(file[PD + 'intracone_dist']['hist_mat'][()])
            temp_edges = file[PD + 'intracone_dist']['bin_edge'][()]
            num_bins = temp_edges.shape[0]

            # ax = show.plotKwargs({}, '')

            # for m in np.arange(0, dist_hists[ind].shape[0]):
            #     plt.stairs(dist_hists[ind][m,:], temp_edges)

            if num_bins > max_bins:
                max_bins = num_bins
                max_bin_edges = temp_edges

            bin_width = temp_edges[1] - temp_edges[0]


    
    # create the population of simulated mosaics to be correlated against
    numcorrmos = 100000
    numcone = coord[0].shape[1]
    print('     Generating ' + str(numcorrmos) + " random uniform mosaics for 2PC...") 
    mc_coord = calc.monteCarlo_uniform(numcone, numcorrmos, xlim, ylim)

    # copied from smp.intracone_dist_process, variable names modified ------------------------------------
    dist = np.zeros((numcorrmos, numcone, numcone))
    mean_nearest = np.zeros(numcorrmos)
    std_nearest = np.zeros(numcorrmos)
    hist = np.empty(numcorrmos, dtype=np.ndarray)

    print('     Running intracone distances on ' + str(numcorrmos) + " random uniform mosaics for 2PC...") 

    for mos in np.arange(0, numcorrmos):
        this_coord = mc_coord[mos, :, :]
        dist[mos, :, :], mean_nearest[mos], std_nearest[mos], hist[mos], bin_edge, annulus_area = intracone_dist_common(this_coord.squeeze(), bin_width, False, False)
        if hist[mos].shape[0] > max_bins:
            max_bins = hist[mos].shape[0]
    
    
    poisson_intervals = np.empty([2, max_bins])
    poisson_intervals[:] = np.nan


    # this is just to convert the returned histograms into a rectangular array
    # (this can't be done in advance because of...slight variability in the number of bins returned? why?)
    hist_mat = np.zeros([numcorrmos, max_bins])
    for mos in np.arange(0, numcorrmos):
        hist_mat[mos, 0:hist[mos].shape[0]] = hist[mos]

    corr_by_hists = hist_mat

    #sierra why did you do this
    corr_by_poisson_95conf = poisson_intervals
    corr_by_pi = corr_by_poisson_95conf

    if numcorrmos > 1:
        for b in np.arange(0, hist_mat.shape[1]):
            poisson_intervals[0, b], poisson_intervals[1, b] = calc.poisson_interval(np.nanmean(hist_mat[:,b]))

    #----------------------------------------------------------------------------

    # while len(bin_edge) < max_hist_bin + 1:
    #     bin_edge = np.append(bin_edge, np.max(bin_edge)+bin_width)

    corr_by_mean = np.nanmean(hist_mat, axis=0)
    corr_by_std = np.nanstd(hist_mat, axis=0)

    # if corr_by_mean.shape[0] < max_bins:
    
    corr_by_mean = util.vector_zeroPad(corr_by_mean, 0, max_bins - (corr_by_mean.shape[0]))
    corr_by_std = util.vector_zeroPad(corr_by_std, 0, max_bins - (corr_by_std.shape[0]))
    corr_by_pi_row0= util.vector_zeroPad(corr_by_pi[0, :], 0, max_bins - (corr_by_pi.shape[1]))
    corr_by_pi_row1 = util.vector_zeroPad(corr_by_pi[1, :], 0, max_bins - (corr_by_pi.shape[1]))
    corr_by_pi = np.empty([2, max_bins])
    corr_by_pi[0,:] = corr_by_pi_row0
    corr_by_pi[1,:] = corr_by_pi_row1

    # else:
    #     corr_by_pi_row0 = corr_by_pi[0, :]
    #     corr_by_pi_row1 = corr_by_pi[1, :]

    if coord:
        hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(max_bin_edges, corr_by_mean, corr_by_pi)

        # ax = show.plotKwargs({}, '')
        # ax = show.line(hist_x, hist_y, '', plot_col='w', ax=ax)
        # ax.fill_between(hist_x, 
        #                 hist_y_plus, 
        #                 hist_y_minus, color='royalblue', alpha=.7)
        # ax.set_title('Random uniform statistics (two-sided .95 confidence interval)')

        corr_by_hist = corr_by_mean

        # for each mosaic to be correlated against the msoaics
        for ind, dist_mat in enumerate(dist_hists):

            # preallocation for new 2PC matrix
            corred = np.empty([dist_mat.shape[0], max_bins])
            corred[:] = np.nan
            #ax = show.plotKwargs({},'')

            corr_pi_row1 = calc.corr(corr_by_pi_row0, corr_by_hist)
            corr_pi_row2 = calc.corr(corr_by_pi_row1, corr_by_hist)
            corr_by_pi = np.empty([2, max_bins])
            corr_by_pi[0,:] = corr_pi_row1
            corr_by_pi[1,:] = corr_pi_row2

            for mos in np.arange(0, dist_mat.shape[0]):
                hist = dist_mat[mos,:]

                if hist.shape[0] < max_bins:
                    hist = util.vector_zeroPad(hist, 0, max_bins - hist.shape[0])

                # new 2PC calculation
                corred[mos,:] = calc.corr(hist, corr_by_hist)
                # ax.stairs(corred[mos,:], max_bin_edges)

            corred = np.float64(corred)

            data_to_set = util.mapStringToLocal(proc_vars, locals())
            flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set, prefix=PD_string[ind])

    

## --------------------------------PRIMARY ANALYSIS FUNCTIONS--------------------------------------

def drp_process(param, sav_cfg):
    """
    Inputs
    ------

    Outputs
    -------


    """
    proc = 'drp'
    proc_vars = sav_cfg[proc]['variables']

    sav_fl = param['sav_fl']
    print(sav_fl)
    coord, PD_string = getDataForPrimaryProcess(sav_fl)
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
        
        density = file['measured_voronoi']['density'][()]

    for ind, point_data in enumerate(coord):
        # print('COORD SHAPE')
        # print(point_data.shape)

        if len(point_data.shape) == 2:
            print('ack 2D data!!!')
        
        print('     Running drps for ' + str(point_data.shape[0]) + " " + PD_string[ind] + ' mosaics...') 
        
        # Generate an array of intercone distances and compute the DRP for all
        # distances (see comment for AddEdgeCorrection

        # This is a correction factor that correct for annuli that extend beyond
        # the field of points that are analyses (i.e edge effect). If you have a
        # square fields of contiguous samples, then this correction is needed to yeoidl accruate estimates of density at all distances.
        # If you have a non-contiguous array over a non-square field, then its best
        # to set this zero.
        
        with h5py.File(sav_fl, 'r') as file:
            PD = PD_string[ind]
            bin_edges = file[PD + 'intracone_dist']['bin_edge'][()]
            bin_width = bin_edges[1]-bin_edges[0]

        # max_pixel_distance = 100
        
        # higher_bins = np.nonzero([bin_edges[i] > max_pixel_distance for i in np.arange(0,bin_edges.shape[0])])[0]
        # bin_edges = bin_edges[0:higher_bins[0]+1]
        max_pixel_distance = bin_edges[len(bin_edges)-1]
        num_bins = bin_edges.shape[0]-1
        
        add_edge_correction = True
        num_mos = point_data.shape[0]
        num_cone = point_data.shape[1]
        drp = np.empty([num_mos, 3, num_bins])
        expected_number = np.empty([num_mos, num_bins])
        expected_drp = np.empty([num_mos, num_bins])
        effective_radius = np.empty([num_mos, ])
        packing_factor = np.empty([num_mos, ])
        drp[:] = np.nan
        expected_number[:] = np.nan
        expected_drp[:] = np.nan
        effective_radius[:] = np.nan
        packing_factor[:] = np.nan

        if point_data.shape[1] > 5: 
            for m in np.arange(0, num_mos):

                temp_coord = np.squeeze(point_data[m,:,:])
                temp = np.empty([temp_coord.shape[1], temp_coord.shape[0]])
                temp[0,:] = temp_coord[:,0]
                temp[1,:] = temp_coord[:,1]
                temp_coord = temp

                drp[m, 0:2, :] = calc.rodieck_func(temp_coord, img_x[1]-img_x[0], img_y[1]-img_y[0], max_pixel_distance, num_bins, add_edge_correction)
                # print('DRP')
                # print(drp[m,:,:])
                # THIS whole section is a series of calculation to determine packing parameters.
                # It IS DESCRIBED IN RODIECK P99 COLUMN 2. IT IS NOT A USEFUL MEASURE FOR
                # NON-CONTIGUOUS DATASETS.

                # name_this_variable_better = False

                # if name_this_variable_better:
                vol = 0
                count = 0

                for i in np.arange(0,num_bins):
                    # this number is computed as the area of each annulus times density times number of cones
                    # (since all cones are added together in DRP)
                    expected_number[m,i] = num_cone * density * np.pi * (2 * i + 1) * np.square(max_pixel_distance / num_bins)

                    if expected_number[m,i] > drp[m, 1, i] and count < 1:
                        # add the open area above the actual densities and below the average density
                        vol += expected_number[m,i] - drp[m, 1, i]
                    else:
                        count = 1

                vol /= num_cone
                effective_radius[m] = np.sqrt(vol / (np.pi * density))
                maximum_radius = np.sqrt(np.sqrt(4.0 / 3.0) / density)
                packing_factor[m] = np.square(effective_radius[m]/ maximum_radius)

                # print(f"The effective radius is {effective_radius}")
                # print(f"The maximum radius is {maximum_radius}")
                # print(f"The packing factor is {np.square(effective_radius / maximum_radius)}")
                # print('')
                # # convert all value to minutes of arc
                # for i in range(num_bins):
                #     drp[2, i] = drp[1, i] * np.square(pixels_per_degree) / (num_sub_points * (np.pi * np.square(max_pixel_distance / num_bins) * (2 * i + 1)))
                #     drp[0, i] = 60.0 / pixels_per_degree * (i + 1) * max_pixel_distance / num_bins
            
                for i in range(num_bins):
                    drp[m, 2, i] = drp[m, 1, i] * 1 / (num_cone * (np.pi * np.square(max_pixel_distance / num_bins) * (2 * i + 1)))
                    #drp[0, i] = 60.0 / pixels_per_degree * (i + 1) * max_pixel_distance / num_bins
                    expected_drp[m,i] = expected_number[m,i] * 1 / (num_cone * (np.pi * np.square(max_pixel_distance / num_bins) * (2 * i + 1)))

                counts_per_bin = np.squeeze(drp[:,1,:])
                drp_mean = np.nanmean(drp[:,2,:], axis=0)
                drp_all = drp[:,2,:]
                drp_std = np.nanstd(drp_all, axis=0)
                drp_ste = np.divide(drp_std, counts_per_bin) 

                #percentiles_per_bin_across_mosaics = np.quantile(Rr_data[np.nonzero(~np.isnan(Rr_data))[0]], np.arange(.001,.999,.001))
                


                # plt.stairs(np.squeeze(drp_mean), bin_edges)
                # plt.xlabel("Distance(pixels)")
                # if add_edge_correction:
                #     plt.ylabel("Cone density per pixel squared")
                # else:
                #     plt.ylabel("uncorrected density (a.u.)")

                # plt.title("Density Recovery Profile")
                # plt.show()

            data_to_set = util.mapStringToLocal(proc_vars, locals())

        else:
            data_to_set = util.mapStringToNan(proc_vars)

        flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set,
                                        prefix=PD_string[ind])
        



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
        # print('COORD SHAPE')
        # print(point_data.shape)
        if len(point_data.shape) == 2:
            print('ack 2D data!!!')
        
        print('     Running voronois for ' + str(point_data.shape[0]) + " " + PD_string[ind] + ' mosaics...') 
        
        # ax = show.scatt(np.squeeze(point_data), 'points to be voronoid')

        [regions, vertices, ridge_vertices, ridge_points, point_region] = calc.voronoi(point_data)
        
        bound_cones = calc.get_bound_voronoi_cells(point_data, img_x, img_y)
        # print('BOUND CONES')
        # print(bound_cones)
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
        # print('NEIGHBORS CONES')
        # print(neighbors_cones)
        
        if np.nansum(bound_regions[0]) > 5:
            # print(np.nansum(bound_regions[0]))
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
                non_nan = np.nonzero(~np.isnan(point_data[m,:,0]))[0]
                for nc in np.arange(0, non_nan.shape[0]):
                    not_nans = [~np.isnan(neighbors_cones[m, nc, x]) for x in np.arange(0, neighbors_cones.shape[2])]
                    neighb_cones = np.array(neighbors_cones[m, nc, np.nonzero(not_nans)[0]], dtype=int)
                    if ~np.any(np.nonzero(bound_cones[m][neighb_cones])[0]):
                        bound_cones[m][nc] = 0
                        r = point_region[m][nc]
                        bound_regions[m][int(r)] = 0
                        neighbors_cones[m, nc, :] = np.nan
                    else:
                        #get ICDs while we're here
                        icd[m, nc, np.array(np.nonzero(not_nans)[0], dtype=int)] = dists[nc, np.array(neighbors_cones[m, nc, np.nonzero(not_nans)[0]], dtype=int)]
                # print('ICD')
                # print(icd)
                icd_mean[m] = np.nanmean(icd[m, :, :])
                icd_std[m] = np.nanstd(icd[m, :, :])
                icd_regularity[m] = icd_mean[m]/icd_std[m]

                
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
            
            try:
                maxnum = int(np.nanmax([np.nanmax(num_neighbor[s]) for s in np.arange(0, len(num_neighbor))]))
            except:
                print(num_neighbor)



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

            max_pr = 0
            for i, p in enumerate(point_region):
                max_pr = np.amax([max_pr, p.shape[0]]) 

            temp_pr = np.empty([len(point_region), max_pr])
            temp_pr[:] = np.nan
            for i, p in enumerate(point_region):
                temp_pr[i, 0:len(p)] = p

            max_br = 0
            for i, br in enumerate(bound_regions):
                max_br = np.amax([max_br, br.shape[0]]) 

            temp_br = np.empty([len(bound_regions), max_br])
            temp_br[:] = np.nan
            for i, br in enumerate(bound_regions):
                temp_br[i, 0:len(br)] = br

            max_bc = 0
            for i, bc in enumerate(bound_cones):
                max_bc = np.amax([max_bc, bc.shape[0]]) 

            temp_bc = np.empty([len(bound_cones), max_bc])
            temp_bc[:] = np.nan
            for i, bc in enumerate(bound_cones):
                temp_bc[i, 0:len(bc)] = bc

            point_region = temp_pr
            bound_regions = temp_br
            bound_cones = temp_bc
        else:
            print(sav_fl)
            print('LESS THAN 5 BOUND CONES')
            regions = np.nan
            vertices = np.nan
            point_region = np.nan
            bound_regions = np.nan
            bound_cones = np.nan
            voronoi_area = np.nan
            voronoi_area_mean = np.nan
            voronoi_area_std = np.nan
            neighbors_cones = np.nan
            num_neighbor = np.nan
            num_neighbor_mean = np.nan
            num_neighbor_std = np.nan
            icd = np.nan
            icd_mean = np.nan
            icd_std = np.nan
            icd_regularity = np.nan
            voronoi_area_regularity = np.nan
            num_neighbor_regularity = np.nan
            density = np.nan
            hex_radius = np.nan

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
        if np.nonzero(np.isnan(row))[0].shape[0] > 0:
            row = np.delete(row, np.nonzero(np.isnan(row))[0])
            
        # get the minimum value in the row
        #print(row)
        if row.shape[0]>0:
            nearest_dist.append(row.min())

    mean_nearest = np.nanmean(np.array(nearest_dist))
    std_nearest = np.nanstd(np.array(nearest_dist))

    hist, bin_edge = calc.distHist(dist, bin_width, offset_bin)

    annulus_area = calc.annulusArea(bin_edge)

    if dist_area_norm:
        # normalize cone counts in each bin by the area of each annulus from which cones were counted
        for ind, bin in enumerate(hist):
            hist[ind] = bin/annulus_area[ind]

    return dist, mean_nearest, std_nearest, hist, bin_edge, annulus_area


def getDataForMosaicSetProcess(sav_fl, proc_metrics):
    #print(sav_fl)
    # print('hi im getting data for degree structure')
    data = []
    metrics_taken = []
    metric_key = []
    mos_key = []

    mos_types = ['monteCarlo_uniform',
                 'monteCarlo_coneLocked',
                 'measured',
                 'coneLocked_maxSpacing',
                 'hexgrid_by_density',
                 'dmin']

    roll_call = np.zeros([len(proc_metrics), len(mos_types)])

    with h5py.File(sav_fl, 'r') as file:
        file_keys = list(file.keys())

        for ind1, m in enumerate(proc_metrics):
            metric_type = m[0]
            metric_name = m[1]
            # print(metric_name)
            for ind2, mos_type in enumerate(mos_types):
                key = mos_type + '_' + metric_type
                # print('key 1')
                # print(key)
                # print('keys in file')
                # print(list(file.keys()))
                if key in list(file.keys()):
                    # print("key 2")
                    if metric_name in list(file[key].keys()):
                        roll_call[ind1, ind2] = 1
                    else:
                        print('key "' + metric_name + '" not found in key "' + metric_type + '" not found for mosaic type "' + mos_types[ind2] + '"' )
                else:
                    print('key "' + metric_type + '" not found for mosaic type "' + mos_types[ind2] + '"' )
                # print('')
            
            if np.sum(roll_call[ind1,:]) == len(mos_types):
                metrics_taken.append(metric_name)
                # print('')
                # print(metric_name)
                for mos_type in mos_types:
                    # print('')
                    # print(mos_type)
                    data.append(file[mos_type + '_' + metric_type][metric_name][()])
                    metric_key.append(metric_name)
                    mos_key.append(mos_type)
                    # if metric_name == 'voronoi_area':
                    #     print(mos_type)
                    #     print(data[len(data)-1])
                    #     print(metric_key[len(data)-1])
                    #     print(mos_key[len(data)-1])
                    #     print('')
                    #     print('')
                    #     print('')

        # print(roll_call)
        # print('')

    return [data, metrics_taken, metric_key, mos_key]


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
            elif key == 'dmin':
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
            print('could not pull mean icd from ' + all_coord_fl)
        bin_width = all_cone_mean_icd
        offset_bin = False
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
            poisson_intervals = np.empty([2, max_hist_bin])
            poisson_intervals[:] = np.nan

            # this is just to convert the returned histograms into a rectangular array
            # (this can't be done in advance because of...slight variability in the number of bins returned? why?)
            hist_mat = np.zeros([num_mosaic, max_hist_bin])
            for mos in np.arange(0, num_mosaic):
                hist_mat[mos, 0:hist[mos].shape[0]] = hist[mos]

            hist = hist_mat

            if num_mosaic > 1:
                for b in np.arange(0, hist_mat.shape[1]):
                    poisson_intervals[0, b], poisson_intervals[1, b] = calc.poisson_interval(np.nanmean(hist_mat[:,b]))

            # print('POISSON BIZ')
            # print(poisson_intervals)

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


def dmin_process(param, sav_cfg):
    proc = 'dmin'
    proc_vars = sav_cfg['dmin']['variables']

    # First load in the inputs that are needed for this function
    sav_fl = param['sav_fl']
    num2gen = util.numSim(proc, param['num_sim'], param['sim_to_gen'])

    with h5py.File(sav_fl, 'r') as file:
        # get coordinates and # of cones
        og_coord = file['input_data']['cone_coord'][()]
        
    if og_coord.shape[0] > 1:
        num_sim = param['num_sim']
        sim_to_gen = param['sim_to_gen']

        # look for all cone mosaic for this data
        save_path = os.path.dirname(sav_fl)
        all_coord_fl = save_path + '\\' + param['mosaic'] + '_all.hdf5'
        try:
            with h5py.File(all_coord_fl, 'r') as file:
                all_coord = file['input_data']['cone_coord'][()]
                all_cone_mean_icd   = file['measured_voronoi']['icd_mean'][()]
        except:
            print('could not pull mean nearest from ' + all_coord_fl)
      

        # number of mosaics to generate
        num2gen = util.numSim('dmin', num_sim, sim_to_gen)
        
        # cones per mosaic to generate
        cones2place = og_coord.shape[0]

        #get input sfor the dmin model
        prob_rej_type = param['dmin_probability_func']
        dmin_maxdist = np.nan
        IND_entry = np.nan
        intercept = np.nan
        coef = np.nan
        if prob_rej_type == 'IND_shift_basic_logistic':
            IND_entry = all_cone_mean_icd
        elif prob_rej_type == 'custom_logistic':
            intercept = param['dmin_func_intercept']
            coef = param['dmin_func_coef']
        elif prob_rej_type == 'inverse_distance_squared':
            dmin_maxdist = all_cone_mean_icd*2  
        elif prob_rej_type == 'all_or_none':
            dmin_maxdist = all_cone_mean_icd*2 
 
        dmin_coord = calc.dmin(all_coord, num2gen, cones2place, dmin_maxdist, prob_rej_type, IND_entry, intercept, coef)
        
        coord = dmin_coord
        
        data_to_set = util.mapStringToLocal(proc_vars, locals())
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)

        
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
            #print(' creating hex radius by rectangluar density')
            hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]

        elif sim_hexgrid_by == 'voronoi':
            #print('creating hex radius by voronoi density')
            hex_radius = file['measured_voronoi']['hex_radius'][()]
        else:
            print('bad input for sim_hexgrid_by, needs to be rectangular or voronoi')
            print(sim_hexgrid_by)
    num_cones_final = []
    hex_radii_used = []
    if num_cones > 1:  # otherwise why bother
        
        temp = []

        maxipad = 0
        for m in np.arange(0,num2gen):
            num_cones_final.append([])
            hex_radii_used.append([])
            t, num_cones_final[m], hex_radii_used[m] = calc.hexgrid(1,
                                                                hex_radius,
                                                                [0, img_x],
                                                                [0, img_y],
                                                                randomize=True,
                                                                target_num_cones=num_cones
                                                                )
            temp.append(t)
            maxipad = np.amax([maxipad, temp[m].shape[1]])

        coord = np.empty([num2gen,maxipad,2])
        coord[:] = np.nan

        for m in np.arange(0,num2gen):
            coord[m, 0:temp[m].shape[1], :] = temp[m]

        num_cones_final = np.array(num_cones_final)
        hex_radii_used = np.array(hex_radii_used)

        # ax = show.plotKwargs({},'')
        # hist, bins = np.histogram(num_cones_final)
        # plt.stairs(hist, bins)
        # plt.title('number of cones set')

        # ax = show.plotKwargs({},'')
        # hist, bins = np.histogram(hex_radii_used)
        # plt.stairs(hist, bins)
        # plt.title('hex_radii_used')
        
        #temp = np.empty([num2gen, num_cones, 2])
        # for m in np.arange(0,num2gen):
        #     # print(['HEX COORD ' + str(m)])
        #     # print(coord[m, :, :])
        #     # print(np.nonzero([np.isnan(coord[m,a,0]) for a in np.arange(0,coord.shape[1])])[0].shape[0])
        #     non_nan = np.array(np.nonzero(~np.isnan(coord[m,:,0]))[0], dtype=int)
        #     # ax = show.plotKwargs({},'')
        #     # before = show.scatt(np.squeeze(coord[m, :, :]), id + ' ' + str(m) + ' before', ax=ax)
        #     #temp[m,:,:] = util.trim_random_edge_points(coord[m, non_nan, :], num_cones, [0, img_x], [0, img_y])
        #     # ax1 = show.plotKwargs({},'')
        #     # after = show.scatt(np.squeeze(coord[m, :, :]), id + ' ' + str(m) + ' after', ax=ax1)

        data_to_set = util.mapStringToLocal(proc_vars, locals())
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def coneLocked_maxSpacing_process(param, sav_cfg):
    proc = 'coneLocked_maxSpacing'
    proc_vars = sav_cfg[proc]['variables']
    sav_fl = param['sav_fl']    
    with h5py.File(sav_fl, 'r') as file:
        og_coord = file['input_data']['cone_coord'][()]
        img_x = file['input_data']['img_x'][()]
        img_y = file['input_data']['img_y'][()]
        sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
        if sim_hexgrid_by == 'rectangular':
            hex_radius = file['input_data']['hex_radius_for_this_density'][()]
        elif sim_hexgrid_by == 'voronoi' :
            hex_radius = file['measured_voronoi']['hex_radius'][()]

    if og_coord.shape[0] > 1 and not np.isnan(hex_radius):
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
        spaced_coord, num_cones_final, hex_radii_used = calc.coneLocked_hexgrid_mask(all_coord, num2gen, cones2place, x_dim, y_dim, hex_radius)
        
        # trim excess cones
        temp = np.empty([num2gen, cones2place, 2])
        # ax = show.plotKwargs({},'')
        for mos in np.arange(0, num2gen):
            # print('num hexgrid pre trim')
            # print(spaced_coord[mos,:,:].shape)
            # print('num unique hexgrid')
            # print(np.unique(np.squeeze(spaced_coord[mos,:,:]), axis=0).shape)
            non_nan = np.array(np.nonzero(~np.isnan(spaced_coord[mos,:,0]))[0], dtype=int)
            # ax0 = show.scatt(all_coord, 'trim test: before', plot_col = 'y')
            # ax0 = show.scatt(spaced_coord[mos,:,:], 'trim test: before', plot_col = 'r', ax=ax0)
            beep = np.unique(np.squeeze(spaced_coord[mos, non_nan, :]), axis=0)
            # print('num unique coords')
            # print(beep.shape)
            #temp[mos, :, :] = util.trim_random_edge_points(spaced_coord[mos, non_nan, :], cones2place, x_dim, y_dim)
            # ax2 = show.scatt(all_coord, '', plot_col='y')
            # ax2 = show.scatt(temp[mos, :, :], 'trim test: after',plot_col='r', ax=ax2)
            # col = util.randCol()
            # #print(temp[mos,:,:].shape)
            # ax = show.scatt(temp[mos,:,:], 'conelocked spacified overlay', plot_col = col, ax=ax)
            
        coord = spaced_coord 
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
    param['dmin_probability_func'] = user_param['dmin_probability_func'][0]
    param['dmin_func_intercept'] = user_param['dmin_func_intercept'][0]
    param['dmin_func_coef'] = user_param['dmin_func_coef'][0]

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
    # 
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
        #print(key)
        if sav_cfg[key]['process_type'] == 'default':
            mand.append(key)

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


def run_analysis_tiers_process(user_param, sav_cfg):
    processes = user_param['processes']
    tiers = getAnalysisTiers(sav_cfg)
    for proc in tiers:
        if proc in user_param['analyses_to_run'][0]:
            for ind in processes[proc[0]]:
                print('Running process "' + proc[0] + '" on file' + str(ind+1) + '/' + str(len(processes[proc[0]])) +'...') 
                param = unpackThisParam(user_param, ind)
                globals()[sav_cfg[proc[0]]['process']](param, sav_cfg)


# def primary_analyses_process(user_param, sav_cfg):
#     """
#     Inputs
#     ------

#     Outputs
#     -------

#     """
#     processes = user_param['processes']
    
#     tiers = getAnalysisTiers(sav_cfg)
    
#     # perform on data
#     for proc in tiers[0]:
#         if proc in user_param['analyses_to_run'][0]:
#             for ind in processes[proc]:
#                 print('Running process "' + proc + '" on file' + str(ind+1) + '/' + str(len(processes[proc])) +'...') 
#                 param = unpackThisParam(user_param, ind)
#                 globals()[sav_cfg[proc]['process']](param, sav_cfg)
#                 #print("     SIMULATED COORDINATES")
#                 # for sim in user_param['sim_to_gen'][0]:
#                 #     if sim == 'monteCarlo_uniform' or sim == 'monteCarlo_coneLocked':
#                 #         numsim = user_param['num_mc']
#                 #     elif sim == 'coneLocked_spacify_by_nearest_neighbors' or sim == 'hexgrid_by_density':
#                 #         numsim = user_param['num_sp']
#                 #     else:
#                 #         print('Error: invalid simulation entry')
#                     # print('     Running process "' + proc + '" on' + str(numsim) + ' ' + sim + 'simulations...') 
#                     # globals()[sav_cfg[sim]['process']](param, sav_cfg)
#         else:
#             print('didnt run ' + proc)


# def secondary_analyses_process(user_param, sav_cfg):
#     """
#     Inputs
#     ------

#     Outputs
#     -------
#     """
#     processes = user_param['processes']
#     tiers = getAnalysisTiers(sav_cfg)

#     # perform on data
#     if len(tiers) >= 2:
#         for proc in tiers[1]:
#             if proc in user_param['analyses_to_run'][0]:
#                 print('Running process "' + proc + '" on ' + str(len(processes[proc])) + ' mosaic coordinate files...') 
#                 for ind in processes[proc]:
#                     param = unpackThisParam(user_param, ind)
#                     globals()[sav_cfg[proc]['process']](param, sav_cfg)


