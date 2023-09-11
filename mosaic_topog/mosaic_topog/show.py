import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import random
from scipy.spatial import voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import mosaic_topog.utilities as util

# --------------------------data viewing and saving functions--------------------------

## --------------------------------SMP VIEWING FUNCTIONS--------------------------------------



def arcminToPix(data, degrees = False):  #degrees is actually density
    # ack add false condition
    CF_arcmin = .0026*60
    converted_data = data / CF_arcmin
    return converted_data


def pixToArcmin(data, degrees = False): #degrees is actually density
    CF_arcmin = .0026*60
    if not degrees:
        converted_data = data * CF_arcmin
    else:
        converted_data = data * (1/np.power((CF_arcmin/60),2))
    return converted_data
        

def mosaic_set_viewDegreeStructure(metric, save_name, save_things=False, save_path=''):
    # print('check the thing')
    # print(metric)
    for fl in save_name:
        with h5py.File(fl, 'r') as file: 
            metric_hists = file['mosaic_set_' + metric[1] + '_degree_structure']['metric_hists'][()]
            bin_edge = file['mosaic_set_' + metric[1] + '_degree_structure']['bin_edge'][()]
            degree_structure = file['mosaic_set_' + metric[1] + '_degree_structure']['degree_structure'][()]
            percentiles_Rr = file['mosaic_set_' + metric[1] + '_degree_structure']['percentiles_Rr'][()]
            percentiles_Sr = file['mosaic_set_' + metric[1] + '_degree_structure']['percentiles_Sr'][()]
            Rr_data = file['mosaic_set_' + metric[1] + '_degree_structure']['Rr_data'][()]
            Sr_data = file['mosaic_set_' + metric[1] + '_degree_structure']['Sr_data'][()]
            M_data = file['mosaic_set_' + metric[1] + '_degree_structure']['M_data'][()]
            Dm_data = file['mosaic_set_' + metric[1] + '_degree_structure']['Dm_data'][()]
            
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode('utf8')

            bin_width = bin_edge[1] - bin_edge[0]

            metric_colors = ['rebeccapurple', 'dodgerblue', 'white', 'darkorange', 'firebrick', 'chartreuse']

            ax = plotKwargs({'figsize':[8,3.5]}, '')

            # print(bin_edge)

            constrained_inds = [1,2,3,5]
            for ind in constrained_inds:
                hist = metric_hists[ind,:]
                if np.nonzero(~np.isnan(hist))[0].shape[0] > 1:
                    hist_x, hist_y, blah, bloo = util.reformat_stat_hists_for_plot(bin_edge, hist, np.zeros([hist.shape[0],]))
                    ax = line(hist_x, hist_y, mosaic + ' ' + metric[1], plot_col = metric_colors[ind], bckg_col = 'w', linewidth = 2, ax = ax)
                else:
                    plt.scatter(hist[0], 0, 50, 'k', 'o', zorder=6)
            
            quant_lines_Rr = [percentiles_Rr[24], percentiles_Rr[974]]
            quant_lines_Sr = [percentiles_Sr[24], percentiles_Sr[974]]
            

            # ax = line(np.array([quant_lines_Rr[0], 
            #                     quant_lines_Rr[0]]), 
            #           np.array([-.05, .05]), 
            #           '', 
            #           ax=ax, 
            #           plot_col=metric_colors[1],
            #           linewidth=2
            #           ) 
            
            # ax = line(np.array([quant_lines_Rr[1], 
            #                     quant_lines_Rr[1]]), 
            #           np.array([-.05, .05]), 
            #           '', 
            #           ax=ax, 
            #           plot_col=metric_colors[1],
            #           linewidth=2
            #           ) 
                    
            # ax = line(np.array([quant_lines_Sr[0], 
            #                     quant_lines_Sr[0]]), 
            #           np.array([-.05, .05]), 
            #           '', 
            #           ax=ax, 
            #           plot_col=metric_colors[3],
            #           linewidth=2
            #           ) 
            
            # ax = line(np.array([quant_lines_Sr[1], 
            #                     quant_lines_Sr[1]]), 
            #           np.array([-.05, .05]), 
            #           '', 
            #           ax=ax, 
            #           plot_col=metric_colors[3],
            #           linewidth=2
            #           ) 

            ax.set_facecolor('w')
            # minval = np.nanmin(metric_hists[1,:])
            # maxval = np.nanmax(metric_hists[3,:])
            # print(minval)
            # print(maxval)
            # print('')
            plt.xlabel(metric)
            plt.ylabel('count per bin / N cones in mosaic type population')
            ax.set_title(mosaic + ' , set, ' + metric[1] + ' degree structure: ' + str(degree_structure))
            ax.figure

            allsims = np.concatenate([Rr_data, Sr_data, Dm_data])
            minval = np.nanmin(allsims)
            maxval = np.nanmax(allsims)
            # print('MAXVAL')
            # print(maxval)
            bin_below = np.nonzero(bin_edge <= minval)[0]
            bin_above = np.nonzero(bin_edge >= maxval)[0]
            # print('AS ABOVE SO BELOW')
            # print(bin_above)
            ax.set_xlim([bin_edge[bin_below[0]], bin_edge[bin_above[0]]])

            # print(mosaic)
            # print('')
            if save_things:
                # print('saved it!')
                savnm = save_path + mosaic + '_' + metric[1] + '_degree_structure' + '.png'
                plt.savefig(savnm)

def mosaic_set_ViewHistogram(metric, save_name, save_things=False, save_path=''):
    # print('check the thing')
    # print(metric)
    for fl in save_name:
        with h5py.File(fl, 'r') as file: 
            metric_hists = file['mosaic_set_' + metric[1] + '_degree_structure']['metric_hists'][()]
            bin_edge = file['mosaic_set_' + metric[1] + '_degree_structure']['bin_edge'][()]
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode('utf8')

        metric_colors = ['rebeccapurple', 'dodgerblue', 'white', 'darkorange', 'firebrick', 'chartreuse']

        ax = plotKwargs({'figsize':10}, '')
       
        for ind in np.arange(0, metric_hists.shape[0]):
            hist = metric_hists[ind,:]
            if np.nonzero(~np.isnan(hist))[0].shape[0] > 1:
                hist_x, hist_y, blah, bloo = util.reformat_stat_hists_for_plot(bin_edge, hist, np.zeros([hist.shape[0],]))
                ax = line(hist_x, hist_y, mosaic + ' ' + metric[1], plot_col = metric_colors[ind], bckg_col = 'w', linewidth = 2, ax = ax)
            else:
                plt.scatter(hist[0], 0, 50, 'w', 'o', 'filled')

        ax.set_facecolor('k')
        plt.xlabel(metric)
        plt.ylabel('count per bin')
        ax.set_title(mosaic + ' , set, ' + metric[1])
        ax.figure

        if save_things:
            # print('saved it!')
            savnm = save_path + mosaic + '_' + metric[1] + '_mosaic_set' + '.png'
            plt.savefig(savnm)


def view2PCmetric(mos_type, save_name, z_dim = 0, scale_std=2, showNearestCone=False, save_things=False, save_path='', save_type='.png'):

    for fl in save_name:
        with h5py.File(fl, 'r') as file: 
            analysis_x_cutoff = file[mos_type + '_' + 'metrics_of_2PC']['analysis_x_cutoff'][()]
            corred = file[mos_type + '_' + 'two_point_correlation']['corred'][()][z_dim, 0:analysis_x_cutoff]
            corr_by_mean = file[mos_type + '_' + 'metrics_of_2PC']['corr_by_mean'][()]
            #corr_by_std = file[mos_type + '_' + 'metrics_of_2PC']['corr_by_std'][()]
            corr_by_pi = file[mos_type + '_' + 'metrics_of_2PC']['corr_by_pi'][()]
            mean_corr = file[mos_type + '_' + 'metrics_of_2PC']['mean_corr'][()]
            std_corr = file[mos_type + '_' + 'metrics_of_2PC']['std_corr'][()]
            dearth_bins = file[mos_type + '_' + 'metrics_of_2PC']['dearth_bins'][()][z_dim]
            peak_bins = file[mos_type + '_' + 'metrics_of_2PC']['peak_bins'][()][z_dim]
            exclusion_bins = file[mos_type + '_' + 'metrics_of_2PC']['exclusion_bins'][()][z_dim]
            exclusion_radius = file[mos_type + '_' + 'metrics_of_2PC']['exclusion_radius'][()][z_dim]
            exclusion_factor = file[mos_type + '_' + 'metrics_of_2PC']['exclusion_factor'][()][z_dim]

            corr_by = bytes(file['input_data']['corr_by'][()]).decode("utf8")
            corr_by_corr = file[corr_by + '_' + 'two_point_correlation']['corred'][()][:, 0:analysis_x_cutoff]
            max_bins = file[corr_by + '_' + 'two_point_correlation']['max_bins'][()]
            bin_edge = file[corr_by + '_' + 'two_point_correlation']['max_bin_edges'][()]

            sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
            if sim_hexgrid_by == 'rectangular':
                hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]
            elif sim_hexgrid_by == 'voronoi':
                hex_radius = file['measured_voronoi']['hex_radius'][()]
            else:
                print('ack!!! problem getting hex_radius in metrics_of_2PC_process')
            
            # print(corr_by_mean)


            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            bin_width = file['input_data']['bin_width'][()]
            if bin_width == -1:
                save_path_h = os.path.dirname(fl)
                all_coord_fl = save_path_h + '\\' + mosaic + '_all.hdf5'
                try:
                    with h5py.File(all_coord_fl, 'r') as file2:
                        all_cone_mean_icd   = file2['measured_voronoi']['icd_mean'][()]
                except:
                    print('could not pull mean nearest from ' + all_coord_fl)

                bin_width = all_cone_mean_icd

        ax = plotKwargs({'figsize':(10,4)}, '')

        bins = bin_edge[0:analysis_x_cutoff+1]

        # the big lazy conversion insert
        convert = True
        if convert:
            bins = pixToArcmin(bins)
            bin_edge = pixToArcmin(bin_edge)
            bin_width = pixToArcmin(bin_width)
            exclusion_radius = pixToArcmin(exclusion_radius)

            coord_unit = 'arcmin'

        c = 'y'
        # boxbiz = plt.boxplot(corr_by_corr, positions=bin_edge[1:analysis_x_cutoff+1]-(bin_width/2),
        #             notch=False,
        #             boxprops=dict({'color': c}),
        #             capprops=dict({'color'                        : c}),
        #             whiskerprops=dict({'color': c}),
        #             flierprops=dict({'color': c}),
        #             )

        corr_by_x, corr_by_y, corr_by_y_plus, corr_by_y_minus = util.reformat_stat_hists_for_plot(bins, corr_by_mean, corr_by_pi)
        #ax = line(corr_by_x, corr_by_y, '', ax=ax, plot_col = 'firebrick')

        ax = line([bin_edge[0], exclusion_radius], [-1.1, -1.1], '', bckg_col = 'w', ax=ax, plot_col = 'g', linewidth=4)
        ax.fill_between(corr_by_x, corr_by_y_plus, corr_by_y_minus, color='rebeccapurple', alpha=.7)

        if len(corred.shape) == 1:
            corr = corred[0:analysis_x_cutoff]
            runs = 1
            bin_dim = 0

            hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bins, corr, np.zeros(corr.shape[0],))

            ax = line(hist_x, hist_y, '', ax=ax, plot_col='k', bckg_col = 'w', linewidth=4)
            # ax.fill_between(hist_x, hist_y_plus, hist_y_minus, color='royalblue', alpha=.7)

            # plt.stairs(corred -.5, bins, color='r')

            # if (exclusion_bins > 0):
            #     for b, ind in enumerate(np.arange(0, exclusion_bins)):
            #         if b in dearth_bins:
                        #ax.scatter(bin_edge[b+1]-(bin_width/2), corr[b], 20, 'g')
                    
                    # plt.stairs(corr-.5, bins, color='r')
                    # ax.fill_between(bin_edge[b:b+2],
                    #                 [corr[b], corr[b]],
                    #                 [corr_by_pi[0, b], corr_by_pi[0,b]], 
                    #                 color='g', alpha=.5)
        else:
            runs = corred.shape[0]
            bin_dim = 1

            for m in np.arange(0, corred.shape[0]):
                corr = corred[m, 0:analysis_x_cutoff]
                hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bins, corr, np.zeros(corr.shape[0],))
                ax = line(hist_x, hist_y, '', ax=ax, plot_col='k', bckg_col='w', linewidth=4)
                
                # if (exclusion_bins > 0):
                #     for b, ind in enumerate(np.arange(0, exclusion_bins)):
                #         ax.fill_between(bin_edge[b:b+2],
                #                         [corr[b], corr[b]],
                #                         [corr_by_pi[0,b], corr_by_pi[0,b]], 
                #                         color='g', alpha=.5)
                        
                ax.fill_between(hist_x, hist_y_plus, hist_y_minus, color='royalblue', alpha=.7)

        # if dearth_bins.shape[0] > 0:
        #     ax.scatter(bin_edge[dearth_bins] + bin_width/2, mean_corr[dearth_bins], color='g')
        # if peak_bins.shape[0] > 0:
        #     ax.scatter(bin_edge[peak_bins] + bin_width/2, mean_corr[peak_bins], color='y')


        title = [mosaic + ', bin width: ' + str(np.around(bin_width, decimals=2)) 
                 + ', excl rad: ' + str(np.around(exclusion_radius, decimals =2))
                 + ', excl fact' + str(np.around(exclusion_factor, decimals = 2))]

        ax.set_title(title)
        xtick_float = np.around(bin_edge[0:analysis_x_cutoff],2)
        xtick_str = xtick_float.astype('str')


        ax.set_xticks(xtick_float, xtick_str)
        ax.set_ylim([-1.25, .5 + np.nanmax([1, np.nanmax(corr), np.nanmax(corr_by_pi[1,:])])])

        ax.set_xlabel('radius from individual S cones, ' + coord_unit)
        ax.set_ylabel('correlation')

        ax.set_xlim([-.5, bin_width * 8 + 0.5])
        # for b in np.arange(0, analysis_x_cutoff):
        #     ax = plotKwargs({'figsize':10}, '')
        #     binhist, binhistedge = np.histogram(corr_by_corr[:,b])
        #     plt.stairs(binhist, binhistedge)
        #     title = ['bin ' + str(b)]

        ax.figure
        # ax.set_aspect(1.5)

        if save_things:
            savnm = save_path + mosaic + '_metricsOf2PC' + save_type
            plt.savefig(savnm)


def viewDRP(mos_type, save_name, z_dim=0, scale_std=2, showNearestCone=False, save_things=False, save_path='', save_type='.png'):

    for fl in save_name:

        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            bin_edges = file[mos_type + '_' + 'drp']['bin_edges'][()]
            drp_mean = file[mos_type + '_' + 'drp']['drp_mean'][()]
            drp_all = file[mos_type + '_' + 'drp']['drp_all'][()]
            drp_std = file[mos_type + '_' + 'drp']['drp_std'][()]
            drp_ste = file[mos_type + '_' + 'drp']['drp_ste'][()]
            add_edge_correction = file[mos_type + '_' + 'drp']['add_edge_correction'][()]
            effective_radius = file[mos_type + '_' 'drp']['effective_radius'][()]
            packing_factor = file[mos_type + '_' + 'drp']['packing_factor'][()]
            maximum_radius = file[mos_type + '_' + 'drp']['maximum_radius'][()]
            expected_DRP = file[mos_type + '_' + 'drp']['expected_drp'][()]
        
        convert = True
        if convert:
            
            bin_edges = pixToArcmin(bin_edges)
            effective_radius = pixToArcmin(effective_radius)
            maximum_radius = pixToArcmin(maximum_radius)
            coord_unit = 'arcmin'

            drp_all = pixToArcmin(drp_all, degrees = True)
            drp_mean = pixToArcmin(drp_mean, degrees = True)
            drp_std = pixToArcmin(drp_std, degrees = True)
            drp_ste = pixToArcmin(drp_ste, degrees = True)
            expected_DRP = pixToArcmin(expected_DRP, degrees = True)
            density_unit = 'density (points per degree squared)'
        else:
            coord_unit = 'pixels'
            density_unit = 'density (points per pixel squared)'

        ax = plotKwargs({'bckg_col':'w'}, '')
        bin_width = bin_edges[1] - bin_edges[0]


        # convert y values to density in cones per degree^2 instead of cones per arcmin^2
        y1 = expected_DRP[0,:] 
        plt.stairs(y1, bin_edges, fill=False, linewidth = 4, color = 'gray')
        y2 = np.squeeze(drp_all[z_dim,:]) 
        
        plt.stairs(y2, bin_edges, fill=False, color = 'k', linewidth = 4)
        y3 = [np.nanmax(drp_all[z_dim,0:8]) * 1.05, np.nanmax(drp_all[z_dim,0:8])*1.05]

        plt.plot([0, effective_radius[z_dim]], y3, 'g', linewidth = 4)
        plt.xlabel("Distance from individual cones, " + coord_unit)
        plt.xlim([0, bin_width * 8])
        plt.ylim([0, y3[0] * 1.05])
        if add_edge_correction:
            plt.ylabel(density_unit + "(w/ edge correction)")
        else:
            plt.ylabel(density_unit + "(w/o edge correction, a.u.)")

        plt.title("Density Recovery Profile")

        #ax.figure

        if save_things:
            savnm = save_path + mosaic + '_DRP' + save_type
            plt.savefig(savnm)

def view2PC(mos_type, save_name, z_dim=0, scale_std=2, showNearestCone=False, save_things=False, save_path='', save_type='.png'):
    for fl in save_name:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            bin_width = file['input_data']['bin_width'][()]
            bin_edge = file[mos_type + '_' + 'two_point_correlation']['max_bin_edges'][()]
            if mos_type == 'measured':
                corred = file[mos_type + '_' + 'two_point_correlation']['corred'][()]
            else:
                corred = file[mos_type + '_' + 'two_point_correlation']['corred'][()][z_dim,:]
            # to_be_corr_colors = [bytes(n).decode('utf8') for n in file['input_data']['to_be_corr_colors'][()]]
            # to_be_corr = [bytes(n).decode('utf8') for n in file['input_data']['to_be_corr'][()]]
            sim_hexgrid_by = bytes(file['input_data']['sim_hexgrid_by'][()]).decode("utf8")
            
            if sim_hexgrid_by == 'rectangular':
                hex_radius = file['basic_stats']['hex_radius_of_this_density'][()]

            elif sim_hexgrid_by == 'voronoi':
                hex_radius = file['measured_voronoi']['hex_radius'][()]

            else:
                print('ack!!! problem getting hex_radius in view_2PC')

            if bin_width == -1:
                save_path_h = os.path.dirname(fl)
                all_coord_fl = save_path_h + '\\' + mosaic + '_all.hdf5'
                try:
                    with h5py.File(all_coord_fl, 'r') as file2:
                        all_cone_mean_icd   = file2['measured_voronoi']['icd_mean'][()]
                except:
                    print('could not pull mean nearest from ' + all_coord_fl)

                bin_width = all_cone_mean_icd

            id_str = mosaic + '_' + conetype
            
            maxY = 0

            fig, ax = plt.subplots(1, 1, figsize = [10,10])

            if not np.isnan(corred.all()):
                if len(corred.shape) >= 2:
                    hist_mean = np.nanmean(corred, axis=0)
                    hist_std = np.nanstd(corred, axis=0)

                else: 
                    hist_mean = corred
                    hist_std = np.zeros([corred.shape[0],])

                plot_label = mos_type
                plot_col = conetype_color

                maxY = np.nanmax(hist_mean)

                # set up inputs to plot
                xlab = 'distance, ' + coord_unit
                ylab = 'bin count (binsize = ' + str(bin_width)

                # *** SS fix this to pull the string from inputs
                tit = 'two-point_correlation' # (' + str(num_cone) + " cones, " + str(num_mc) + " MCs)"

                lin_extent = .5

                hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bin_edge, hist_mean, hist_std*2)
        
                ax = line([hex_radius, hex_radius], [-1 * lin_extent, lin_extent], id='hex_radius', ax=ax, plot_col='r', linewidth=3)
                ax = line(hist_x, hist_y, '', plot_col='w', ax=ax)
                #ax.stairs(hist_mean-.5, bin_edge, color='r')
                ax.fill_between(hist_x, 
                                hist_y_plus, 
                                hist_y_minus, color='w', alpha=.7)
                ax.set_label = tit
                ax.set_xlabel = xlab
                ax.set_ylabel = ylab

                # c='y'
                # plt.boxplot(corred, positions=bin_edge[1:bin_edge.shape[0]]-(bin_width/2),
                #     notch=True,
                #     boxprops=dict({'color': c}),
                #     capprops=dict({'color': c}),
                #     whiskerprops=dict({'color': c}),
                #     flierprops=dict({'color': c}),
                # )
            ax.set_xlim([0, bin_width * 8])
            ax.set_ylim([-1.2, 3])
            # ax.set_ylim([np.nanmin(hist_y_minus) - (np.nanmax(hist_y_plus)/10), 
            #             np.nanmax(hist_y_plus) + (np.nanmax(hist_y_plus)/10)])

            #ax.figure
            ax.legend()
            
            ax.figure

            if save_things:
                savnm = save_path + id_str + '_2PC' + save_type

                plt.savefig(savnm)
            
            if len(save_name) == 1:
                return(ax)

            
def viewIntraconeDist(mos_type, save_name, save_things=False, prefix='',
            save_path='', id='', z_dim=0, scale_std=2,
            mosaic_data=True, marker='.', label=None, save_type='.png', **kwargs):
            
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
            poisson_intervals = file[mos_type + '_intracone_dist']['poisson_intervals'][()]

        num_mos = coord.shape[0]
        num_cone = coord.shape[1]
        id_str = mosaic + '_' + conetype


       # the big lazy conversion insert
        convert = True
        if convert:
            bin_edge = pixToArcmin(bin_edge)
            mean_hist = pixToArcmin(mean_hist)
            poisson_intervals = pixToArcmin(poisson_intervals)

            coord_unit = 'arcmin'

        if not np.isnan(mean_hist[0]):
            bin_width = bin_edge[1] - bin_edge[0]
            

            # set up inputs to plot
            x = bin_edge[1:]-(bin_width/2)
            xlab = 'distance, ' + coord_unit
            ylab = 'bin count (binsize = ' + str(bin_width)
            #
            tit = mos_type + ' intracone distance (' + str(num_cone) + " cones, " + str(num_mos) + " mosaics)"
            


            ax = plotKwargs({}, '')

            hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bin_edge, mean_hist, poisson_intervals)
            ax = line(hist_x, hist_y, '', plot_col='k', bckg_col='w', ax=ax, linewidth=2)
            #ax.stairs(hist_mean-.5, bin_edge, color='r')
            ax.fill_between(hist_x, 
                            hist_y_plus, 
                            hist_y_minus, color='k', alpha=.4)
            # ax = shadyStats(x, mean_hist, std_hist, id_str, scale_std=scale_std,
            #                 plot_col=conetype_color, title=tit, xlabel=xlab,
            #                 ylabel=ylab)
            ax.set_xlim([0,bin_width * 8])
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.figure

            if save_things:
                savnm = save_path + id_str + save_type
                plt.savefig(savnm)
                 

def viewVoronoiHistogram(mos_type, metric, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, 
            mosaic_data=True, marker='.', label=None, **kwargs):

    for fl in save_name:

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
        # print('coord_unit: ' + coord_unit)
        # print(metric + ' mean: ' + str(metric_mean))
        # print(metric + ' std: ' + str(metric_std))
        # print(metric + ' regularity: ' + str(metric_regularity))
        ax = getAx(kwargs)
        nonnan_br = np.nonzero(~np.isnan(bound_regions))[0]
        counts, bins = np.histogram(metric_data[np.nonzero(bound_regions[nonnan_br])])
        ax.stairs(counts, bins)
        if metric_std > 1:
            plt.xlim([metric_mean - (2 * metric_std), metric_mean + (2 * metric_std)])
        else:
            plt.xlim([metric_mean - 5, metric_mean + 5])
        plt.xlabel(metric)
        plt.ylabel('count per bin')
        ax.figure

        if save_things:
            savnm = save_path + mosaic + '_' + str(z_dim) + '_' + conetype + save_type
            plt.savefig(savnm)


def viewVoronoiDiagram(mos_type, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, 
            mosaic_data=True, marker='.', label=None, save_type = '.png', xlim = [-1,-1], ylim = [-1,-1], **kwargs):
    for fl in save_name:

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

            if not np.all(xlim == [-1, -1]):
                xlim = [0, file['input_data']['img_x'][()]]

            if not np.all(ylim == [-1, -1]):
                ylim = [0, file['input_data']['img_y'][()]]
            
            regions = file[mos_type+'_voronoi']['regions'][()]
            vertices = file[mos_type+'_voronoi']['vertices'][()]
            num_neighbor = file[mos_type+'_voronoi']['num_neighbor'][()]
            bound_regions = file[mos_type+'_voronoi']['bound_regions'][()]
            voronoi_area = file[mos_type+'_voronoi']['voronoi_area'][()]
            convert_coord_unit = file['input_data']['convert_coord_unit'][()]
            
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            hex_radius = file[mos_type+'_voronoi']['hex_radius'][()]
            density = file[mos_type+'_voronoi']['density'][()]

    if ~np.all(np.isnan(regions)): # type(regions) is np.ndarray or type(regions is np.float64):

        ax = getAx(kwargs)

        for i in range(0, len(regions[z_dim])):
            if bound_regions[z_dim][i]:
                if not np.isnan(num_neighbor[z_dim][i]):
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
                        colour = [50, 50, 255]
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

        # this needs to be thought out better for non-converted case
        scalebar_len = arcminToPix(10)
        # print('SCALEBAR LENGTH PIX -> ARCMIN')
        # print(pixToArcmin(scalebar_len))

        ax = line(np.array([.5, .5 + scalebar_len]), 
                  np.tile(np.nanmax(coord[z_dim, :, : ]) - 2, 2), 
                  '', ax=ax, linewidth = 3,
                  plot_col = 'r', bckg_col = 'w')
        ax = scatt(np.squeeze(coord[z_dim, :, :]),'bound_voronoi_cells', plot_col = 'k', bckg_col = 'w', s = 120, ax=ax, mosaic_data=True)
        
        if not np.all(xlim == [-1, -1]):
            ax.set_xlim(xlim)

        if not np.all(ylim == [-1, -1]):
            ax.set_ylim(ylim)

        ax.figure



        # the big lazy conversion insert
        convert = True
        if convert:
            # xlim = pixToArcmin(np.array(xlim))
            # ylim = pixToArcmin(np.array(ylim))
            xticks = np.array(ax.get_xticks())
            yticks = np.array(ax.get_yticks())

            # print(xticks)
            # print(yticks)
            # print('')
            xtick_labels = pixToArcmin(xticks)
            ytick_labels = pixToArcmin(yticks)

            # print(xtick_labels)
            # print(ytick_labels)
            # print(type(xtick_labels))
            # print('')
            xtick_labels = xtick_labels.astype('str')
            ytick_labels = ytick_labels.astype('str')

            # print(xtick_labels)
            # print(ytick_labels)
            # print(type(xtick_labels))

            # print('')
            # print('')
            # print('')
            ax.set_xticklabels(xtick_labels)
            ax.set_yticklabels(ytick_labels)
            coord_unit = 'arcmin'

        plt.xlabel(coord_unit)
        plt.ylabel(coord_unit)




        # print('num bound cells: ' + str(sum(bound_regions[z_dim])))
        # print('total voronoi area: ' + str(sum(voronoi_area[z_dim][np.nonzero(bound_regions[z_dim])])) + density_unit + '^2')
        # print('voronoi density: ' + str(density) + ' points per ' + density_unit + '^2')
        # print('hex radius calc from voronoi: ' + str(hex_radius) + ' ' + coord_unit)

        if save_things:
            savnm = save_path + mosaic + '_bound_cells_' + str(z_dim) + '_' + conetype + save_type
            plt.savefig(savnm)
    else:
        print('skipping Voronoi diagram for ' + mosaic + ' due to lack of bound cones...')


def viewMosaic(mos_type, save_things=False, save_name=[], prefix='',
            save_path='',save_type='.png', id='', z_dim=0, plot_col='w',
            mosaic_data=True, marker='.', s=30, label=None, xlim = [-1,-1], ylim = [-1,-1], **kwargs):

    for fl in save_name:

        # get spacified coordinate data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")   
            if not np.all(xlim == [-1, -1]):
                xlim = [0, file['input_data']['img_x'][()]]

            if not np.all(ylim == [-1, -1]):
                ylim = [0, file['input_data']['img_y'][()]]

            if mos_type == 'measured':
                coord = file['input_data']['cone_coord'][()]
            else:
                try:
                    coord = file[mos_type]['coord'][()]
                except:
                    raise Exception('bad mosaic type sent to viewMosaic: ' + mos_type)

       

        save_path2 = os.path.dirname(fl)
        all_coord_fl = save_path2 + '\\' + mosaic + '_all.hdf5'
        try:
            with h5py.File(all_coord_fl, 'r') as file:
                all_cone_coord   = file['input_data']['cone_coord'][()]
        except:
            print('could not pull cone coord from ' + all_coord_fl)

        # the big lazy conversion insert
        convert = True
        if convert:
            coord = pixToArcmin(coord)
            all_cone_coord = pixToArcmin(all_cone_coord)
            coord_unit = 'arcmin'
            xlim = pixToArcmin(np.array(xlim))
            ylim = pixToArcmin(np.array(ylim))

        #if not np.isnan(coord[0]).any():
        if len(coord.shape) == 3:
            num_mos = coord.shape[0]
            num_cone = coord.shape[1]
        elif len(coord.shape) == 2:
            num_mos = 1
            num_cone = coord.shape[0]


        for mos in [z_dim]:
            id_str = mos_type + '_(' + str(mos+1) + '//' + str(num_mos) + ')_' + mosaic + '_(' + str(num_cone) + ' cones)'
            xlab = coord_unit
            ylab = coord_unit
            if len(coord.shape) == 3:
                this_coord = np.zeros([num_cone, 2])
                this_coord[:, :] = coord[mos, :, :]
            else: 
                this_coord = coord

            ax = plotKwargs({'figsize':10}, '')

            # this needs to be thought out better for non-converted case
            # ax = line(np.array([.5, 10.5]), 
                    #   np.tile(np.nanmax(this_coord[:, 1])-2, 2), 
                    #   '', ax=ax, linewidth = 5,
                    #   plot_col = 'r', bckg_col = 'w')

            # ax = line(np.array([.5, 10.5]),           
            #           np.tile(np.nanmax(this_coord[:, 1])/2, 2), 
            #           '', ax=ax, linewidth = 5,
            #           plot_col = 'r', bckg_col = 'w')

            crop_window = True
            if crop_window:
                allcone_sz = 100
                scone_sz = 1000
            else:
                allcone_sz = 50
                scone_sz = 500

            ax = scatt(all_cone_coord, id_str, s=allcone_sz, plot_col='k', bckg_col = 'w', ax=ax, xlabel=xlab, ylabel=ylab, mosaic_data= mosaic_data, z_dim=z_dim, marker=marker, label=label)
            # ax = scatt(all_cone_coord, id_str, s=140, plot_col='w', bckg_col = 'w', ax=ax, xlabel=xlab, ylabel=ylab, mosaic_data= mosaic_data, z_dim=z_dim, marker=marker, label=label)
            ax = scatt(this_coord, id_str, s=scone_sz, plot_col='dodgerblue', bckg_col='w', xlabel=xlab, ax=ax, ylabel=ylab, mosaic_data= mosaic_data, z_dim=z_dim, marker=marker, label=label)

            if not np.all(xlim == [-1, -1]):
                ax.set_xlim(xlim)

            if not np.all(ylim == [-1, -1]):
                ax.set_ylim(ylim)

            
            if crop_window:
                # print('CROPPING IT')
                arcmin_w = 25
                arcmin_h = 10
                x_temp = ax.get_xlim()
                y_temp = ax.get_ylim()
                x_mid = x_temp[0] + (x_temp[1] - x_temp[0])/2
                y_mid = y_temp[0] + (y_temp[1] - y_temp[0])/2
                x_set = [x_mid-arcmin_w/2, x_mid+arcmin_w/2]
                y_set = [y_mid-arcmin_h/2, y_mid+arcmin_h/2]
                ax.set_xlim(x_set)
                ax.set_ylim(y_set)
                ax = line(np.array([x_set[0]+.5, x_set[0] + 10.5]),           
                          np.tile(y_set[1] - .5, 2), 
                          '', ax=ax, linewidth = 5,
                          plot_col = 'r', bckg_col = 'w')
            else:
                ax = line(np.array([.5, 10.5]),           
                          np.tile(np.nanmax(this_coord[:, 1])/2, 2), 
                          '', ax=ax, linewidth = 5,
                          plot_col = 'r', bckg_col = 'w')

            ax.figure
            
            if save_things:
                savnm = save_path + '\\' + mosaic + '_' + str(mos) + '_' + conetype + save_type
 
                plt.savefig(savnm)

                

        # else:
        #     print('no coords for for "' + fl + '," skipping')



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
        try: 
            fig, ax = plt.subplots(1, 1, figsize=[kwargs['figsize'],
                                kwargs['figsize']])
        except:
            try:
                fig, ax = plt.subplots(1, 1, figsize=kwargs['figsize'])
            except:
                print('Hey sorry dude I broke plotKwargs trying to get non square big figures. My bad. Ou1')
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


def scatt(coords, id, plot_col='w', bckg_col='k', z_dim=0, mosaic_data=True, marker='.', label=None, s=30, **kwargs):
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
            s=s,
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
    ax.plot(x, mean, color=plot_col, linewidth = 3, label = label)
    ax.fill_between(x, err_low, err_high, color=plot_col, alpha=.7)

    return ax

# CV2

def drawPoint(img, p, colour):
    cv2.circle(img, p, 6, colour, cv2.FILLED, 8, 0)
