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


def mosaic_set_View2PCmetricHistogram(metric, save_name, save_things=False, save_path=''):
    print(metric)
    for fl in save_name:
        print(fl)
        # get spacified coordinate data and plotting parameters from the save file
        values = []
        labels = []
        colors = []
        minval = 0
        maxval = 0
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
           
            if 'monteCarlo_uniform_metrics_of_2PC' in file:
                values.append(file['monteCarlo_uniform_metrics_of_2PC'][metric][()])
                labels.append('monteCarlo_uniform')
                colors.append('rebeccapurple')
                print(values)

            if 'monteCarlo_coneLocked_metrics_of_2PC' in file:
                values.append(file['monteCarlo_coneLocked_metrics_of_2PC'][metric][()])
                labels.append('monteCarlo_coneLocked')
                colors.append('royalblue')
                minval = np.amin([minval, np.amin(values[len(values)-1])])
                maxval = np.amax([maxval, np.amax(values[len(values)-1])])
                print(values)

            if 'coneLocked_maxSpacing_metrics_of_2PC' in file:
                values.append(file['coneLocked_maxSpacing_metrics_of_2PC'][metric][()])
                labels.append('coneLocked_maxSpacing')
                colors.append('darkorange')
                print(values)

            if 'hexgrid_by_density_metrics_of_2PC' in file:
                values.append(file['hexgrid_by_density_metrics_of_2PC'][metric][()])
                labels.append('hexgrid_by_density')
                colors.append('firebrick')
                print(values)

            if 'measured_metrics_of_2PC' in file:
                values.append(file['measured_metrics_of_2PC'][metric][()])
                labels.append('measured')
                colors.append('white')
                print(values)

        ax = plotKwargs({'figsize':10}, '')

        bin_width = int(5)

        for ind, mos_type in enumerate(labels):
            print(mos_type)
            with h5py.File(fl, 'r') as file:  # context manager
                metric_data = file[mos_type + '_metrics_of_2PC'][metric][()]
                print(metric_data)
                if mos_type == 'measured':
                    ax.scatter(metric_data, 0, 100, c='w')
                else:
                    counts, bins = np.histogram(metric_data, bins=bin_width)
                    ax.stairs(counts, bins, color=colors[ind])
                

        # if metric_std > 1:
        #     plt.xlim([metric_mean - (4 * metric_std), metric_mean + (4 * metric_std)])
        # else:
        #     plt.xlim([metric_mean - 5, metric_mean + 5])
        ax.set_facecolor('k')
        plt.xlabel(metric)
        plt.ylabel('count per bin')
        ax.set_title(mosaic + ', conetype: ' + conetype + ', bin width: ' + str(bin_width))
        ax.figure

        if save_things:
            savnm = save_path + mosaic + '_' + conetype + '.png'
            plt.savefig(savnm)


def mosaic_set_ViewVoronoiHistogram(metric, save_name, save_things=False, save_path=''):
    print(metric)
    for fl in save_name:
        print(fl)
        # get spacified coordinate data and plotting parameters from the save file
        values = []
        labels = []
        colors = []
        minval = 0
        maxval = 0
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
           
            if 'monteCarlo_uniform_voronoi' in file:
                values.append(file['monteCarlo_uniform_voronoi'][metric][()])
                labels.append('monteCarlo_uniform')
                colors.append('rebeccapurple')
                minval = np.amin([minval, np.amin(values[len(values)-1])])
                maxval = np.amax([maxval, np.amax(values[len(values)-1])])
                print(values[len(values)-1].shape)

            if 'monteCarlo_coneLocked_voronoi' in file:
                values.append(file['monteCarlo_coneLocked_voronoi'][metric][()])
                labels.append('monteCarlo_coneLocked')
                colors.append('royalblue')
                minval = np.amin([minval, np.amin(values[len(values)-1])])
                maxval = np.amax([maxval, np.amax(values[len(values)-1])])
                print(values[len(values)-1].shape)

            if 'coneLocked_maxSpacing_voronoi' in file:
                values.append(file['coneLocked_maxSpacing_voronoi'][metric][()])
                labels.append('coneLocked_maxSpacing')
                colors.append('darkorange')
                minval = np.amin([minval, np.amin(values[len(values)-1])])
                maxval = np.amax([maxval, np.amax(values[len(values)-1])])
                print(values[len(values)-1].shape)

            if 'hexgrid_by_density_voronoi' in file:
                values.append(file['hexgrid_by_density_voronoi'][metric][()])
                labels.append('hexgrid_by_density')
                colors.append('firebrick')
                minval = np.amin([minval, np.amin(values[len(values)-1])])
                maxval = np.amax([maxval, np.amax(values[len(values)-1])])
                print(values[len(values)-1].shape)

            if 'measured_voronoi' in file:
                values.append(file['measured_voronoi'][metric][()])
                labels.append('measured')
                colors.append('white')
                minval = np.amin([minval, np.amin(values[len(values)-1])])
                maxval = np.amax([maxval, np.amax(values[len(values)-1])])
                print(values[len(values)-1].shape)
        
        if metric == 'voronoi_area':
            bin_width = 15
        if metric == 'num_neighbor':
            bin_width = np.arange(0, 10)
        if metric == 'icd':
            bin_width = 10

        ax = plotKwargs({'figsize':10}, '')

        for ind, mos_type in enumerate(labels):
            with h5py.File(fl, 'r') as file:  # context manager
                if metric == 'icd':
                    bound = file[mos_type +'_voronoi']['bound_cones'][()]
                else:
                    bound = file[mos_type +'_voronoi']['bound_regions'][()]
                metric_data = file[mos_type +'_voronoi'][metric][()]
                metric_mean = file[mos_type+'_voronoi'][metric+'_mean'][()]
                metric_std = file[mos_type+'_voronoi'][metric+'_std'][()]
                metric_regularity = file[mos_type+'_voronoi'][metric+'_regularity'][()]

            for m in np.arange(0, metric_data.shape[0]):
                if metric == 'icd':
                    temp = np.reshape(metric_data[m][np.nonzero(bound[m]),:], [metric_data[m][np.nonzero(bound[m]),:].size,])
                    temp = temp[np.nonzero([not x for x in np.isnan(temp)])[0]]
                    counts, bins = np.histogram(temp, bins=bin_width)
                else:
                    counts, bins = np.histogram(metric_data[m][np.nonzero(bound[m])], bins=bin_width)
                
                ax.stairs(counts, bins, color=colors[ind])

        # if metric_std > 1:
        #     plt.xlim([metric_mean - (4 * metric_std), metric_mean + (4 * metric_std)])
        # else:
        #     plt.xlim([metric_mean - 5, metric_mean + 5])
        ax.set_facecolor('k')
        plt.xlabel(metric)
        plt.ylabel('count per bin')
        ax.set_title(mosaic + ', conetype: ' + conetype + ', bin width: ' + str(bin_width))
        ax.figure

        if save_things:
            savnm = save_path + mosaic + '_' + conetype + '.png'
            plt.savefig(savnm)


def view2PCmetric(mos_type, save_name, z_dim = 0, scale_std=2, showNearestCone=False, save_things=False, save_path='', save_type='.png'):
    for fl in save_name:
        print(fl)

        with h5py.File(fl, 'r') as file: 
            analysis_x_cutoff = file[mos_type + '_' + 'metrics_of_2PC']['analysis_x_cutoff'][()]
            corred = file[mos_type + '_' + 'two_point_correlation']['corred'][()][z_dim, 0:analysis_x_cutoff]
            corr_by_mean = file[mos_type + '_' + 'metrics_of_2PC']['corr_by_mean'][()]
            corr_by_std = file[mos_type + '_' + 'metrics_of_2PC']['corr_by_std'][()]
            mean_corr = file[mos_type + '_' + 'metrics_of_2PC']['mean_corr'][()]
            std_corr = file[mos_type + '_' + 'metrics_of_2PC']['std_corr'][()]
            dearth_bins = file[mos_type + '_' + 'metrics_of_2PC']['dearth_bins'][()]
            peak_bins = file[mos_type + '_' + 'metrics_of_2PC']['peak_bins'][()]
            exclusion_bins = file[mos_type + '_' + 'metrics_of_2PC']['exclusion_bins'][()]
            exclusion_radius = file[mos_type + '_' + 'metrics_of_2PC']['exclusion_radius'][()]
            exclusion_area = file[mos_type + '_' + 'metrics_of_2PC']['exclusion_area'][()]

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
            
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            bin_width = file['input_data']['bin_width'][()]
            if bin_width == -1:
                save_path = os.path.dirname(fl)
                all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5'
                try:
                    with h5py.File(all_coord_fl, 'r') as file2:
                        all_cone_mean_icd   = file2['measured_voronoi']['icd_mean'][()]
                except:
                    print('could not pull mean nearest from ' + all_coord_fl)

                bin_width = all_cone_mean_icd

    ax = plotKwargs({'figsize':10}, '')

    bins = bin_edge[0:analysis_x_cutoff+1]
    c = 'y'
    plt.boxplot(corr_by_corr, positions=bin_edge[1:analysis_x_cutoff+1]-(bin_width/2),
                notch=True,
                boxprops=dict({'color': c}),
                capprops=dict({'color': c}),
                whiskerprops=dict({'color': c}),
                flierprops=dict({'color': c}),
                )

    #plt.violinplot(corr_by_corr, bin_edge[1:analysis_x_cutoff+1]-(bin_width/2), showmeans=True)

    corr_by_x, corr_by_y, corr_by_y_plus, corr_by_y_minus = util.reformat_stat_hists_for_plot(bins, corr_by_mean, corr_by_std*2)
    ax = line(corr_by_x, corr_by_y, '', ax=ax, plot_col = 'firebrick')
    ax.fill_between(corr_by_x, corr_by_y_plus, corr_by_y_minus, color='firebrick', alpha=.7)

    if len(corred.shape) == 1:
        corr = corred[0:analysis_x_cutoff]
        runs = 1
        bin_dim = 0
        hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bins, corr, np.zeros(corr.shape[0],))
        ax = line(hist_x, hist_y, '', ax=ax, plot_col='w')
        # ax.fill_between(hist_x, hist_y_plus, hist_y_minus, color='royalblue', alpha=.7)

        # plt.stairs(corred -.5, bins, color='r')

        if (exclusion_bins > 0):
            for b, ind in enumerate(np.arange(0, exclusion_bins)):

                ax.fill_between(bin_edge[b:b+2],
                                [corr[b], corr[b]],
                                [corr_by_mean[b] - (2 * corr_by_std[b]), corr_by_mean[b] - (2 * corr_by_std[b])], 
                                color='g', alpha=.5)
    else:
        runs = corred.shape[0]
        bin_dim = 1

        for m in np.arange(0, corred.shape[0]):
            corr = corred[M, 0:analysis_x_cutoff]
            hist_x, hist_y, hist_y_plus, hist_y_minus = util.reformat_stat_hists_for_plot(bins, corr, np.zeros(corr.shape[0],))
            ax = line(hist_x, hist_y, '', ax=ax, plot_col='w')

            if (exclusion_bins > 0):
                for b, ind in enumerate(np.arange(0, exclusion_bins)):

                    ax.fill_between(bin_edge[b:b+2],
                                    [corr[b], corr[b]],
                                    [corr_by_mean[b] - (2 * corr_by_std[b]), corr_by_mean[b] - (2 * corr_by_std[b])], 
                                    color='g', alpha=.5)
            ax.fill_between(hist_x, hist_y_plus, hist_y_minus, color='royalblue', alpha=.7)

    # if dearth_bins.shape[0] > 0:
    #     ax.scatter(bin_edge[dearth_bins] + bin_width/2, mean_corr[dearth_bins], color='g')
    # if peak_bins.shape[0] > 0:
    #     ax.scatter(bin_edge[peak_bins] + bin_width/2, mean_corr[peak_bins], color='y')


    title = ['bin width: ' + str(bin_width) + ', excl rad: ' + str(exclusion_radius) + ', excl area: ' + str(exclusion_area)]
    print(title)
    ax.set_title(title)
    ax.set_xticks(bin_edge[0:analysis_x_cutoff])
    ax.set_ylim([-1.5, 4])

    # for b in np.arange(0, analysis_x_cutoff):
    #     ax = plotKwargs({'figsize':10}, '')
    #     binhist, binhistedge = np.histogram(corr_by_corr[:,b])
    #     plt.stairs(binhist, binhistedge)
    #     title = ['bin ' + str(b)]


def view2PC(mos_type, save_name, scale_std=2, showNearestCone=False, save_things=False, save_path='', save_type='.png'):
    for fl in save_name:
        print(fl)
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            mosaic = bytes(file['mosaic_meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['mosaic_meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            bin_width = file['input_data']['bin_width'][()]
            bin_edge = file[mos_type + '_' + 'two_point_correlation']['max_bin_edges'][()]
            corred = file[mos_type + '_' + 'two_point_correlation']['corred'][()]
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
                save_path = os.path.dirname(fl)
                all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5'
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

                ax = line([hex_radius, hex_radius], [-1 * lin_extent, lin_extent], id='hex_radius', ax=ax, plot_col='firebrick', linewidth=3)
                ax = line(hist_x, hist_y, '', plot_col=plot_col, ax=ax)
                ax.fill_between(hist_x, 
                                hist_y_plus, 
                                hist_y_minus, color=plot_col, alpha=.7)
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

            ax.set_ylim([np.nanmin(hist_y_minus) - (np.nanmax(hist_y)/5), 
                           np.nanmax(hist_y_plus) + (np.nanmax(hist_y)/5)])

            ax.figure
            ax.legend()
            
            if save_things:
                savnm = save_path + '\\' + id_str + save_type
                plt.savefig(savnm)
            
            if len(save_name) == 1:
                return(ax)

            
def viewIntraconeDist(mos_type, save_things=False, save_name=[], prefix='',
            save_path='', id='', z_dim=0, scale_std=2,
            mosaic_data=True, marker='.', label=None, **kwargs):
            
    for fl in save_name:
        print(fl)
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
    print(metric)
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
            print(metric_data.shape)
            metric_mean = file[mos_type+'_voronoi'][metric+'_mean'][()][z_dim]
            metric_std = file[mos_type+'_voronoi'][metric+'_std'][()][z_dim]
            metric_regularity = file[mos_type+'_voronoi'][metric+'_regularity'][()][z_dim]

        #print(metric_data[np.nonzero(bound_regions)])
        # print('coord_unit: ' + coord_unit)
        # print(metric + ' mean: ' + str(metric_mean))
        # print(metric + ' std: ' + str(metric_std))
        # print(metric + ' regularity: ' + str(metric_regularity))
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

    # print('num bound cells: ' + str(sum(bound_regions[z_dim])))
    # print('total voronoi area: ' + str(sum(voronoi_area[z_dim][np.nonzero(bound_regions[z_dim])])) + density_unit + '^2')
    # print('voronoi density: ' + str(density) + ' points per ' + density_unit + '^2')
    # print('hex radius calc from voronoi: ' + str(hex_radius) + ' ' + coord_unit)

    if save_things:
        savnm = save_path + mosaic + '_bound_cells_' + str(z_dim) + '_' + conetype + '.png'
        plt.savefig(savnm)


def viewMosaic(mos_type, save_things=False, save_name=[], prefix='',
            save_path='',save_type='.png', id='', z_dim=0, plot_col='w',
            mosaic_data=True, marker='.', s=30, label=None, **kwargs):

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
            if len(coord.shape) == 3:
                num_mos = coord.shape[0]
                num_cone = coord.shape[1]
            elif len(coord.shape) == 2:
                num_mos = 1
                num_cone = coord.shape[0]
            print('num mosaics: ' + str(num_mos))
            print('num points per mosaic: ' + str(num_cone))

            for mos in [z_dim]:
                id_str = mos_type + '_(' + str(mos+1) + '//' + str(num_mos) + ')_' + mosaic + '_(' + str(num_cone) + ' cones)'
                xlab = coord_unit
                ylab = coord_unit
                if len(coord.shape) == 3:
                    this_coord = np.zeros([num_cone, 2])
                    this_coord[:, :] = coord[mos, :, :]
                else: 
                    this_coord = coord

                ax = scatt(this_coord, id_str, s=s, plot_col=plot_col, xlabel=xlab, ylabel=ylab, mosaic_data= mosaic_data, z_dim=z_dim, marker=marker, label=label)

                ax.figure

                if save_things:
                    savnm = save_path + mosaic + '_' + str(mos) + '_' + conetype + save_type
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
