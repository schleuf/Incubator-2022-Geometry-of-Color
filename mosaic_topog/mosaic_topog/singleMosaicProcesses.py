from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import mosaic_topog.flsyst as flsyst
import mosaic_topog.calc as calc
import mosaic_topog.show as show
import mosaic_topog.utilities as util


def norm_by_MCU_mean_process(param, sav_cfg):
    proc = 'norm_by_MCU_mean'
    proc_vars = sav_cfg[proc]['variables']

    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        bin_width = file['input_data']['bin_width'][()]
        hist = file['intracone_dist']['hist'][()]
        bin_edge = file['intracone_dist']['bin_edge'][()]
        MCL_mean = file['monteCarlo_coneLocked_intracone_dist']['mean_hist'][()]
        MCL_std = file['monteCarlo_coneLocked_intracone_dist']['std_hist'][()]
        MCU_mean = file['monteCarlo_uniform_intracone_dist']['mean_hist'][()]
        MCU_std = file['monteCarlo_uniform_intracone_dist']['std_hist'][()]
        all_coord = file['monteCarlo_coneLocked']['all_coord'][()]

    if not np.isnan(hist).any():
        max_num_bins = np.max([hist.shape[0], MCL_mean.shape[0], MCU_mean.shape[0]])

        if hist.shape[0] < max_num_bins:
            num_pad = max_num_bins - hist.shape[0]
            hist = np.append(hist, np.zeros(num_pad))

        if MCL_mean.shape[0] < max_num_bins:
            num_pad = max_num_bins - MCL_mean.shape[0]
            MCL_mean = np.append(MCL_mean, np.zeros(num_pad))
            MCL_std = np.append(MCL_std, np.zeros(num_pad))

        if MCU_mean.shape[0] < max_num_bins:
            num_pad = max_num_bins - MCU_mean.shape[0]
            MCU_mean = np.append(MCU_mean, np.zeros(num_pad))
            MCU_std = np.append(MCU_std, np.zeros(num_pad))

        while bin_edge.shape[0] <= max_num_bins:
            bin_edge = np.append(bin_edge, max(bin_edge)+bin_width)

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

        hist = hist / MCU_mean
        MCL_mean = MCL_mean / MCU_mean
        MCL_std = MCL_std / MCU_mean
        MCU_std = MCU_std / MCU_mean
        MCU_mean = MCU_mean / MCU_mean
        data_to_set = util.mapStringToLocal(proc_vars, locals())

    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


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


def monteCarlo_intracone_dist_common(param, sav_cfg, mc_type):
    # get any needed info from the save file
    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        coord = file['monteCarlo_' + mc_type]['coord'][()]
        num_mc = file['input_data']['bin_width'][()]
        bin_width = file['input_data']['bin_width'][()]
        dist_area_norm = file['input_data']['dist_area_norm'][()]

    proc = 'monteCarlo_'+mc_type+'_intracone_dist'
    proc_vars = sav_cfg[proc]['variables']

    if len(coord[0].shape) == 2 and coord[0].shape[1] == 2:

        dist = np.zeros((num_mc, coord[0].shape[0], coord[0].shape[0]))
        nearest_dist = np.zeros(num_mc)
        mean_nearest = np.zeros(num_mc)
        std_nearest = np.zeros(num_mc)
        hist = np.empty(num_mc, dtype=np.ndarray)
        max_hist_bin = 0
        for mc in np.arange(0, num_mc):
            this_coord = coord[mc, :, :]
            dist[mc, :, :], mean_nearest[mc], std_nearest[mc], hist[mc], bin_edge, annulus_area = intracone_dist_common(this_coord, bin_width, dist_area_norm)
            if hist[mc].shape[0] > max_hist_bin:
                max_hist_bin = hist[mc].shape[0]

        hist_mat = np.zeros([num_mc, max_hist_bin])
        for mc in np.arange(0, num_mc):
            hist_mat[mc, 0:hist[mc].shape[0]] = hist[mc]

        hist = hist_mat

        while len(bin_edge) < max_hist_bin + 1:
            bin_edge = np.append(bin_edge, np.max(bin_edge)+bin_width)

        mean_hist = np.mean(hist_mat, 0)
        std_hist = np.std(hist_mat, 0)

        data_to_set = util.mapStringToLocal(proc_vars, locals())
        
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def monteCarlo_coneLocked_intracone_dist_process(param, sav_cfg):
    """
    """
    monteCarlo_intracone_dist_common(param, sav_cfg, 'coneLocked')


def monteCarlo_uniform_intracone_dist_process(param, sav_cfg):
    """
    """
    monteCarlo_intracone_dist_common(param, sav_cfg, 'uniform')


def monteCarlo_process(param, sav_cfg, mc_type):
    # get any needed info from the save file
    sav_fl = param['sav_fl']
    with h5py.File(sav_fl, 'r') as file:
        real_coord = file['input_data']['cone_coord'][()]
        num_mc = file['input_data']['num_mc'][()]
        img = file['input_data']['cone_img'][()]
        bin_width = file['input_data']['bin_width'][()]
        dist_area_norm = file['input_data']['dist_area_norm']

    proc = 'monteCarlo_' + mc_type
    proc_vars = sav_cfg[proc]['variables']

    if len(real_coord.shape) == 2 and real_coord.shape[1] == 2:
        data_to_set = {}
        num_coord = real_coord.shape[0]
        if mc_type == 'uniform':
            xlim = [0, img.shape[0]]
            ylim = [0, img.shape[1]]
            coord = calc.monteCarlo_uniform(num_coord, num_mc, xlim, ylim)
            data_to_set = util.mapStringToLocal(proc_vars, locals())
        elif mc_type == 'coneLocked':
            # look for all cone mosaic for this data
            mosaic = param['mosaic']
            save_path = os.path.dirname(sav_fl)
            all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5'
            # try:
            with h5py.File(all_coord_fl, 'r') as file:
                all_coord = file['input_data']['cone_coord'][()]
            coord = calc.monteCarlo_coneLocked(num_coord, all_coord, num_mc)
            data_to_set = util.mapStringToLocal(proc_vars, locals())
            # except:
            #     print('could not find "' + all_coord_fl + ", cannot create coneLocked monteCarlo, skipping...")
            #     data_to_set = util.mapStringToNan(proc_vars)
        
    else:
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def monteCarlo_uniform_process(param, sav_cfg):
    monteCarlo_process(param, sav_cfg, 'uniform')


def monteCarlo_coneLocked_process(param, sav_cfg):
    monteCarlo_process(param, sav_cfg, 'coneLocked')


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

    # get any needed info from the save file
    with h5py.File(sav_fl, 'r') as file:
        coord = file['input_data']['cone_coord'][()]
        bin_width = file['input_data']['bin_width'][()]
        dist_area_norm = file['input_data']['dist_area_norm'][()]

    # to store the outputs of this process
    data_to_set = {}

    # if this is a valid coordinate dataset for this process...
    if len(coord.shape) == 2 and coord.shape[1] == 2:
        dist, mean_nearest, std_nearest, hist, bin_edge, annulus_area = intracone_dist_common(coord, bin_width, dist_area_norm)
        data_to_set = util.mapStringToLocal(proc_vars, locals())

    else:  # otherwise set these values to NaNs
        data_to_set = util.mapStringToNan(proc_vars)

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def input_data_process(param, sav_cfg):
    """

    """
    proc = 'input_data'
    proc_vars = sav_cfg[proc]['variables']
    data_to_set = {}
    for var in proc_vars:
        if var == 'cone_img':
            data_to_set[var] = plt.imread(param['img_fl'])
        elif var == 'cone_coord':
            data_to_set[var] = (np.loadtxt(param['coord_fl'], delimiter=','))
        else:
            data_to_set[var] = param[var]
    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def meta_process(param, sav_cfg):
    """

    """
    proc = 'meta'
    proc_vars = sav_cfg[proc]['variables']
    data_to_set = {}
    for var in proc_vars:
        data_to_set[var] = param[var]

    flsyst.setProcessVarsFromDict(param, sav_cfg, proc, data_to_set)


def mandatoryProcesses(user_param, sav_cfg):
    """

    """
    # identify mandatory processes
    # should consider making this a list in the yaml rather than a property of the individual files
    mand = sav_cfg['mandatory_proc']

    # get all files to check
    fls = []
    processes = user_param['processes']
    for proc in processes:
        fls = np.union1d(fls, processes[proc])

    fls = fls.astype(int)

    for ind in fls:
        param = unpackThisParam(user_param, ind)
        if not os.path.exists(param['sav_fl']):
            print('created file: ' + param['sav_fl'])
        for proc in mand:
            globals()[sav_cfg[proc]['process']](param, sav_cfg)


def unpackThisParam(user_param, ind):
    index = user_param['coord_index']
    param = {}
    param['coord_fl'] = user_param['coord_fl_name'][ind]
    param['img_fl'] = user_param['img_fl_name'][index['mosaic'][ind]]
    param['sav_fl'] = user_param['save_name'][ind]
    param['mosaic'] = user_param['mosaic'][index['mosaic'][ind]]
    param['subject'] = user_param['subject'][index['subject'][ind]]
    param['angle'] = user_param['angle'][index['angle'][ind]]
    param['eccentricity'] = user_param['eccentricity'][index['eccentricity'][ind]]
    param['conetype'] = user_param['conetype'][index['conetype'][ind]]
    param['coord_unit'] = user_param['coord_unit']
    param['bin_width'] = user_param['bin_width']
    param['dist_area_norm'] = user_param['dist_area_norm']
    param['conetype_color'] = user_param['conetype_color'][index['conetype'][ind]]
    param['num_mc'] = user_param['num_mc']

    return param


# this can be streamlined and made not single/multi specific if I add
# parameters to the yaml that define order of operations
def runSingleMosaicProcess(user_param, sav_cfg):
    """

    """
    mandatoryProcesses(user_param, sav_cfg)
    processes = user_param['processes']

    for proc in sav_cfg['order_optional_proc']:
        if proc not in sav_cfg.keys():
            print('process "' + proc + '" listed under optional processes is not found in the configuration file, skipping...')
        elif proc in processes.keys():
            print('Running process "' + proc + '" on ' + str(len(processes[proc])) + ' mosaic coordinate files...')
            for ind in processes[proc]:
                param = unpackThisParam(user_param, ind)
                globals()[sav_cfg[proc]['process']](param, sav_cfg)


# def viewMosaics(save_name, mosaic, index, selection = [:]):

#     for mos in mosaic:
#         for sav_fl in save_name[selection]:


#         all_fl = 
#         coord = {}
#         cone_type = {}
#         id = mosaic + ', ' + cone_type 
#         img = 
#         color = 


#         with h5py.File()
#         plotOnROI(img, coord, cone_type, id, color, **kwargs)


def viewIntraconeDistHists(save_names, save_things=False, save_path=''):
    for fl in save_names:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            hist = file['intracone_dist']['hist'][()]
            bin_edge = file['intracone_dist']['bin_edge'][()]
            mosaic = bytes(file['meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
        num_cone = coord.shape[0]
        id = mosaic + '_' + conetype
        if not np.isnan(hist[0]):

            # set up inputs to plot
            xlab = 'distance, ' + coord_unit
            ylab = 'bin count (binsize = ' + str(bin_edge[1]-bin_edge[0])
            tit = 'intracone distance (' + str(num_cone) + " cones)"
            x = bin_edge[1:]/2

            # view histogram
            
            ax = show.line(x, hist, id, plot_col=conetype_color, title=tit, xlabel=xlab, ylabel=ylab)

            ax.figure

            if save_things:
                savnm = save_path + id + '.png'
                plt.savefig(savnm)
            
        else:
            print(id + ' contains < 2 cones, skipping... ')


def viewMonteCarlo(save_name, mc_type, mc, save_things=False, save_path=''):
    for fl in save_name:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            mosaic = bytes(file['meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            num_mc = file['input_data']['num_mc'][()]
            coord = file['monteCarlo_'+mc_type]['coord'][()]
        if not np.isnan(coord[0]).any():
            num_cone = coord.shape[1]
            for m in mc:
                id_str = 'monteCarlo_' + mc_type + '_(' + str(m+1) + '//' + str(num_mc) + ')_' + mosaic + '_(' + str(num_cone) + ' cones)'
                xlab = coord_unit
                ylab = coord_unit
                this_coord = np.zeros([num_cone,2])
                this_coord[:,:] = coord[m, :, :]

                ax = show.scatt(this_coord, id_str, plot_col=conetype_color, xlabel=xlab, ylabel=ylab)

            ax.figure

            if save_things:
                savnm = save_path + mosaic + '_' + conetype + '.png'
                plt.savefig(savnm)

        else:
            print('no monteCarlo for "' + fl + '," skipping')


def viewMonteCarloStats(save_name, mc_type, scale_std=1, save_things=False, save_path=''):
    for fl in save_name:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            mosaic = bytes(file['meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            num_mc = file['input_data']['num_mc'][()]
            bin_edge = file['monteCarlo_' + mc_type + '_intracone_dist']['bin_edge'][()]
            mean_hist = file['monteCarlo_' + mc_type + '_intracone_dist']['mean_hist'][()]
            std_hist = file['monteCarlo_' + mc_type + '_intracone_dist']['std_hist'][()]
        num_cone = coord.shape[0]
        id_str = mosaic + '_' + conetype
        if not np.isnan(mean_hist[0]):

            # set up inputs to plot
            xlab = 'distance, ' + coord_unit
            ylab = 'bin count (binsize = ' + str(bin_edge[1]-bin_edge[0])
            tit = 'MCU intracone distance (' + str(num_cone) + " cones, " + str(num_mc) + " MCUs)"
            x = bin_edge[1:]/2

            ax = show.shadyStats(x, mean_hist, std_hist, id_str, scale_std=scale_std,
                            plot_col=conetype_color, title=tit, xlabel=xlab,
                            ylabel=ylab)

            ax.figure

            if save_things:
                savnm = save_path + id_str + '.png'
                plt.savefig(savnm)

            
        # saving images
        # .png if it doesn't need to be gorgeous and scaleable
        # .pdf otherwise, or eps, something vectorized 
        # numpy does parallelization under the hood

        # manually setting up parallel in python kinda sucks
        #   mpi is one approach


def viewMCUnormed(save_name, scale_std=1, showNearestCone=False, save_things=False, save_path=''):
    for fl in save_name:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager
            coord = file['input_data']['cone_coord'][()]
            mosaic = bytes(file['meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")
            num_mc = file['input_data']['num_mc'][()]
            bin_width = file['input_data']['bin_width'][()]
            bin_edge = file['norm_by_MCU_mean']['bin_edge'][()]
            hist = file['norm_by_MCU_mean']['hist'][()]
            MCU_mean = file['norm_by_MCU_mean']['MCU_mean'][()]
            MCU_std = file['norm_by_MCU_mean']['MCU_std'][()]
            MCL_mean = file['norm_by_MCU_mean']['MCL_mean'][()]
            MCL_std = file['norm_by_MCU_mean']['MCL_std'][()]
            all_cone_mean_nearest = file['norm_by_MCU_mean']['all_cone_mean_nearest'][()]

        num_cone = coord.shape[0]
        id_str = mosaic + '_' + conetype
        if not np.isnan(hist).any():
            print(fl)
            # set up inputs to plot
            xlab = 'distance, ' + coord_unit
            ylab = 'bin count (binsize = ' + str(bin_width)
            tit = 'intracone dists normed by MCU (' + str(num_cone) + " cones, " + str(num_mc) + " MCs)"
            x = bin_edge[1:]/2

            print(bin_edge[0])

            half_cone_rad = all_cone_mean_nearest / 2
            cone_rad_x = np.arange(half_cone_rad, half_cone_rad + (5 * all_cone_mean_nearest + 1), step=all_cone_mean_nearest)
            lin_extent = 1.5

            if showNearestCone:
                for lin in cone_rad_x:
                    if lin == cone_rad_x[0]:
                        ax = show.line([lin, lin], [-1 * lin_extent, lin_extent], id='cone-dist', plot_col='olive')
                    else:
                        ax = show.line([lin, lin], [-1 * lin_extent, lin_extent], id='cone-dist', ax=ax, plot_col='olive')

                ax = show.shadyStats(x, MCU_mean, MCU_std, id_str, scale_std=scale_std,
                                    ax=ax, plot_col='dimgray')
            else: 
                ax = show.shadyStats(x, MCU_mean, MCU_std, id_str, scale_std=scale_std,
                                    plot_col='dimgray')

            ax = show.shadyStats(x, MCL_mean, MCL_std, id_str, ax=ax, scale_std=scale_std,
                                 plot_col=conetype_color)

            ax = show.line(x, hist, ax=ax, plot_col='w', id=id_str,
                      xlabel=xlab, ylabel=ylab, title=tit)
            

            if showNearestCone:
                plt.xlim([0, half_cone_rad + 5 * all_cone_mean_nearest + 1])
                plt.ylim([-0, lin_extent])

            ax.figure

            if save_things:
                savnm = save_path + id_str + '.png'
                plt.savefig(savnm)
        else:
            print('no')