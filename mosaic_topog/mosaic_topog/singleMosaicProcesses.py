from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import mosaic_topog.flsyst as flsyst
import mosaic_topog.calc as calc
import mosaic_topog.show as show


def monteCarlo_process(sav_fl, mc_type):
    # get any needed info from the save file
    with h5py.File(sav_fl, 'r') as file:
        coords = np.asarray(file['input_data']['cone_coord'][()])
        num_mc = np.asarray(file['meta']['img'])
    #     img =
    # num_coords
    # xlim =
    # ylim =

    # mc_coords = calc.monteCarlo(num_coords, num_mc, mc_type, xlim, ylim)

    # proc = 'input_data'
    # data_to_set = [];
    # flsyst.setProcessVarsFromDict(sav_fl, sav_cfg, proc, data_to_set)


def monteCarlo_uniform_process(sav_fl):
    monteCarlo_process(sav_fl, 'uniform')


def monteCarlo_coneLocked_process(sav_fl):
    monteCarlo_process(sav_fl, 'coneLocked')


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
        coords = np.asarray(file['input_data']['cone_coord'][()])
        bin_width = float(file['input_data']['bin_width'][()])
        dist_area_norm = int(file['input_data']['dist_area_norm'][()])

    # to store the outputs of this process
    data_to_set = {}

    # if this is a valid coordinate dataset for this process...
    if len(coords.shape) == 2 and coords.shape[1] == 2:
        # get intracone distances
        dist = calc.dist_matrices(coords)

        # get avg and std of nearest cone distance in the mosaic
        nearest_dist = []
        for cone in np.arange(0, dist.shape[0]):
            # get row for this cone's distance to every other cone
            row = dist[cone, :]
            # find the index where the distance = -1 if it exists - this is self and shouldn't be included
            row = np.delete(row, np.nonzero(row == -1))
            # get the minimum value in the row
            nearest_dist.append(row.min())

        mean_nearest = np.mean(np.array(nearest_dist))
        std_nearest = np.std(np.array(nearest_dist))

        hist, bin_edge = calc.distHist(dist, bin_width)

        annulus_area = calc.annulusArea(bin_edge)

        if dist_area_norm:
            # normalize cone counts in each bin by the area of each annulus from which cones were counted 
            for ind, bin in enumerate(hist):
                dist[ind] = bin/annulus_area[ind]

        for var in proc_vars:
            data_to_set[var] = locals()[var]
    else:  # otherwise set these values to NaNs
        for var in proc_vars:
            data_to_set[var] = np.asarray([nan])

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
            with h5py.File(param['sav_fl'], 'w') as file:
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

    return param


def runSingleMosaicProcess(user_param, sav_cfg):
    """

    """
    mandatoryProcesses(user_param, sav_cfg)
    
    processes = user_param['processes']
    # this can be streamlined and made not single/multi specific if I add
    # parameters to the yaml that define order of operations
    if 'intracone_dist' in processes:
        for ind in processes['intracone_dist']:
            param = unpackThisParam(user_param, ind)
            globals()[sav_cfg['intracone_dist']['process']](param, sav_cfg)


def viewIntraconeDistHists(save_names):
    for fl in save_names:
        # get intracone distance histogram data and plotting parameters from the save file
        with h5py.File(fl, 'r') as file:  # context manager

            hist = file['intracone_dist']['hist'][()]
            bin_edge = file['intracone_dist']['bin_edge'][()]
            mosaic = bytes(file['meta']['mosaic'][()]).decode("utf8")
            conetype = bytes(file['meta']['conetype'][()]).decode("utf8")
            coord_unit = bytes(file['input_data']['coord_unit'][()]).decode("utf8")
            conetype_color = bytes(file['input_data']['conetype_color'][()]).decode("utf8")

        id = mosaic + '_' + conetype
        if not np.isnan(hist[0]):

            # set up inputs to plot
            xlab = 'distance, ' + coord_unit
            ylab = 'cones in bin (note histogram represents numcones^2)'
            tit = 'intracone distance'
            x = bin_edge[1:]/2

            # view histogram
            show.line(x, hist, id, plot_col=conetype_color, title=tit, xlabel=xlab, ylabel=ylab)

        else:
            print(id + ' contains < 2 cones, skipping... ')

        # saving images
        # .png if it doesn't need to be gorgeous and scaleable
        # .pdf otherwise, or eps, something vectorized 
        # numpy does parallelization under the hood

        # manually setting up parallel in python kinda sucks
        #   mpi is one approach
    