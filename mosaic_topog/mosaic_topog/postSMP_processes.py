import numpy as np
import h5py
import mosaic_topog.flsyst as flsyst
import mosaic_topog.utilities as util
import mosaic_topog.flsyst as flsyst
import mosaic_topog.show as show
import mosaic_topog.calc as calc
import matplotlib.pyplot as plt
import scipy
import numpy
import os

# def getAnalysisTiers(sav_cfg):
#     """
#     Inputs
#     ------

#     Outputs
#     -------
#     """
#     analysis_proc = []
#     analysis_tiers = []
#     for key in sav_cfg.keys():
#         if 'process_type' in sav_cfg[key]:
#             if sav_cfg[key]['process_type'] == 'analysis':
#                 analysis_proc.append(key)
#                 analysis_tiers.append(sav_cfg[key]['analysis_tier'])
#     max_tier = np.amax(analysis_tiers)
#     analyses_by_tier = []
#     for tier in np.arange(0, max_tier):
#         analyses_by_tier.append(np.array(analysis_proc)[np.nonzero(analysis_tiers == tier+1)[0]])

#     return analyses_by_tier


# def run_analysis_tiers_process(user_param, sav_cfg):
#     processes = user_param['processes']
#     tiers = getAnalysisTiers(sav_cfg)
#     for proc in tiers:
#         if proc in user_param['analyses_to_run'][0]:
#             for ind in processes[proc[0]]:
                
#                 print('Running process "' + proc[0] + '" on file' + str(ind+1) + '/' + str(len(processes[proc[0]])) +'...') 
#                 param = unpackThisParam(user_param, ind)
#                 globals()[sav_cfg[proc[0]]['process']](param, sav_cfg)


def ND_process(psmp_sav_cfg, psmp_params):
    proc = 'ND'
     ## GET THINGS
    proc_vars = psmp_sav_cfg[proc]['variables']
    smp_sav_cfg_path = psmp_params['smp_sav_cfg_path'][0]
    smp_input_cfg_path = psmp_params['smp_input_cfg_path'][0]
    smp_input_cfg = flsyst.readYaml(smp_input_cfg_path)
    smp_sav_cfg = flsyst.readYaml(smp_sav_cfg_path)
    test_var = psmp_params['test_conetype_ROIs_to_pull'][0]
    percentiles = [.025, .25, .5, .75, .975] #SHOULD BE DEFINED IN CONFIG FILE
    xlsx_path = psmp_params['update_ecc_path'][0]
    A_var = psmp_params['all_conetype_ROIs_to_pull'][0]
    nd_n_nearest = 6

    A_d2g = [['basic_stats', 'num_cones'],
            ['measured_voronoi', 'icd_mean'],
            ['input_data', 'cone_coord'],
            ['measured_intracone_dist', 'hist_mat'],
            ['measured_voronoi', 'icd'],
            ['measured_voronoi', 'neighbors_cones'],
            ['measured_voronoi', 'num_neighbor'],
            ['measured_voronoi', 'density']]
              
    S_d2g = [['mosaic_meta', 'mosaic'],
            ['mosaic_meta', 'eccentricity'],
            ['mosaic_meta', 'subject'],
            ['mosaic_meta', 'angle'],
            ['measured_voronoi', 'density'],
            ['basic_stats', 'num_cones'],
            ['input_data', 'cone_coord']]

    S_smp_user_param = flsyst.fileInfoToUserParams(smp_input_cfg, smp_sav_cfg)
    S_smp_user_param = getParam(S_smp_user_param, smp_sav_cfg)
    S_smp_data = getOutputs(S_d2g, test_var, S_smp_user_param)
    ROIs = [bytes(z).decode('utf8') for z in S_smp_data['mosaic_meta__mosaic']]
    cfs_p2d = util.getCF_DpP(xlsx_path, S_smp_data)
    cfs_p2a = cfs_p2d * 60
    S_coord = S_smp_data['input_data__cone_coord']

    if S_smp_user_param['path_AllConeRef']:
        meas_A_folder = S_smp_user_param['fileinput_for_AllConeRef']
    else:
        meas_A_folder = smp_input_cfg_path[0:np.nonzero([z=='\\' for z in smp_input_cfg_path])[0][-1]+1] 
 
    A_cfg = flsyst.readYaml(meas_A_folder + 'input_config_1.yml')        
    A_user_param = flsyst.fileInfoToUserParams(A_cfg, smp_sav_cfg)
    A_user_param = getParam(A_user_param, smp_sav_cfg)
    A_smp_data = getOutputs(A_d2g, A_var, A_user_param)
    A_coord = A_smp_data['input_data__cone_coord']
    A_icd = [A_smp_data['measured_voronoi__icd'][z] * cfs_p2a[z] for z in np.arange(0,len(ROIs))]
    all_A_ind2_S = []

    ND_mwu = util.NaN([len(ROIs), nd_n_nearest, 3, 2])
    ND_siglowhigh = np.zeros([len(ROIs), nd_n_nearest, 3]) 
    ND_Q = util.NaN([len(ROIs), nd_n_nearest, len(percentiles), 2]) 
    dims = {}
    dims['ND_mwu'] = ['ROI', 'nth nearest cone', 'left, right, both tailed mwu', 'mwu metrics u,p']
    dims['ND_Q'] = ['ROI', 'nth nearest cone', 'quantiles', 'S-cone neighbors, L-cone neighbors']
    dims['ND_siglowhigh'] = ['ROI', 'nth nearest cone', 'sig low, high (left, right, both tailed)']

    for mos, tit in enumerate(ROIs):

        A_ind2_S = util.all_coord_match_inds(A_coord[mos], S_coord[mos])
        A_ind2_LM = np.nonzero([not z in A_ind2_S for z in np.arange(0, A_coord[mos].shape[0])])[0]
        A_ind2_LM = A_ind2_LM.astype('int64')
        all_A_ind2_S.append(A_ind2_S)

        S_nd = np.sort(A_icd[mos][A_ind2_S], axis=1)
        LM_nd = np.sort(A_icd[mos][A_ind2_LM], axis=1)

        for n in np.arange(0,nd_n_nearest):
            S_toplot = util.deNaN(S_nd[:, n])
            LM_toplot = util.deNaN(LM_nd[:, n])

            mwu_stat_less = scipy.stats.mannwhitneyu(S_toplot, LM_toplot, alternative='less')
            mwu_stat_greater = scipy.stats.mannwhitneyu(S_toplot, LM_toplot, alternative='greater')
            mwu_stat = scipy.stats.mannwhitneyu(S_toplot, LM_toplot)

            ND_Q[mos, n, :, 0] = np.quantile(S_toplot, percentiles, method = 'nearest') #* CF_arcmin_to_pixels
            ND_Q[mos, n, :, 1] = np.quantile(LM_toplot, percentiles, method = 'nearest') #* CF_arcmin_to_pixels
            
            ND_mwu[mos, n, 0, :] = [mwu_stat_less.statistic, mwu_stat_less.pvalue]
            ND_mwu[mos, n, 1, :] = [mwu_stat_greater.statistic, mwu_stat_greater.pvalue]
            ND_mwu[mos, n, 2, :] = [mwu_stat.statistic, mwu_stat.pvalue]

            if ND_mwu[mos, n, 0, 1] < 0.05:
                ND_siglowhigh[mos,n,0] = 1
            if ND_mwu[mos, n, 1, 1] < 0.05:
                ND_siglowhigh[mos,n,1] = 1
            if ND_mwu[mos, n, 2, 1] < 0.05:
                ND_siglowhigh[mos,n,2] = 1

    ## SAVE TO HDF5
    ## this should be a flsyst function that's called in a default psmp 
    if not os.path.exists(psmp_params['psmp_save_path'][0]):
        with h5py.File(psmp_params['psmp_save_path'][0], 'w') as file:
            print('created file: ' + psmp_params['psmp_save_path'][0])
    data_to_set = util.mapStringToLocal(proc_vars, locals())
    flsyst.setProcessVarsFromDict(psmp_params, psmp_sav_cfg, proc, data_to_set)


def NN_process(psmp_sav_cfg, psmp_params):
    proc = 'NN'
     ## GET THINGS
    proc_vars = psmp_sav_cfg[proc]['variables']
    S_smp_sav_cfg_path = psmp_params['smp_sav_cfg_path'][0]
    S_smp_input_cfg_path = psmp_params['smp_input_cfg_path'][0]
    S_smp_input_cfg = flsyst.readYaml(S_smp_input_cfg_path)
    smp_sav_cfg = flsyst.readYaml(S_smp_sav_cfg_path)
    test_var = psmp_params['test_conetype_ROIs_to_pull'][0]
    percentiles = [.025, .25, .5, .75, .975] #SHOULD BE DEFINED IN CONFIG FILE
    xlsx_path = psmp_params['update_ecc_path'][0]
    A_var = psmp_params['all_conetype_ROIs_to_pull'][0]
    
    NN_S_d2g = [['input_data', 'cone_coord'],
                ['mosaic_meta', 'mosaic']]

    S_smp_user_param = flsyst.fileInfoToUserParams(S_smp_input_cfg, smp_sav_cfg)
    S_smp_user_param = getParam(S_smp_user_param, smp_sav_cfg)
    NN_S_data = getOutputs(NN_S_d2g, test_var, S_smp_user_param)
    ROIs = [bytes(z).decode('utf8') for z in NN_S_data['mosaic_meta__mosaic']]
    cfs_p2d = util.getCF_DpP(xlsx_path, NN_S_data)
    cfs_p2a = cfs_p2d * 60
    S_coord = NN_S_data['input_data__cone_coord']
    NN_A_d2g = [['measured_voronoi','num_neighbor'],
                ['input_data', 'cone_coord']]
    
    if S_smp_user_param['path_AllConeRef']:
        meas_A_folder = S_smp_user_param['fileinput_for_AllConeRef']
    else:
        meas_A_folder = S_smp_input_cfg_path[0:np.nonzero([z=='\\' for z in S_smp_input_cfg_path])[0][-1]+1] 
 
    A_cfg = flsyst.readYaml(meas_A_folder + 'input_config_1.yml')        
    A_user_param = flsyst.fileInfoToUserParams(A_cfg, smp_sav_cfg)
    A_user_param = getParam(A_user_param, smp_sav_cfg)
    NN_A_data = getOutputs(NN_A_d2g, A_var, A_user_param)
    A_coord = NN_A_data['input_data__cone_coord']

    A_num_neighb = NN_A_data['measured_voronoi__num_neighbor']

    NN_mwu = util.NaN([len(ROIs), 3, 2])
    NN_siglowhigh = np.zeros([len(ROIs), 3]) 
    NN_Q = util.NaN([len(ROIs), len(percentiles), 2]) 
    dims = {}
    dims['NN_mwu'] = ['ROI', 'left,right tailed mwu', 'mwu metrics u,p']
    dims['NN_Q'] = ['ROI', 'quantiles', 'S-cone neighbors, L-cone neighbors']
    dims['NN_siglowhigh'] = ['ROI', 'sig low, high (left, right tailed)']


    for mos, tit in enumerate(ROIs):
        A_ind2_S = util.all_coord_match_inds(A_coord[mos], S_coord[mos])
        A_ind2_LM = np.nonzero([not z in A_ind2_S for z in np.arange(0, A_coord[mos].shape[0])])[0]
        A_ind2_LM = A_ind2_LM.astype('int64')

        S_nn = util.deNaN(A_num_neighb[mos][A_ind2_S])
        LM_nn = util.deNaN(A_num_neighb[mos][A_ind2_LM])

        S_nn_unique = np.unique(S_nn)
        S_nn_unique_count = [np.sum(z==S_nn) for z in S_nn_unique]
        S_nn_unique_prob = [z/S_nn.shape[0] for z in S_nn_unique_count]

        LM_nn_unique = np.unique(LM_nn)
        LM_nn_unique_count = [np.sum(z==LM_nn) for z in LM_nn_unique]
        LM_nn_unique_prob = [z/LM_nn.shape[0] for z in LM_nn_unique_count]

        NN_Q[mos, :, 0] = np.quantile(S_nn, percentiles, method='nearest')
        NN_Q[mos, :, 1] = np.quantile(LM_nn, percentiles, method='nearest')
        
        mwu_stat_less = scipy.stats.mannwhitneyu(S_nn, LM_nn, alternative='less')
        mwu_stat_greater = scipy.stats.mannwhitneyu(S_nn, LM_nn, alternative='greater')
        mwu_stat = scipy.stats.mannwhitneyu(S_nn, LM_nn)

        NN_mwu[mos, 0, :] = [mwu_stat_less.statistic, mwu_stat_less.pvalue]
        NN_mwu[mos, 1, :] = [mwu_stat_greater.statistic, mwu_stat_greater.pvalue]
        NN_mwu[mos, 2, :] = [mwu_stat.statistic, mwu_stat.pvalue]

        if NN_mwu[mos,0,1] < 0.05:
            NN_siglowhigh[mos, 0] = 1

        if NN_mwu[mos,1,1] < 0.05:
            NN_siglowhigh[mos, 1] = 1 

        if NN_mwu[mos,2,1] < 0.05:
            NN_siglowhigh[mos, 2] = 1 

    ## SAVE TO HDF5
    ## this should be a flsyst function that's called in a default psmp 
    if not os.path.exists(psmp_params['psmp_save_path'][0]):
        with h5py.File(psmp_params['psmp_save_path'][0], 'w') as file:
            print('created file: ' + psmp_params['psmp_save_path'][0])
    data_to_set = util.mapStringToLocal(proc_vars, locals())
    flsyst.setProcessVarsFromDict(psmp_params, psmp_sav_cfg, proc, data_to_set)


def DRP_process(psmp_sav_cfg, psmp_params):
    proc = 'DRP'

    ## GET THINGS
    proc_vars = psmp_sav_cfg[proc]['variables']
    smp_sav_cfg_path = psmp_params['smp_sav_cfg_path'][0]
    smp_input_cfg_path = psmp_params['smp_input_cfg_path'][0]
    smp_input_cfg = flsyst.readYaml(smp_input_cfg_path)
    smp_sav_cfg = flsyst.readYaml(smp_sav_cfg_path)
    test_var = psmp_params['test_conetype_ROIs_to_pull'][0]
    test_types = psmp_params['test_types'][0]
    null_type = psmp_params['null_type'][0]
    mosaic_types = psmp_params['mosaic_types'][0]
    num_simtrl = psmp_params['num_sim'][0][0]
    percentiles = [.025, .25, .5, .75, .975] #SHOULD BE DEFINED IN CONFIG FILE
    xlsx_path = psmp_params['update_ecc_path'][0]
    null_ind = np.nonzero([z == null_type for z in mosaic_types])[0][0].astype('int64')
    test_inds = np.nonzero([not z == null_ind for z in np.arange(0, len(mosaic_types))])[0].astype('int64')

    DRP_d2g = [['drp', 'effective_radius'],
               ['drp', 'packing_factor']]
    DRP_d2g = [[mtype + '_' + met[0], met[1]] for met in DRP_d2g for mtype in mosaic_types]
    DRP_d2g.append(['mosaic_meta', 'mosaic'])
    DRP_d2g.append(['measured_drp', 'maximum_radius'])

    smp_user_param = flsyst.fileInfoToUserParams(smp_input_cfg, smp_sav_cfg)
    smp_user_param = getParam(smp_user_param, smp_sav_cfg)
    DRP_smp_data = getOutputs(DRP_d2g, test_var, smp_user_param)
    ROIs = [bytes(z).decode('utf8') for z in DRP_smp_data['mosaic_meta__mosaic']]
    cfs_p2d = util.getCF_DpP(xlsx_path, DRP_smp_data)
    cfs_p2a = cfs_p2d * 60
    MR = np.squeeze(np.multiply(DRP_smp_data['measured_drp__maximum_radius'], cfs_p2a))
    # print(cfs_p2d)
    ## INITIALIZE THINGS
    metric_labs = ['ER (arcmin): Effective Radius', 'PF (squared ratio): Packing Factor']
    met_dirs = ['_drp__effective_radius', '_drp__packing_factor']
    DRP_metric_alltrl = util.NaN([len(mosaic_types), len(ROIs), num_simtrl, len(metric_labs)])
    DRP_metric_qs = util.NaN([len(mosaic_types), len(ROIs), len(percentiles), len(metric_labs)])
    sig_ER = np.zeros([len(mosaic_types)-1, len(ROIs)], dtype='int64')
    dims = {}
    dims['DRP_metric_alltrl'] = ['mosaic type', 'ROI', 'trl', 'DRP metrics']
    dims['MR'] = ['ROIs']
    dims['DRP_metric_qs'] = ['mosaic type', 'ROI', 'quantiles', 'DRP metrics']
    dims['sigERif'] = ['mtype', 'mos']

    
    for s_ind, s in enumerate(mosaic_types):
        for met_ind, met in enumerate(metric_labs):
            
            if 'ER' in met:
                cf_mult = cfs_p2a
                met_temp  = np.array([DRP_smp_data[s + met_dirs[met_ind]][z] * cf_mult[z] for z in np.arange(0, len(ROIs))])
                
                
            if 'PF' in met:
                met_temp = DRP_smp_data[s + met_dirs[met_ind]] 

            nan_inds = [np.isnan(z) for z in met_temp]
            met_temp[nan_inds] = 0

            if len(met_temp.shape) == 1:
                met_temp = np.expand_dims(met_temp, 1)  
            
            DRP_metric_alltrl[s_ind, :, 0:met_temp.shape[1], met_ind] = met_temp          
            
            for m_ind, mos in enumerate(ROIs):
                met_temp[m_ind,:]
                d2q = util.deNaN(met_temp[m_ind,:])

                if d2q.shape[0] > 0:
                    DRP_metric_qs[s_ind, m_ind, :, met_ind] = np.quantile(d2q, percentiles)
                else:
                    print('no DRP data for....')
                    print(s + ' ' + met + ' ' + mos)
                
                    print(DRP_smp_data[s + met_dirs[met_ind]][m_ind])
                    print(cf_mult[m_ind])
    
    ## ID SIGNIFICANT ROIS
    for c_ind, c in enumerate(test_types):
        sigs = [DRP_metric_qs[null_ind,z,4,0] < DRP_metric_qs[test_inds[c_ind], z, 0, 0] 
                    for z in np.arange(0, len(ROIs))]
        sig_ER[c_ind, :] = sigs

    ## SAVE TO HDF5
    ## this should be a flsyst function that's called in a default psmp 
    if not os.path.exists(psmp_params['psmp_save_path'][0]):
        with h5py.File(psmp_params['psmp_save_path'][0], 'w') as file:
            print('created file: ' + psmp_params['psmp_save_path'][0])
    data_to_set = util.mapStringToLocal(proc_vars, locals())
    flsyst.setProcessVarsFromDict(psmp_params, psmp_sav_cfg, proc, data_to_set)



def CRA_process(psmp_sav_cfg, psmp_params):
    proc = 'CRA'
    proc_vars = psmp_sav_cfg[proc]['variables']
    smp_sav_cfg_path = psmp_params['smp_sav_cfg_path'][0]
    smp_input_cfg_path = psmp_params['smp_input_cfg_path'][0]
    smp_input_cfg = flsyst.readYaml(smp_input_cfg_path)
    smp_sav_cfg = flsyst.readYaml(smp_sav_cfg_path)
    test_var = psmp_params['test_conetype_ROIs_to_pull'][0]
    test_types = psmp_params['test_types'][0]
    null_type = psmp_params['null_type'][0]
    mosaic_types = psmp_params['mosaic_types'][0]
    num_simtrl = psmp_params['num_sim'][0][0]
    num_rings = 5 #is this hardset by smp?
    y_percentiles = [.025, .25, .5, .75, .975]
    null_ind = np.nonzero([z == null_type for z in mosaic_types])[0][0].astype('int64')
    test_inds = np.nonzero([not z == null_ind for z in np.arange(0, len(mosaic_types))])[0].astype('int64')

    xlsx_path = psmp_params['update_ecc_path'][0]
    

    # can define these dependencies in the input config file?
    CRA_d2g =  [['concentric_ring_analysis', 'num_rings'],
                ['concentric_ring_analysis', 'all_Sprop_in_rings'],
                ['concentric_ring_analysis', 'all_dist_in_rings'],
                ['concentric_ring_analysis', 'all_numS_in_rings'],
                ['concentric_ring_analysis', 'all_numA_in_rings'],
                ['concentric_ring_analysis','percentile_dist_in_rings']]
    temp = []
    for mtype in mosaic_types:
        for met in CRA_d2g:
            temp.append([mtype + '_' + met[0], met[1]])

    CRA_d2g = temp
    CRA_d2g.append(['coneLocked_maxSpacing', 'hex_radii_used'])
    CRA_d2g.append(['mosaic_meta','mosaic'])

    smp_user_param = flsyst.fileInfoToUserParams(smp_input_cfg, smp_sav_cfg)
    smp_user_param = getParam(smp_user_param, smp_sav_cfg)
    CRA_smp_data = getOutputs(CRA_d2g, test_var, smp_user_param)
    cfs_p2d = util.getCF_DpP(xlsx_path, CRA_smp_data)
    cfs_p2a = cfs_p2d * 60
    eMR = np.array([cfs_p2a[z] * np.nanmean(CRA_smp_data['coneLocked_maxSpacing__hex_radii_used'][z]) for z in np.arange(0,len(cfs_p2a))])    
    try:
        ROIs = [bytes(z).decode('utf') for z in CRA_smp_data['mosaic_meta__mosaic']]
    except:
        ROIs = CRA_smp_data['mosaic_meta__mosaic']

    ## change to pre-allocate list structure w/ utilities function
    rings_numS = []
    rings_numA = []
    rings_propS = []
    rings_dists = []

    for s_ind,s in enumerate(mosaic_types):
        rings_numS.append([])
        rings_numA.append([])
        rings_propS.append([])
        rings_dists.append([])
        
        set_rnumS = CRA_smp_data[s + '_concentric_ring_analysis__all_numS_in_rings']
        set_rnumA = CRA_smp_data[s + '_concentric_ring_analysis__all_numA_in_rings']
        set_rpropS = CRA_smp_data[s + '_concentric_ring_analysis__all_Sprop_in_rings']
        set_rdists = CRA_smp_data[s + '_concentric_ring_analysis__all_dist_in_rings']

        if len(set_rnumS[0].shape) == 4:
            set_rnumS = [set_rnumS[z][0:num_simtrl, :, :] for z in np.arange(len(set_rnumS))]
            set_rnumA = [set_rnumA[z][0:num_simtrl, :, :] for z in np.arange(len(set_rnumS))]
            set_rpropS = [set_rpropS[z][0:num_simtrl, :, :] for z in np.arange(len(set_rnumS))]
            set_rdists = [set_rdists[z][0:num_simtrl, :, :] for z in np.arange(len(set_rnumS))]

        print('getting ' + s + 'data from each smp file...')
        skipit = np.zeros([len(ROIs)])

        # this needs to be set directly from the input cfg file
        if s == 'measured':
            numtrl = 1
        else:
            numtrl = num_simtrl

        for mos, tit in enumerate(ROIs):

            if not np.all([np.isnan(z) for z in [set_rnumS[0]]]):
                if np.array(set_rnumS[mos]).shape[0] > 0:
                    if not np.all(np.array([np.isnan(z) for z in np.array(set_rnumS[mos])])): # isinstance(set_rnumS, list):

                        temp = np.empty([set_rnumS[mos].shape[0], set_rnumS[mos].shape[1], num_rings])
                        temp[:] = np.nan
                        
                        mos_rnumA = set_rnumA[mos]
                        mos_rnumS = set_rnumS[mos]
                        mos_rpropS = set_rpropS[mos]

                        mos_rdists = set_rdists[mos] * cfs_p2a[mos]

                        #does anything bad even happen without this? 
                        if (len(mos_rnumS.shape) == 2):
                            mos_rnumA = np.expand_dims(mos_rnumA, axis=0)
                            mos_rnumS = np.expand_dims(mos_rnumS, axis=0)
                            mos_rpropS = np.expand_dims(mos_rpropS, axis=0)
                            mos_rdists = np.expand_dims(mos_rdists, axis=0)

                        rings_numA[s_ind].append(mos_rnumA)
                        rings_numS[s_ind].append(mos_rnumS)
                        rings_propS[s_ind].append(mos_rpropS)
                        rings_dists[s_ind].append(mos_rdists)
                    else:
                        rings_numA[s_ind].append(np.nan)
                        rings_numS[s_ind].append(np.nan)
                        rings_propS[s_ind].append(np.nan)
                        rings_dists[s_ind].append(np.nan)
                        skipit[mos] = 1
                else:
                    rings_numA[s_ind].append(np.nan)
                    rings_numS[s_ind].append(np.nan)
                    rings_propS[s_ind].append(np.nan)
                    rings_dists[s_ind].append(np.nan)
                    skipit[mos] = 1
            else:
                rings_numA[s_ind].append(np.nan)
                rings_numS[s_ind].append(np.nan)
                rings_propS[s_ind].append(np.nan)
                rings_dists[s_ind].append(np.nan)
                skipit[mos] = 1

    prevalence_metrics = ['# S-cones', '% S-cones']
    mwu_met_labs = ['u', 'p']

    CRA_mwu = util.NaN([len(test_types), len(ROIs), num_simtrl, num_rings, len(prevalence_metrics), 2, len(mwu_met_labs)]) # [2nd to last dimension is high and low] [last dim]
    rings_siglowhigh = util.NaN([len(test_types), len(ROIs), num_simtrl, num_rings, len(prevalence_metrics), 2]) 
    rings_sigcounts = util.NaN([len(test_types), num_rings, len(prevalence_metrics),2])
    dat2see = [rings_numS, rings_propS]
    prevalence_null_alltrl_qs  = util.NaN([len(ROIs), num_rings, len(dat2see), 5])
    prevalence_test_onetrl_qs = util.NaN([len(test_types), len(ROIs), num_simtrl, num_rings, len(dat2see), 5])
    prevalence_test_alltrl_qs = util.NaN([len(test_types), len(ROIs), num_rings, len(dat2see), 5])

    dims = {}
    dims['rings_siglowhigh'] = ['test moosaic type', 'ROI', 'trial', 'ring', 'prevalence metric', 'low or high']
    dims['rings_sigcounts'] = ['test mosaic type', 'ring', 'prevalence metric', 'low or high']
    dims['prevalence_test_onetrl_qs']  = ['test mosaic type', 'ROI', 'trl', 'ring', 'prevalence metric', 'percentilesentile']
    dims['prevalence_null_alltrl_qs ']  = ['ROI', 'ring', 'prevalence metric', 'percentilesentile']
    dims['prevalence_test_alltrl_qs ']  = ['test mosaic type', 'ROI', 'ring', 'prevalence metric', 'percentilesentile']


    for d, dat in enumerate(dat2see): # does the analysis using numS or propS 
        print('')
        print('doing stats for ' + prevalence_metrics[d])
        for s_ind, s in enumerate(test_types):  
            print(s)
            if s == 'measured':
                numtrl = 1
            else:
                numtrl = num_simtrl

            mt_ind = np.nonzero([z == s for z in mosaic_types])[0][0].astype('int64')
            for mos, tit in enumerate(ROIs):  
                if not skipit[mos]:     
                    print('        ' + tit)  
                    for r in np.arange(0, num_rings): 
                        # print('            ring #' + str(r) )            
                        r_data = util.deNaN(dat[null_ind][mos][:,:,r].flatten())
                        
                        #this is dumb, overwrites itself
                        q = np.quantile(r_data, y_percentiles, method = 'nearest')    
                        prevalence_null_alltrl_qs [mos, r, d, :] =  q
                        
                        alltrl_dat = []
                        for trl in np.arange(0,numtrl):
                            # print('trl: ' + str(trl))
                            t_data = util.deNaN(dat[mt_ind][mos][trl,:,r].flatten())

                            alltrl_dat.extend(t_data.tolist())

                            q = np.quantile(t_data, y_percentiles, method = 'nearest')    
                            prevalence_test_onetrl_qs[s_ind, mos, trl, r, d, :] =  q

                            mwu_stat_less = scipy.stats.mannwhitneyu(t_data, r_data, alternative='less')
                            mwu_stat_greater = scipy.stats.mannwhitneyu(t_data, r_data, alternative='greater')

                            CRA_mwu[s_ind, mos, trl, r, d, 0, :] = [mwu_stat_less.statistic, mwu_stat_less.pvalue]
                            CRA_mwu[s_ind, mos, trl, r, d, 1, :] = [mwu_stat_greater.statistic, mwu_stat_greater.pvalue]
                            
                            if CRA_mwu[s_ind, mos, trl, r, d, 0, 1] < 0.05:
                                rings_siglowhigh[s_ind, mos, trl, r, d, 0] = 1
                            else:
                                rings_siglowhigh[s_ind, mos, trl, r, d, 0] = 0
                                
                            if CRA_mwu[s_ind, mos, trl, r, d, 1, 1] < 0.05:
                                rings_siglowhigh[s_ind, mos, trl, r, d, 1] = 1 
                            else:
                                rings_siglowhigh[s_ind, mos, trl, r, d, 1] = 0

                        q = np.quantile(alltrl_dat, y_percentiles, method = 'nearest')    
                        prevalence_test_alltrl_qs[s_ind, mos, r, d, :] =  q    
                    
                else:
                    print('        skipping ' + tit + ' for stats bc lack of cones')   
                            
            for r in np.arange(0, num_rings):
                rings_sigcounts[s_ind,r,d,0] = np.nansum(rings_siglowhigh[s_ind, :, :, r, d, 0].flatten()) 
                rings_sigcounts[s_ind,r,d,1] = np.nansum(rings_siglowhigh[s_ind, :, :, r, d, 1].flatten()) 

    CRA_met_lab = ['ExR_rings', 'ExF', 'ExS', 'ExRa']
    
    CRA_met = util.NaN([len(test_types), len(ROIs), num_simtrl, len(prevalence_metrics), len(CRA_met_lab)-1])
    percentiles = [.025, .25, .5, .75, .975]

    ExRa_trial_qs = util.NaN([len(test_types), len(ROIs), num_simtrl, len(prevalence_metrics), len(percentiles)])
    CRA_metric_alltrl_qs =  util.NaN([len(test_types), len(ROIs), len(prevalence_metrics), len(CRA_met_lab), len(percentiles)])

    ExRa_all_data = []
    dims['ExRa_all_data']  = ['prevalence metric', 'test mosaic type', 'ROI', 'trial', 'cones in the ExRth ring of S-cones']
    dims['ExRa'] = ['test moosaic type', 'ROI', 'trial', 'ring', 'prevalence metric', 'percentiles']
    dims['ExRr_ExF_ExS'] =  ['test moosaic type', 'ROI', 'trial', 'ring', 'prevalence metric', 'ExRr, ExF, ExS', 'percentiles']
    dims['CRA_metric_alltrl_qs'] = ['test mosaic type', 'ROI', 'prevalence metric', 'ExR, ExF, ExS, ExRa', 'percentiles']
    
    max_ExRa = 0
    for tm, topmet in enumerate(prevalence_metrics):
        ExRa_all_data.append([])
        for s_ind, s in enumerate(test_types):
            if s == 'measured':
                trl2see = 1
            else:
                trl2see = num_simtrl
            mt_ind = np.nonzero([s==z for z in mosaic_types])[0][0].astype('int64')
            ExRa_all_data[tm].append([])
            for mos, tit in enumerate(ROIs):
                ExRa_all_data[tm][len(ExRa_all_data[tm])-1].append([])   
                alltrl_ExRa = [] 
                
                for trl in np.arange(0, trl2see):
                    if not skipit[mos]:
                        lowrings = 0
                        siglow_inds = np.nonzero(rings_siglowhigh[s_ind, mos, trl, :, tm, 0].flatten())[0]
                        if siglow_inds.shape[0] > 0:                       
                            if siglow_inds[0] == 0:
                                diff_inds = np.diff(siglow_inds)
                                lowrings = 1

                                for i in diff_inds:
                                    if i == 1:
                                        lowrings+=1
                                    else:
                                        break
                
                        CRA_met[s_ind, mos, trl, tm, 0] = lowrings

                        if lowrings > 0:
                            try:
                                ExR_arcmin = util.deNaN(rings_dists[mt_ind][mos][trl, :, lowrings-1].flatten())
                            except:
                                print(trl)
                                

                            ExRa_all_data[tm][len(ExRa_all_data[tm])-1][mos].append(ExR_arcmin)
                            alltrl_ExRa.extend(ExR_arcmin.tolist())

                            max_ExRa = max([max_ExRa, len(ExR_arcmin)])
                            
                            ExR_ci = np.quantile(ExR_arcmin, percentiles, method='nearest')
                            ExRa_trial_qs[s_ind, mos, trl, tm, :] = ExR_ci
                            
                            if np.array([ExR_arcmin]).shape[0] == 0:
                                CRA_ExF = np.nan
                                numS_in_ExR = np.nan
                                numS_in_ExR = np.nan
                                CRA_ExS = np.nan
                            elif np.all(np.isnan(np.array(ExR_arcmin))):
                                CRA_ExF = np.nan
                                numS_in_ExR = np.nan
                                numS_in_ExR = np.nan
                                CRA_ExS = np.nan
                            else: 
                                CRA_ExF = (np.quantile(util.deNaN(ExR_arcmin), [.5], method='nearest')/ eMR[mos])**2
                                numS_in_ExR = np.nansum(rings_numS[mt_ind][mos][trl, :, 0:lowrings].flatten())
                                numA_in_ExR = np.nansum(rings_numA[mt_ind][mos][trl, :, 0:lowrings].flatten())
                                propS_in_ExR = numS_in_ExR/numA_in_ExR
                                rand_numS_in_ExR = np.nansum(rings_numS[null_ind][mos][trl, :, 0:lowrings].flatten())
                                rand_numA_in_ExR = np.nansum(rings_numA[null_ind][mos][trl, :, 0:lowrings].flatten())
                                rand_propS_in_ExR = rand_numS_in_ExR/rand_numA_in_ExR
                                
                                CRA_ExS = 1 - (propS_in_ExR/rand_propS_in_ExR)         
      
                            CRA_met[s_ind, mos, trl, tm, 1] = CRA_ExF
                            CRA_met[s_ind, mos, trl, tm, 2] = CRA_ExS
                            
                        else:
                            ExRa_all_data[tm][len(ExRa_all_data[tm])-1][mos].append([np.nan])
                            ExRa_trial_qs[s_ind, mos, trl, tm, :] = np.zeros([len(percentiles)])
                            alltrl_ExRa.extend([np.nan])
                    else:
                        ExRa_all_data[tm][len(ExRa_all_data[tm])-1][mos].append([np.nan])

                alltrl_ExRr = util.deNaN(CRA_met[s_ind, mos, :, tm, 0])
                alltrl_ExF = util.deNaN(CRA_met[s_ind, mos, :, tm, 1])
                alltrl_ExS = util.deNaN(CRA_met[s_ind, mos, :, tm, 2])
                alltrl_ExRa = util.deNaN(alltrl_ExRa)
                
                CRA_metric_alltrl_qs[s_ind, mos, tm, 0, :] = util.attemptQs(alltrl_ExRr, percentiles, 'ExRr')
                CRA_metric_alltrl_qs[s_ind, mos, tm, 1, :] = util.attemptQs(alltrl_ExF, percentiles, 'ExF')
                CRA_metric_alltrl_qs[s_ind, mos, tm, 2, :] = util.attemptQs(alltrl_ExS, percentiles, 'ExS')
                CRA_metric_alltrl_qs[s_ind, mos, tm, 3, :] = util.attemptQs(alltrl_ExRa, percentiles, 'ExRa')


    # what is this part even for???
    ExRa_all_data_temp = util.NaN([len(prevalence_metrics), len(test_types), len(ROIs), num_simtrl, max_ExRa])

    for tm_ind, tm in enumerate(prevalence_metrics):
        # print(tm)
        for s_ind in np.arange(0, len(test_types)):
            test_ind = test_inds[s_ind]
            for mos, tit in enumerate(ROIs):
                # print(mos)
                if not skipit[mos]:
                    if mosaic_types[test_ind] == 'measured':
                        trl2see = 1
                    else:
                        trl2see = num_simtrl
                    for trl in np.arange(0, trl2see):
                        ExRa_all_data_temp[tm_ind, s_ind, mos, trl, 0:len(ExRa_all_data[tm_ind][s_ind][mos][trl])] = ExRa_all_data[tm_ind][s_ind][mos][trl]

                # print('')
    ExRa_all_data = ExRa_all_data_temp

    ## this should be a flsyst function that's called in a default psmp 
    if not os.path.exists(psmp_params['psmp_save_path'][0]):
        with h5py.File(psmp_params['psmp_save_path'][0], 'w') as file:
            print('created file: ' + psmp_params['psmp_save_path'][0])

    data_to_set = util.mapStringToLocal(proc_vars, locals())
    flsyst.setProcessVarsFromDict(psmp_params, psmp_sav_cfg, proc, data_to_set)


def scatterBitsByTit(tit):

    angle_colors = ['red', 'gold', 'yellowgreen', 'blueviolet']
    angles = ['temporal', 'inferior', 'nasal', 'superior']
    
    if 'AO001R' in tit:
        marker='o'
    if 'AO008R' in tit:
        marker='^'
        
    for a_ind, a in enumerate(angles):
    
        if a in tit:
            color = angle_colors[a_ind]
    
    if not color:
        color = 'k'
        
    return marker, color


def getFromCat(var, category):
    #print(category.keys())
    for cat in category.keys():
        if var in category[cat]:
            #print(category[cat])
            return cat, category[cat].index(var)
    if not var:
        return []


def unpackParams(user_param):
    # unpack critical variables 
    category = user_param['category_data']
    coord_unit = user_param['coord_unit']
    fl_name = user_param['fl_name']
    index = user_param['index']
    
    return [category, coord_unit, fl_name, index]


def getFileIndsByVar(process,var_set,category,index):
     #find the files to read the data from
        fl_inds_to_get = {}
        id_str = process + ' '
        
        for ind, var in enumerate(var_set):
            try:
                cat, ind = getFromCat(var, category) 
            except:
                print('requested var ' + var + ' is not specified by any category')
                
            if ind == 0:
                id_str = id_str + " " + var
            else: 
                id_str = id_str + " + " + var
                
            var_inds = np.nonzero(index[cat]==ind)[0]
            
            if not cat in fl_inds_to_get.keys():
                fl_inds_to_get[cat] = var_inds
            else: 
                fl_inds_to_get[cat] = np.append(fl_inds_to_get[cat], var_inds)
        
        temp = []
        for key in fl_inds_to_get.keys():
            if len(temp) == 0:
                temp = fl_inds_to_get[key]
            else:
                temp = np.intersect1d(temp,fl_inds_to_get[key])

        fl_inds_to_get = temp
        
        return fl_inds_to_get, id_str


def getParam(user_param, sav_cfg):

    waitlist = [
                'mosaic',
                'fl_name',
                'index'
                ] # make sure that parameters in this list are set to user_parm downstream


    #Load in the configuration file that determines the Single_mosaic_processes save file structure

    category = ['subject', 'angle', 'eccentricity', 'conetype', 'conetype_color']

    category_data = {}
    for cat in category:
        # print(user_param[cat])
        category_data[cat] = user_param[cat][0]

    user_param['category_data'] = category_data

    user_param['data_path'] = user_param['save_path']

    mosaic, fl_name, index = flsyst.getFilesByDataGroup(user_param, '.hdf5')

    for p in waitlist:
        user_param[p] = locals()[p]
        
    return user_param


def getMetric(variables, user_param, process, metric, string=False):
    [category, coord_unit, fl_name, index] = unpackParams(user_param)
    """
    LOOKS LIKE THIS WILL FAIL IF THERE'S MORE THAN ONE VARIABLE SET
    """

    for var_set in variables:
        [fl_inds_to_get, id_str] = getFileIndsByVar(process, var_set, category, index)
        output = []
        for fl in fl_inds_to_get:
            with h5py.File(fl_name[fl], 'r') as file:
                ## COMMENTED THIS OUT 8/5/24 TO REPLACE WITH THE ABOVE, WILL ALMOST CERTAINLY 
                ## MESS UP SOMETHING WITH THE OUTPUT OF LISTS OF SIZE "ONE", FIX IN RECURSIVE DATA
                ## LOADING FUNCTION IF THIS IS THE CASE

                if string:
                    output.append(bytes(file[process][metric][()]).decode("utf8"))
                    # print(file[process][metric][()])
                    # print(output[len(output) - 1])
                elif isinstance(file[process][metric], h5py._hl.group.Group):
                    output.append(flsyst.recursive_load_list_or_dict(file[process][metric]))
                elif type(file[process][metric][()]) == list:
                    if len(file[process][metric][()]) == 1:
                        output.append(file[process][metric][()][0])
                else:
                    output.append(np.squeeze(file[process][metric][()]))
                    # output.append(file[process][metric][()])        
        try:
            #get rid of dtype=object if this starts throwing errors, put it there because of a deprecation warning about making ndarrays from ragged nested sequences
            output = np.array(output)
        except: 
            output = output
        return output


def getOutputs(toRun, variables, user_param):
    outputs = {}
    for pair in toRun:

        process = pair[0]
        metric = pair[1]
        if len(pair) == 3:
            string = True 
        else:
            string = False
        # print(user_param.keys())
        outputs[process + '__' + metric] = getMetric(variables, user_param, process, metric, string=string)
        # if metric == 'subject':
        #     print('SUBJECT')
        #     print(outputs[process + '__' + metric])    
    return outputs

