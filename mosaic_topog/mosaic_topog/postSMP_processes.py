import numpy as np
import h5py
import mosaic_topog.flsyst as flsyst


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
                if string:
                    output.append(bytes(file[process][metric][()]).decode("utf8"))
                else:
                    #print(type(file[process][metric][()]))
                    # print(process + ' ' + metric)
                    if type(file[process][metric][()]) == list:
                        if len(file[process][metric][()]) == 1:
                            output.append(file[process][metric][()][0])
                    else:
                        output.append(np.squeeze(file[process][metric][()]))
                        # output.append(file[process][metric][()])
                    
        try:
            output = np.array(output)

        except: 
            print('Warning: ' + metric + ' contains datasets of irregular size and were output as a list')
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
        outputs[process + '__' + metric] = getMetric(variables, user_param, process, metric, string=string)
            
    return outputs

