import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import mosaic_topog.utilities as util
import yaml
import h5py
import csv

def save_list_to_hdf5(h5file, nested_list, path='/'):
    # GPT code. not tested yet. 
    def recursive_save_list(h5group, nested_list):
        for i, item in enumerate(nested_list):
            if isinstance(item, list):
                subgroup = h5group.create_group(str(i))
                recursive_save_list(subgroup, item)
            else:
                h5group.create_dataset(str(i), data=item)
                
    recursive_save_list(h5file.create_group(path), nested_list)

# def load_list_from_hdf5(h5file, path='/'):
#     # GPT code. not tested yet. 
#     def recursive_load_list(h5group):
#         result = []
#         for key in sorted(h5group.keys(), key=lambda x: int(x)):
#             if isinstance(h5group[key], h5py.Group):
#                 result.append(recursive_load_list(h5group[key]))
#             else:
#                 result.append(h5group[key][()])
#         return result
    
#     return recursive_load_list(h5file[path])


# def load_list_from_hdf5(h5file, path='/'):
#     def recursive_load_list(h5group):
#         print('load list')
#         print(h5group.keys())
        
#         keys = sorted(h5group.keys(), key=lambda x: int(x))
#         result = []
        
#         for key in keys:
#             if isinstance(h5group[key], h5py.Group):
#                 result.append(recursive_load_list(h5group[key]))
#             else:
#                 result.append(h5group[key][()])
#         return result
#     return recursive_load_list(h5file[path])


def save_dict_to_hdf5(h5file, dictionary, path='/'):
    # GPT code. not tested yet. 
    for key, value in dictionary.items():
        full_path = f"{path}{key}"
        if isinstance(value, dict):
            group = h5file.create_group(full_path)
            save_dict_to_hdf5(h5file, value, path=f"{full_path}/")
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(full_path, data=value)
        else:
            # Convert other data types to string representation
            h5file.create_dataset(full_path, data=str(value))


# def load_dict_from_hdf5(h5file, path='/'):
#     # GPT code. not tested yet. 
#     result = {}
#     keys = h5file[path].keys()
#     numkeys = len(keys)
#     keystr = [key for key in keys]
#     print(keystr)
#     print([keystr[z] == str(z) for z in np.arange(0,numkeys)])
#     if np.all([keystr[z] == str(z) for z in np.arange(0,numkeys)]):
#         result[key] = load_list_from_hdf5(h5file, path="/")
#     else:   
#         for key in h5file[path].keys():
#             print(key)
#             full_path = f"{path}{key}"
#             if isinstance(h5file[full_path], h5py.Group):
#                 result[key] = load_dict_from_hdf5(h5file, path=f"{full_path}/")
#             else:
#                 result[key] = h5file[full_path][()]
#                 # Convert back from bytes if necessary
#                 if isinstance(result[key], bytes):
#                     result[key] = result[key].decode('utf-8')
#     return result 

def is_sequential_int_keys(keys):
    try:
        int_keys = list(map(int, keys))
        return int_keys == list(range(len(int_keys)))
    except ValueError:
        return False

def loadByType(obj):
    # print(type(obj))
    if isinstance(obj, h5py.Group):
        # print('toload is group')
        loaded = recursive_load_list_or_dict(obj) 
    elif (isinstance(obj[()],  np.bytes_) or
            isinstance(obj[()],  bytes)):
        # print(type(obj[()]))
        try:
            loaded = [bytes(z).decode("utf8") for z in obj]
            # print('toload is bytes list')
        except:
            try:
                loaded = bytes(obj[()]).decode('utf8')
                # print('toload is nonlist bytes')
            except:
                loaded = obj[()]
                # print('toload is bytes problems???')
    else:
        # print('toload is ' + str(type(obj[()])))
        loaded = obj[()]

    return loaded


def recursive_load_list_or_dict(h5group):
    try:
        keys = sorted(h5group.keys(), key=int)
    except ValueError:
        keys = sorted(h5group.keys())
    # keys = sorted(h5group.keys(), key=lambda x: int(x))
    if is_sequential_int_keys(keys):
        result = []
        # print('output as list')
    else:
        result = {}
        # print('output as dict')
    
    for key in keys:
        # print(type(h5group[key][()])) 
        toload = loadByType(h5group[key])
        if is_sequential_int_keys(keys):
            result.append(toload)
        else:
            result[key] = toload
        
    return result

    # def load_dataset:
    #     if isinstance(h5group[key], h5py.Group):
    #         result[key] = recursive_load_list_or_dict(h5group[key])
    #     elif isinstance(value,  np.bytes_):
    #         result[key] = [bytes(z).decode("utf8") for z in h5group[key]]
    #     elif isinstance(value,  bytes):
    #         result[key] = [bytes(z).decode("utf8") for z in h5group[key]]
    #     else:
    #         result[key] = h5group[key][()]



def saveOutputsAsCSV(output_savename, outputs):
    with open(output_savename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key in outputs.keys():
            writer.writerow([key])
            writer.writerow(outputs[key])

def getAllConeDat(sav_fl, mosaic, spec=[], sep_folder="nah", use_cut_in_coord = False):
    all_dat = []
    save_path = os.path.dirname(sav_fl)
    needhelp = False
    if not sep_folder == "nah":
        all_coord_fl = sep_folder + mosaic + '_all.hdf5'
    else:
        all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5'    

    try: 
        with h5py.File(all_coord_fl, 'r') as file:
            for sp in spec:
                try:
                    # if sp[1] == 'cone_coord':
                    #     if use_cut_in_coord:
   
                    #         cut_in_coord = file['input_data']['cut_in_coord']
                    #         all_dat.append(cut_in_coord)
                    #     else:
                    #         all_dat.append(file[sp[0]][sp[1]][()])
                    # else:
                    all_dat.append(file[sp[0]][sp[1]][()])
                except:
                    print('no spec "' + sp + 'found in ' + all_coord_fl)
    except:
        print("could not access all-cone file in: " + sep_folder)
        print("attempting to pull it from the save folder instead....")
        all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5' 
        try: 
            with h5py.File(all_coord_fl, 'r') as file:
                for sp in spec:
                    try:
                        all_dat.append(file[sp[0]][sp[1]][()])
                    except:
                        print('no spec "' + sp + 'found in ' + all_coord_fl)
        except:
            print("could not access all-cone file in " + all_coord_fl + ', either')
            

    # except:
    #     print('could not pull total cones coordinates from ' + all_coord_fl)
    return all_dat


def getAllConeCoord(sav_fl, mosaic):
     # should replace this function everywhere with getAllConeDat
     # look for all cone mosaic for this data
    save_path = os.path.dirname(sav_fl)
    all_coord_fl = save_path + '\\' + mosaic + '_all.hdf5'
    try:
        with h5py.File(all_coord_fl, 'r') as file:
            all_coord = file['input_data']['cone_coord'][()]
                
    except:
        print('could not pull total cones coordinates from ' + all_coord_fl)
    
    return all_coord


def setProcessByType(file, proc, var, data, prefix=''):
    """
    sets individual vars of a process in a read-in .hdf5 file open for writing or reading+writing

    """
    
    proc_to_set = prefix + proc
    if isinstance(data, str):
        file[proc_to_set][var] = np.string_(data)
    elif isinstance(data, float):
        file[proc_to_set][var] = np.float_(data)
    elif isinstance(data, int):
        file[proc_to_set][var] = np.int_(data)
    elif isinstance(data, bool):
        file[proc_to_set][var] = np.bool_(data)
    elif isinstance(data, dict):
        # file[proc_to_set][var] = data
        save_dict_to_hdf5(file, data, path = '/' + proc_to_set + '/' + var + '/')
    elif isinstance(data, list):
        try:
            save_list_to_hdf5(file, data, path = '/' + proc_to_set + '/' + var + '/')
            
            # UNCOMMENT HERE IF SMP STARTS BREAKING IN THIS TRY/CATCH LOOP, 
            # ADDED THIS FUNCTIONALITY FOR PSMP AND HAVEN'T TESTED WHETHER 
            # IT BREAKS SMP YET

            # if not data:
            #     file[proc_to_set][var] = np.array(data)
            # elif isinstance(data[0], str):
            #     data = [n.encode('utf8', 'ignore') for n in data]
            #     file[proc_to_set][var] = np.array(data)
            # else:
            #     try:
            #         file[proc_to_set][var] = np.ndarray(data)
            #     except:
            #         file[proc_to_set][var] = np.array([data])
        except TypeError:
            print('')
            print(var)
            print(data)
            print(type(data))
            print('')
            print('still not handling lists well in flsyst.setProcessByType, need to go fix')
    elif isinstance(data, np.ndarray) or isinstance(data, np.int32):
        try:
            file[proc_to_set][var] = data
        except TypeError:
            # print(proc_to_set)
            # print(var)
            print('GAH something weird about ndarrays in flsyst.setproc_to_setessByType')
            file[proc_to_set][var] = np.float_(data.astype('float64'))   
    else:
        print('data for "' + var + '" is type "' + str(type(data)) + '" and flsyst.setproc_to_setessByType() does not know what to do.')
        print('something is wrong with the data being sent to proc_to_setess:'+proc_to_set+', variable:'+var+', OR')
        print('a condition for this type needs to be addded to flsyst.setproc_to_setessByType().')


def setProcessVarsFromDict(param, sav_cfg, proc, data_to_set, spec='all', prefix=''):
    """
    this is called by process functions to initiate the data-saving process
    for that function after it has run
    """
    
    sav_fl = param['sav_fl']

    with h5py.File(sav_fl, 'r+') as file:

        # find all variables for this process
        proc_vars = sav_cfg[proc]['variables']

        if spec == 'all':
            # set spec to equal all variables
            spec = proc_vars

        # check if requested variables for this process are valid for our configuration file
        for var in spec:
            if var not in proc_vars:
                print('variable"' + var + '" being set is not found in the configuration file for process "' + proc + '". it will not be set.')
                del spec[spec.index(var)]

        # stash all the variables that have already been set for this
        # process.  they will only be overwritten if the variable is
        # specified in spec
        # delete the process key in the save file if it already exists 
        proc_to_set = prefix + proc
        
        if proc_to_set in file.keys():
            temp_vars = file[proc_to_set]
            del file[proc_to_set]
        else:
            temp_vars = []

        # check if variables found in the save file for this process are valid for our configuration file
        for var in temp_vars:
            if var not in proc_vars:
                print('variable "' + var + '" found in the save file is not found in the configuration file for process "' + proc + '". it will be removed from the save file.')
                del temp_vars[var]

        # create process key in the save file
        if (proc_to_set) in file.keys():
            del file[proc_to_set]
        file.create_group(proc_to_set)
  
        # set variables
        for var in proc_vars:
            if var in spec:
                setProcessByType(file, proc, var, data_to_set[var], prefix=prefix)
            elif var in temp_vars:
                setProcessByType(file, proc, var, temp_vars[var], prefix=prefix)
            else:
                print('check for variables failed, go look for the bug earlier in this function')



def saveNameFromLoadPath(fls, save_path, load_type='.csv', save_type='.hdf5'):
    """

    """
    save_name = [
        save_path + os.path.splitext(os.path.basename(fl))[0] + save_type for fl in fls
    ]
    return(save_name)


def readYaml(flnm):
    # load in the config file that dictates the information that can be saved by mosaic_topog
    with open(flnm, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            print('bad day for config file: ' + flnm)
    return cfg


# def getConeData(fold_path, user_param, filetype):

#     """
#     Get cone data based on mosaic and datatype specifications

#     Pulls cone data from a single folder based on the file specifications
#     indicated by the user, and creates indexes that define all pulled data in
#     terms of subject, angle, eccentricity, filetype, and [if applicable] the
#     subset of cones. Supports cone coordinates as .csv files and cone images
#     as .tiffs.

#     Looks in the input folder for every permutation of the input
#     lists in the following format (text between asterisks only applies to
#     coordinate data):

#         '[subject]_[angle]_[eccentricity]*_[conetypes]_*[filetypes]'

#     Parameters
#     ----------
#     fold_path : str
#         Path to folder containing all data files to be loaded in.
#         The path must end with a slash. Note the syntax for reserved
#         characters in your path, such as double-slashes for single-
#         slashes.
#         e.g. 'C:\\Users\\schle\\Documents\\GitHub\\Incubator-2022-Geometry-
#               of-Color\\CSV data all OCT locs Jan 2022\\'
#     subject : list of str
#         List of subjects to search for
#         e.g. ['AO001R','AO008R']
#     angle : list of str
#         List of angles from the fovea to search for
#         e.g. ['temporal','nasal']
#     eccentricity : list of float
#         List of eccentricities from the fovea to search for
#         e.g. [1.5, 4, 10]
#     conetypes : list of str
#         List of angles from the fovea to search for
#         e.g. ['all','L','M','S','unclassed']
#     filetype : {'.csv','.tiff'}
#         List of filetypes to search for. only .csv and .tiff are supported.

#     Returns
#     -------
#     data : numpy.ndarray
#         a list of np.arrays corresponding to cone coordinates (filetype
#         = '.csv') or pyplot images of cone mosaics (filetype ='.tiff')
#     mosaics : list of str
#         list of datasets of origin collected
#     index : dict
#         'subj','ang','ecc','contyp' : np.array
#             {'subj', 'ang', 'ecc', 'contyp'} are indexing vectors that
#             correspond to input variables [subject, angle, eccentricity,
#             conetypes], respectively, such that the nth value of all
#             vectors indicate the descriptors of the nth piece of data
#             in the list returned.
#         'mos' : numpy.ndarray
#             index vector like the other keys but corresponds to the
#             returned variable 'mosaics'
#     flnames_all : list of str

#     """
#     mosaic, flnames_all, index = getFilesByDataGroup(fold_path, user_param, filetype)

#     data = []

#     for ind, fl in enumerate(flnames_all):

#         if filetype == '.csv':
#             # load in cone coordinates (relative to ROI lower left corner)
#             data.append(np.loadtxt(fl, delimiter=','))

#         elif filetype == '.png':
#             # load in ROI image
#             data.append(plt.imread(fl))

#     data = np.asarray(data)

#     return data, mosaic, index, flnames_all


def getAllOfCat(cat, fl_list):
    allofcat = []
    for fl in fl_list:
        split = fl.split('_')
        if split[cat] not in allofcat:
            allofcat.append(split[cat])

    return allofcat


def getFilesByDataGroup(user_param, filetype, getAll = False):
    """
    Get cone data based on mosaic and datatype specifications

    Support function of getConeData. Returns are identical to getConeData
    except that instead of loaded data, 'data' contains the filenames in
    the input 'fold_path' corresponding to the users specifications

    Parameters
    ----------
    path : str
        Path to folder containing all data files to be loaded in.
        The path must end with a slash. Note the syntax for reserved
        characters in your path, such as double-slashes for single-
        slashes.
        e.g. 'C:\\Users\\schle\\Documents\\GitHub\\Incubator-2022-Geometry-
              of-Color\\CSV data all OCT locs Jan 2022\\'
    subject : list of str
        List of subjects to search for
        e.g. ['AO001R','AO008R']
    angle : list of str
        List of angles from the fovea to search for
        e.g. ['temporal','nasal']
    eccentricity : list of float
        List of eccentricities from the fovea to search for
        e.g. [1.5, 4, 10]
    conetypes : list of str
        List of angles from the fovea to search for
        e.g. ['all','L','M','S','unclassed']
    filetype : {'.csv','.tiff'}
        List of filetypes to search for. only .csv and .tiff are supported.

    Returns
    -------
    mosaics : list
        all unique combinations of the subjects, angles, and eccentricities
        (= all datasets)
    cat_comb : np.array
        all unique combinations of mosaics and conetypes (if applicable) for
        this file type (= all files read in)
    cat_index : dict of np.array
        vectors that index the subject, angle, eccentricity, conetype, and
        mosaic in cat_comb

    """
    # DON'T USE THE GETALL FEATURE - DIDN'T THINK IT THROUGH ALL THE WAY, IT DOESN'T WORK

    folder = user_param['data_path'][0]

    # get all file paths in the directory
    fl_list = glob.glob(folder + '*')
    fl_list = [os.path.basename(file) for file in fl_list]  # listcomprehension


    #ADD ALLFILE READ-IN OPTION HERE
    if not getAll:

        subject = user_param['subject'][0]
        angle = user_param['angle'][0]
        eccentricity = user_param['eccentricity'][0]
        conetype = user_param['conetype'][0]

        mosaic = []
        mos_fls = []
        mos_count = 0
        fl_count = 0

        # build the file paths I want based on the category inputs
        delim = '_'
        cat_comb = []  # will contain all the permutations of categories

        cats = [subject, angle, eccentricity, conetype] # MUST BE IN ORDER OF THE FILE SPLIT
        for c_ind, c in enumerate(cats):
            if c[0] == 'getAll':
                cats[c_ind] = getAllOfCat(c_ind, fl_list)
        
        subject = cats[0]
        angle = cats[1]
        eccentricity = cats[2]
        conetype = cats[3]

        for s in subject:
            for a in angle:
                for e in eccentricity:
                    mosaic.append(s + delim + a + delim + e)
                    mos_fls.append([])
                    if filetype == '.csv' or filetype == '.hdf5':
                        for c in conetype:  # only look for conetype specific data
                            # if this is coordinate data
                            cat_comb.append(s + delim + a + delim + e + delim
                                            + c + filetype)
                            mos_fls[mos_count].append(fl_count)
                            fl_count = fl_count+1
                    elif filetype == '.png':
                        cat_comb.append(s + delim + a + delim + e + '_raw' +
                                        filetype)
                        mos_fls[mos_count].append(fl_count)
                        fl_count = fl_count + 1
                    mos_count = mos_count + 1

        # search for if those files exist.  if they don't, remove them from my list
        

        pop_fls = util.indsNotInList(cat_comb, fl_list)
        cat_comb = util.removeListInds(cat_comb, pop_fls)

        # if any of the mosaics we looked for have no files associated with them, remove them from the mosaics list
        pop_mos = []
        for ind, mos in enumerate(mos_fls):
            files_kept = util.indsNotInList(mos, pop_fls)
            if not files_kept:
                pop_mos.append(ind)
        mosaic = util.removeListInds(mosaic, pop_mos)
    else:

        hdf5_list = [f for f in fl_list if 'hdf5' in f]
        
        cat_comb = hdf5_list
        mosaic = []
        subject = []
        angle = []
        eccentricity = []
        conetype = []
        for f_ind, fl in enumerate(hdf5_list):
            len_fl_suff = len(filetype)
            mosaic.append(fl[0:len(fl)-len_fl_suff])
            split = fl.split('_')
            subject.append(split[0])
            angle.append(split[1])
            eccentricity.append(split[2])
            conetype.append(split[3])

    # get substrings of filenames that indicate the file's value for each
    # category (so that we can use them to create indexes for each category)
    [subj_str, ang_str, ecc_str, end_str] = getFileSubstrings(cat_comb)

    # define the categories to be indexed the substrings to reference to get
    # the indexes
    categories = [subject, angle, eccentricity, conetype, mosaic]
    cat_names = ['subject', 'angle', 'eccentricity', 'conetype', 'mosaic']
    cat_str = [subj_str, ang_str, ecc_str, end_str, cat_comb]



    # get category vectors to output
    cat_index = getIndexes(categories, cat_str, cat_names)


    for ind, fl in enumerate(cat_comb):
        cat_comb[ind] = folder + fl

    return mosaic, cat_comb, cat_index



def getFileSubstrings(fl_list):
    """
    breaks filenames into strings that indicate the subject, angle,
    eccentricity, conetype (if applicable), and filetype of the data
    in the file
    (support function of getConeData)

    PARAMETERS
    ----------
    fl_list : list of str
        full paths to files

    RETURNS
    -------
    flnames_all : list of str
        filenames without their preceding path
    subj_str : list of str
        subject of each data file
    ang_str : list of str
        angle of each data file
    ecc_str : list of str
        eccentricity of each data file
    end_str : list of str
        conetype or image indication and filetype of each data file

    """
    flnames_all = []
    subj_str = []
    ang_str = []
    ecc_str = []
    end_str = []

    for fl in fl_list:
        flnm = os.path.basename(fl)
        flnames_all.append(flnm)
        flsplit = flnm.split('_')
        subj_str.append(flsplit[0])
        ang_str.append(flsplit[1])
        ecc_str.append(flsplit[2])
        end_str.append(flsplit[len(flsplit)-1])

    return subj_str, ang_str, ecc_str, end_str


def getIndexes(categories, cat_str, cat_names):
    """
    Retrieves file index vectors for all the input categories and
    stores them in a dictionar

    (support function of getConeData)

    Parameters
    ----------
    categories : list of str
    cat_str : list of str

    Returns
    -------
    cat_index : dict of np.array

    """
    cat_index = {}

    for ind, cat_name in enumerate(cat_names):
        vect = getIndex(categories[ind], cat_str[ind])
        cat_index[cat_name] = vect

    return cat_index


def getIndex(cat_vals, fl_strings):
    """
    (support function of getConeData)

    Parameters
    ----------
    strings : list of str
    category : list of str

    Returns
    -------
    index : np.array

    """

    directory = []

    index = np.zeros(len(fl_strings))
    index = index.astype(int)
    index[:] = -1

    for ind, c in enumerate(cat_vals):

        directory.append(np.nonzero(np.char.find(np.asarray(fl_strings),
                                    c) > -1)[0])
        index[directory[ind]] = int(ind)

    return index


def checkFileForProcess(proc, sav_cfg, save_name, collect_all):
    # make sure the category is valid
    if proc not in sav_cfg.keys():
        print('element "' + str(proc) + '" in variable "proc_to_run" is not a process specified in the configuration file. skipping.')
    else:
        
        # select only files missing data for the requested category
        collect_this = []
        for ind, fl in enumerate(save_name):
            if ind not in collect_all:
                with h5py.File(save_name[ind], 'r') as temp:
                    if proc not in temp.keys():
                        collect_this.append(ind)
                    else:
                        for v in sav_cfg[proc]['variables']:
                            if v not in temp[proc].keys():
                                collect_this.append(ind)

        # get a sorted list of files missing entirely and files missing data from the requested category
        collect_this = np.union1d(collect_all, collect_this).astype(int)


def fileInfoToUserParams(user_param, sav_cfg):
    """
    pulled this from the smp jupyter notebook, not positive this is the best place for it
    
    set up for "user_param," a dictionary that is passed to the process manager that
    stores all the variables determined from the user's inputs.
    """
    # most of these parameters have been set directly by the user, 
    # a few need to be interpreted from the filesystem, which go into waitlist. 
    waitlist = [
                'mosaic',
                'coord_fl_name',
                'coord_index',
                'img_fl_name',
                'img_index',
                'save_name',
                'processes'
                ] # make sure that parameters in this list are set to user_parm downstream


    # fl_name is the output variable we care about for now -
    #     it's a list of all files found for combinations of the input parameters.

    # mosaic is a list of all the classified mosaics the located data come from
    # for instance a single mosaic may be mapped to mosaic coordinate files for 
    # L, M, S, 'all', and 'unclassed' cone sub-mosaics 

    # index is a dictionary for indexing through the list flnames according to:
    #     inputs to single_mosaic_processes - subject, angle, eccentricity, conetype_ids 
    #     determined by single_mosaic_processes - 

    mosaic, coord_fl_name, coord_index = getFilesByDataGroup(user_param, user_param['coord_fltype'][0])
    img_mosaic, img_fl_name, img_index = getFilesByDataGroup(user_param, user_param['img_fltype'][0])

    # Make list of expected save-file names for this data
    save_name = saveNameFromLoadPath(coord_fl_name,user_param['save_path'][0])


    cats = [user_param['subject'], 
            user_param['angle'], 
            user_param['eccentricity'],
            user_param['conetype']] # MUST BE IN ORDER OF THE FILE SPLIT

    for c_ind, c in enumerate(cats):
        if c[0][0] == 'getAll':
            cats[c_ind] = [getAllOfCat(c_ind, mosaic)]
            
    user_param['subject'] = cats[0]
    user_param['angle'] = cats[1]
    user_param['eccentricity'] = cats[2]
    user_param['conetype'] = cats[3]

    if user_param['data_to_run'][0] == 'all':
        processes = {}
        for proc in user_param['sim_to_gen'][0]:
            processes[proc] = np.arange(0, len(coord_fl_name)).tolist()
        for proc in user_param['analyses_to_run'][0]:
            processes[proc] = np.arange(0, len(coord_fl_name)).tolist()
            for sim in user_param['sim_to_gen'][0]:
                if sav_cfg[proc]['analysis_tier'] == 1:
                    processes[sim + '_' + proc] = np.arange(0, len(coord_fl_name)).tolist()
    # elif data_to_run == 'absent_only': # ABSENT_ONLY OPTION NOT FUNCTIONAL ANYMORE, NEEDS FIXED
    #     processes = flsyst.getProcessesToRun(save_name, user_param['save_path'], user_param['sim_to_gen'][0], user_param['analyses_to_run'][0], sav_cfg)
        
    # print(user_param['sim_to_gen'])
    # print(user_param['analyses_to_run'])

    for p in waitlist:
        user_param[p] = vars()[p]

    return user_param


def getProcessesToRun(save_name, save_path, sim_to_gen, analyses_to_run, sav_cfg):
    """
    Determine which mosaics will be run through which processes

    Mosaics are selected to be run on all processes if there is not already an .hdf5 in the save folder
    that describes this mosaic.  If there is an .hdf5 for this file, mosaics are only selected to be run
    on requested processes that are not already present in the file.

    Modifications that will make this mroe useful down the line:
        - Add a name to the savefl_config keys file that clarifies this is the single_mosaic_processes save file, 
          and make the check for whether the savefile exists check for this
        - Add a git-commit-hash key to each process in the save file so that when processes are checked in found 
          save files, the data can be flagged to be re run if it is found but was not run on the current version

    Parameters
    ----------
    fl_names : list of str
        mosaic .csv filenames, path not included
    save_name: list of str
        single_mosaic_process .hdf5 filenames, path included.
        corresponds to fl_names
    proc_to_run: list of str
        processes requested by the user
    sav_cfg : dict
        map organizing single_mosaic_processes, read in from configuration file
    

    Returns
    -------
    processes : dict of list of int
        keys are comprised of 1) mandatory keys for data run & saved by single_mosaic_processes 
                              2) processes requested by user in the input "data_to_run"
        lists returned for each key/process are int indices corresponding to data files in the list "flnames" 
        that need to be run through that process  

    Raises
    ------
    

    """

    # look for files associated with each data file in the save folder
    # If no data file is found, add it to the list to collect all data for
    folder_hdf5s = glob.glob(save_path + '*.hdf5')  # get all .hdf5 files in the save path
    collect_all = util.indsNotInList(save_name, folder_hdf5s)

    # initialize map between categories and files to be run on them
    processes = {}

    # for each category of processes, identify files that will be run through 
    for proc in sim_to_gen:
        processes[proc] = checkFileForProcess(proc, sav_cfg, save_name, collect_all)

    for proc in analyses_to_run: 
        processes[proc] = checkFileForProcess(proc, sav_cfg, save_name, collect_all)
        
    return processes


def printSaveFile(sav_fl):
    print(sav_fl)
    if os.path.exists(sav_fl):
        with h5py.File(sav_fl, 'r') as file:
            # print(file['measured_intracone_dist']['bin_edge'][()])
            for key in file.keys():
                key_div = '***************************************'
                print(key_div + key + key_div)
                for var in file[key]:
                    print(var)
                    if isinstance(file[key][var], h5py._hl.group.Group):
                        data = recursive_load_list_or_dict(file[key][var])
                        if isinstance(data, dict):
                            print(data.keys())
                            # util.printDict(data)
                        elif isinstance(data, list):
                            print(len(data))
                            # util.printList(data)
                    elif isinstance(file[key][var][()], np.bytes_):
                        data = bytes(file[key][var][()]).decode("utf8")
                        try:
                            print('    ' + data)
                        except:
                            print(type(data))
                            print('-')
                    else:
                        data = file[key][var][()]
                        try:
                            print('     ' + str(data.shape))
                        except:
                            print(type(data))
                            print('-')
                    print('')
    else:
        print('printSaveFile cannot print "' + sav_fl + '" because it does not exist')
