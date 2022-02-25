import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import mosaic_topog.utilities as util

# Functions
# ---------
# getConeData
# getFilesByDataGroup
# getFileSubstrings
# getIndexes
# getIndex


def getConeData(fold_path, subject, angle, eccentricity, conetypes,
                filetype):

    """
    Get cone data based on mosaic and datatype specifications

    Pulls cone data from a single folder based on the file specifications
    indicated by the user, and creates indexes that define all pulled data in
    terms of subject, angle, eccentricity, filetype, and [if applicable] the
    subset of cones. Supports cone coordinates as .csv files and cone images
    as .tiffs.

    Looks in the input folder for every permutation of the input
    lists in the following format (text between asterisks only applies to
    coordinate data):

        '[subject]_[angle]_[eccentricity]*_[conetypes]_*[filetypes]'

    Parameters
    ----------
    fold_path : str
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
    data : list
        a list of np.arrays corresponding to cone coordinates (filetype
        = '.csv') or pyplot images of cone mosaics (filetype ='.tiff')
    mosaics : list of str
        list of datasets of origin collected
    index : dict
        'subj','ang','ecc','contyp' : np.array
            {'subj', 'ang', 'ecc', 'contyp'} are indexing vectors that
            correspond to input variables [subject, angle, eccentricity,
            conetypes], respectively, such that the nth value of all
            vectors indicate the descriptors of the nth piece of data
            in the list returned.
        'mos' : np.array
            index vector like the other keys but corresponds to the
            returned variable 'mosaics'

    """

    mosaics, flnames_all, index = getFilesByDataGroup(fold_path, subject,
                                                      angle, eccentricity,
                                                      conetypes, filetype)

    data = []

    for ind, fl in enumerate(flnames_all):

        if filetype == '.csv':
            # load in cone coordinates (relative to ROI lower left corner)
            data.append(np.loadtxt(fold_path + fl, delimiter=','))

        elif filetype == '.png':
            # load in ROI image
            data.append(plt.imread(fold_path + fl))

    data = np.asarray(data)

    return data, mosaics, index


def getFilesByDataGroup(path, subject, angle, eccentricity, conetypes,
                        filetype):
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
    # get all file paths in the directory
    fl_list = glob.glob(path + '*')

    fl_list = [os.path.basename(file) for file in fl_list]  # listcomprehension

    mosaics = []
    mos_fls = []
    mos_count = 0
    fl_count = 0

    # build the file paths I want based on the category inputs
    delim = '_'
    cat_comb = []  # will contain all the permutations of categories
    
    for s in subject:
        for a in angle:
            for e in eccentricity:
                mosaics.append(s + delim + a + delim + e)
                mos_fls.append([])
                if filetype == '.csv':
                    for c in conetypes:  # only look for conetype specific data
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
    mosaics = util.removeListInds(mosaics, pop_mos)

    # get substrings of filenames that indicate the file's value for each
    # category (so that we can use them to create indexes for each category)
    [subj_str, ang_str, ecc_str, end_str] = getFileSubstrings(cat_comb)

    # define the categories to be indexed the substrings to reference to get
    # the indexes
    categories = [subject, angle, eccentricity, conetypes, mosaics]
    cat_names = ['subject', 'angle', 'eccentricity', 'conetypes', 'mosaics']
    cat_str = [subj_str, ang_str, ecc_str, end_str, cat_comb]

    # get category vectors to output
    cat_index = getIndexes(categories, cat_str, cat_names)

    print('found ' + str(len(cat_comb)) + ' files')
    print('')

    cat_comb = np.asarray(cat_comb)
    return mosaics, cat_comb, cat_index


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
    stores them in a dictionary

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


def getDataByCatVal(data, index, val):
    """

    """
    data_by_ind = data[np.nonzero(index == val)[0]]

    return data_by_ind
