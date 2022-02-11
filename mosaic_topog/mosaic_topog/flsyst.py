import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def getFileStrings(fl_list):
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

    flnames_all = np.array(flnames_all)

    return flnames_all, subj_str, ang_str, ecc_str, end_str

def getIndices(strings,category):

    directory = {}
    code = {}
    vector = np.zeros(len(strings))
    vector = vector.astype(int)
    vector[:] = -1

    for ind,c in enumerate(category):
        if not isinstance(c, str):
            c = str(c)
            c = str(np.core.defchararray.replace(c,'.0',''))
        directory[c] = np.nonzero(np.char.find(np.asarray(strings),c)>-1)
        code[c] = int(ind)
        vector[directory[c]] = int(ind)
        
    return directory, code, vector

def getFileVectors(categories,cat_str):

    file_indices = []
    cat_vectors = []
    intersection = []

    for ind,c in enumerate(categories):
        [direct,code,vect] = getIndices(cat_str[ind],categories[ind])
        file_indices.append([direct,code,vect])
        cat_vectors.append(vect)

        if not isinstance(intersection,list):
            intersection = np.intersect1d(intersection,np.nonzero(vect != np.nan)[0])
        else:
            intersection = np.nonzero(vect != np.nan)[0]

    return file_indices, cat_vectors, intersection

def getFilesByDataGroup(path,subject,angle,eccentricity,conetype_ids,filetypes):    
    
    #get all file paths in the directory
    fl_list = glob.glob(path + '*')
    
    fl_list = [os.path.basename(file) for file in fl_list] #list comprehension

    mosaics = []

    #build the file paths I want based on the category inputs
    delim = '_'
    cat_comb = []
    for s in subject:
        for a in angle:
            for e in eccentricity:
                e_str = str(e)
                e_str = str(np.core.defchararray.replace(e_str,'.0',''))
                mosaics.append(s + delim + a + delim + e_str)
                for f in filetypes:
                    if f == '.csv':
                        for c in conetype_ids:
                            cat_comb.append(s + delim + a + delim + e_str + delim + c + f) 
                    else:
                        cat_comb.append(s + delim + a + delim + e_str + '_raw' + f) 
    
    lookedfor = len(cat_comb)
    notfound = 0
    
    #search for if those files exist.  if they don't, remove them from my list.  
    for ind,name in enumerate(cat_comb):
        if name not in fl_list:
         #   print(name + ' not found')
            cat_comb.pop(ind)
            notfound = notfound + 1
            
    #print(str(notfound) + ' of ' + str(lookedfor) + ' filenames not found')
    #print(' ')
            
    #get file substring lists
    [cat_comb, subj_str, ang_str, ecc_str, end_str] = getFileStrings(fl_list)
    
    #get category indexes from the final list
    categories = [subject,angle,eccentricity,conetype_ids,filetypes,mosaics]
    cat_str = [subj_str,ang_str,ecc_str,end_str,end_str,cat_comb]    

    #get category vectors to output
    [file_indices, cat_vectors, intersection] = getFileVectors(categories,cat_str)
    
    print('found ' + str(len(cat_comb)) + ' files')
    print('')
    
    cat_comb = np.asarray(cat_comb)
    
    return cat_comb, cat_vectors[0], cat_vectors[1], cat_vectors[2], cat_vectors[3], cat_vectors[4], cat_vectors[5]




def getConeData(fold_path,subject,angle,eccentricity,conetype_ids,filetypes):

    flnames_all, subj_vect, ang_vect, ecc_vect, contyp_vect, fltyp_vect, mos_vect = getFilesByDataGroup(fold_path, subject, angle, eccentricity, conetype_ids, filetypes)

    csv_ind = np.array([ind  for ind,f in enumerate(filetypes) if f == '.csv'])
    png_ind = np.array([ind  for ind,f in enumerate(filetypes) if f == '.png'])
        
    data = []
    num_type = {}
    ROI = {}
        
    for ind,fl in enumerate(flnames_all):
        
        if np.array_equal(fltyp_vect[ind],csv_ind[0]):
            #load in cone coordinates (relative to ROI lower left corner)
            data.append(np.loadtxt(fold_path + fl, delimiter=','))
            #print(data[ind])
            num_type[ind] = data[ind].shape[0]
        else:     
            #load in ROI image
            data.append(plt.imread(fold_path + fl))
            ROI[ind] = [data[ind].shape[1],data[ind].shape[0]]

    return data, subj_vect, ang_vect, ecc_vect, contyp_vect, fltyp_vect, mos_vect