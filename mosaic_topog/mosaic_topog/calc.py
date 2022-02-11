import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

import mosaic_topog.show as show

#-------------------------------------------------------------------------------------------------------

# CALCULATE DISTANCE BETWEEN ALL CONES IN A MOSAIC
#   this needs to be edited to have a less gross solution for dealing with one
#   versus multiple sets of coordinates

def dist_matrices(coords):
    """
    function that returns a [num_cones,num_cones] shaped matrix of intercone distances and a vectorized sorted version of the same matrix. size of dim [2] of square matrix and dim [1] of vector will = size of dim [2] of the coordinate array input, so the operation can be performed on a "stack" of cone coordinates.
    """
    numzeros = coords.shape[0]
    if len(coords.shape) == 2:    # 1 COORDINATE ARRAY
    
        #2D array of every cone in the mosaic's distance from every other cone
        dist_square = spatial.distance_matrix(coords,coords)
        
        #turn ^ into a vector
        temp = np.reshape(dist_square[:,:], -1)
    
        #sort the vector
        dist_vect = np.sort(temp) 
        
        findzeros = np.nonzero(dist_vect==0)[0].size
        
        if numzeros != findzeros:
            print('ERROR: bad logic about distance = 0. lreal num zeros is ', findzeros , ', expected ', numzeros)
        dist_vect = dist_vect[numzeros+1:]    

        
    else:    # >1 COORDINATE ARRAY 
    
        #initialize 3D array for stack of square distance matrices and 2D array for columns of sorted cone distances
        dist_square = np.zeros([coords.shape[0],coords.shape[0],coords.shape[2]])
        dist_vect = np.zeros([pow(coords.shape[0],2),coords.shape[2]])
        
        for z in np.arange(0,coords.shape[2]):
        
            # fill w stack of 2D arrays of intercone distances
            dist_square[:,:,z] = spatial.distance_matrix(coords[:,:,z],coords[:,:,z])

            #turn ^ into 2D array with every column being an array of distances
            temp = np.reshape(dist_square[:,:,z], -1)

            #sort the vectors
            dist_vect[:,z] = np.sort(temp) 
            
        #remove zeros
        findzeros = np.nonzero(dist_vect[:,z]==0)[0].size
        if numzeros != findzeros:
            print('ERROR: bad logic about distance = 0. real # zeros is ', findzeros , ', expected ', numzeros)
        dist_vect = dist_vect[numzeros+1:,:]    

    return dist_square, dist_vect

#-------------------------------------------------------------------------------------------------------

#GENERATE A QUANTITY OF MONTE CARLO MOSAICS SUCH THAT CONES MAY
#OCCUPY ANY (X,Y) POSITION IN THE ROI
def MonteCarlo_uniform(num_mc,num_coords,xlim,ylim):
    """ 
        generate array of monte carlo shuffled points distributed uniformly within a 2D range
    """
    # [0] = cone index, [1] = (x,y), [2] = Monte Carlo trial
    mc_coords = np.zeros([num_coords,2,num_mc])
    
    # randomly sample coordinates from a uniform distribution of the the ROI space
    mc_coords[:,0,:] = np.random.uniform(low=xlim[0],high=xlim[1], size=(num_coords,num_mc))
    mc_coords[:,1,:] = np.random.uniform(low=ylim[0],high=ylim[1], size=(num_coords,num_mc))
    
    return(mc_coords)

#-------------------------------------------------------------------------------------------------------

#CALCULATE THE MEAN AND STD OF DATA GIVEN
#   needs to be edited as something that just plots 
#   a mean & std, and calculate the mean & std elsewhere
def quad_stats(x,plot_dict,xstd,unit,ids,colors):
    axes,fig = show.quad_fig([9,9])
    
    mean = {}
    std = {}
    
    for ind,id_str in enumerate(ids):
            mean[id_str] = plot_dict[id_str].mean(1)
            std[id_str] = plot_dict[id_str].std(1)
            
            axes[ind].set_facecolor('k')
            
            axes[ind].plot(x[id_str],mean[id_str],color=colors[ind])
            
            err_high = mean[id_str]+(std[id_str]*xstd)
            err_low = mean[id_str]-(std[id_str]*xstd)
            
            axes[ind].fill_between(x[id_str],err_low,err_high,color=colors[ind],alpha=.5)
            
            axes[ind].set_title(id_str)
                        
    return [axes, fig, mean, std]

#-------------------------------------------------------------------------------------------------------
