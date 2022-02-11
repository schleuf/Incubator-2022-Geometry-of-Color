import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------

def plotOnROI(img, coords, ids, colors, size):
    
    #plot the ROI, then outline all classed cones in yellow and fill in with their respective cone type 
    fig, ((ax0)) = plt.subplots(1,1)
    fig.set_size_inches(size,size)

    #overlay cone coordinates on the image
    ax0.imshow(img)
    
    for ind,cone_type in enumerate(ids):
        
        #outline all cones in the 'all' mosaic in white
        if cone_type == 'all':
            csize = 30
            cface = 'none'
            cedge = colors[ind]
            
        else: # for 'L','M', and 'S' mosaics plot all cones in solid circles of their respective color
            csize = 10
            cface = colors[ind]
            cedge = 'none'
            
        if coords[ind].size > 0:
            
            if coords[ind].size > 2:
                xcoords = coords[ind][:,0]
                ycoords = coords[ind][:,1]
                
            else:
                xcoords = coords[ind][0]
                ycoords = coords[ind][1]
                
            ax0.scatter(x=xcoords, y=ycoords, s=csize, facecolors=cface,edgecolors=cedge)

#-------------------------------------------------------------------------------------------------------

def quad_fig(size):
    """
    initialize 2x2 figure.  input: size = [x,y] in inches
    """
    
    fig, ((ax0, ax1),(ax2, ax3)) = plt.subplots(2,2)
    axes = [ax0,ax1,ax2,ax3]
    fig.set_size_inches(size[0],size[1])
    fig.tight_layout()
    
    return axes,fig

#-------------------------------------------------------------------------------------------------------

def quad_coord(coord_dict,z_dim,unit,ids,colors):
    """
    """
    axes,fig = quad_fig([9,9])
    
    for ind,id_str in enumerate(ids): 
        
        if len(coord_dict[id_str].shape) == 2:    # 2D COORDINATE ARRAY
            scatter_x = coord_dict[id_str][:,0]
            scatter_y = coord_dict[id_str][:,1]
            
        else: #3D COORDINATE ARRAY
            scatter_x = coord_dict[id_str][:,0,z_dim]
            scatter_y = coord_dict[id_str][:,1,z_dim]
            
        axes[ind].scatter(x = scatter_x, y = scatter_y,
                     s = 10,
                     facecolors=colors[ind],
                     edgecolors='none') 
        axes[ind].set_xlabel('distance (' + unit +')')
        axes[ind].set_title(id_str)
        axes[ind].set_ylabel('distance (' + unit +')')
        axes[ind].set_aspect('equal')
    
    return axes

def quad_scat(x_dict,y_dict,z_dim,unit,ids,colors):
    """
    """
    axes,fig = quad_fig([9,9])
    
    for ind,id_str in enumerate(ids): 
        
        if len(y_dict[id_str].shape) == 1:    # 2D COORDINATE ARRAY
            scatter_x = x_dict[id_str]
            scatter_y = y_dict[id_str]
            
        else: #3D COORDINATE ARRAY
            for z in z_dim:
                scatter_x = x_dict[id_str]
                scatter_y = y_dict[id_str][:,z]
                
                axes[ind].scatter(x = scatter_x, y = scatter_y,
                     s = 10,
                     facecolors=colors[ind],
                     edgecolors='none') 
            
        axes[ind].scatter(x = scatter_x, y = scatter_y,
                     s = 10,
                     facecolors=colors[ind],
                     edgecolors='none') 
        axes[ind].set_xlabel('distance (' + unit +')')
        axes[ind].set_title(id_str)
        axes[ind].set_ylabel('distance (' + unit +')')
    
    return axes, fig

#-------------------------------------------------------------------------------------------------------

def quad_hist(hist_dict,bin_edges,unit,ids,colors,x_dim):
    """
    """
    axes,fig = quad_fig([9,9])
        
    # plot histogram of intercone distances for each mosaic
    for  ind,id_str in enumerate(ids):
        if len(hist_dict[id_str].shape)==1:
            hist_data = hist_dict[id_str]
        else:
            hist_data = hist_dict[id_str][:,x_dim]
        bin_width = bin_edges[id_str][1] - bin_edges[id_str][0]
        axes[ind].hist(hist_data, 
                    bins=bin_edges[id_str],
                    color = colors[ind])
        axes[ind].set_xlabel('distance ('+ unit + ')')
        axes[ind].set_title(id_str)
        axes[ind].set_ylabel('cones per ' + str(bin_width) + unit + ' bin')   
        
    return axes

#-------------------------------------------------------------------------------------------------------

def quad_plot(x, plot_dict,unit,ids,colors):
    axes,fig = quad_fig([9,9])
    
    for ind,id_str in enumerate(ids):
        axes[ind].plot(x[id_str],plot_dict[id_str],color=colors[ind])
        axes[ind].set_title(id_str)
        
    return axes,fig

#-------------------------------------------------------------------------------------------------------