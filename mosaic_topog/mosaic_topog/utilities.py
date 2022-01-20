import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

def quad_fig(size):
    """
    initialize 2x2 figure.  input: size = [x,y] in inches
    """
    
    fig, ((ax0, ax1),(ax2, ax3)) = plt.subplots(2,2)
    axes = [ax0,ax1,ax2,ax3]
    fig.set_size_inches(size[0],size[1])
    fig.tight_layout()
    
    return axes,fig