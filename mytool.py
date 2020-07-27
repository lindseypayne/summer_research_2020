import numpy 
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.ndimage import gaussian_filter


num_ch = 16

# code for displaying one full E2V CCD 
def plotonesensor_E2V(img_list, title):
    
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    
    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []

    for i in range(num_ch):
        if i < columns:
            image_com = numpy.rot90(img_list[i + 8], 2)
        else:
            image_com = img_list[num_ch - (i + 1)]

        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])

        N = 0.3
        im = plt.imshow(image_com, vmin=mean-N*std,vmax=mean+N*std,origin="lower")
        
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    
    

# code for displaying one full E2V CCD with a smoothing filter
def plotonesensor_E2V_smooth(img_list, title):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2

    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []

    for i in range(num_ch):
        if i < columns:
            image_com = numpy.rot90(img_list[i + 8], 2)
        else:
            image_com = img_list[num_ch - (i + 1)]

        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])

        N = 0.3
        sigma=10              # changes width of smoothing
        im = plt.imshow(gaussian_filter(image_com,sigma=sigma),vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap=pylab.get_cmap("tab20c"))

    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
    

    


# code for displaying one full ITL CCD    
def plotonesensor_ITL(img_list, title):
    
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    
    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    for i in range(num_ch):
        if i < columns:
            
            image_com = np.flipud(img_list[columns - (i + 1)])    # top row must be reversed (so 07 ... 00 NOT 00 ... 07)
        else:
            image_com = img_list[i]                               # bootom row in normal order (08 ... 15)

        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])

        N = 0.3
        im = plt.imshow(image_com, vmin=mean-N*std,vmax=mean+N*std,origin="lower")
        
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
    
    
    
    
    
# code for displaying one full ITL CCD with a smoothing filter
def plotonesensor_ITL_smooth(img_list, title):
    
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    
    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    for i in range(num_ch):
        if i < columns:
            
            image_com = np.flipud(img_list[columns - (i + 1)])    # top row must be reversed (so 07 ... 00 NOT 00 ... 07)
        else:
            image_com = img_list[i]                               # bootom row in normal order (08 ... 15)

        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])

        N = 0.3
        sigma=10              # changes width of smoothing
        im = plt.imshow(gaussian_filter(image_com,sigma=sigma),vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap=pylab.get_cmap("tab20c"))
        
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()