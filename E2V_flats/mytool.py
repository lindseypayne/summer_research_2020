import numpy 
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.ndimage import gaussian_filter
#   #   #   #   #   #
import glob
import os
from lsst import afw
print(afw.version.__version__)
from lsst.daf.persistence import Butler
from lsst.eotest.sensor import MaskedCCD, makeAmplifierGeometry
from exploreRun import exploreRun
from lsst.eo_utils.base.image_utils import get_ccd_from_id,\
    get_amp_list, get_data_as_read, sort_sflats
from lsst.eo_utils.base.data_access import get_data_for_run
import lsst.eotest.image_utils as imutil
from astropy.io import fits
from astropy.stats import mad_std
import scipy
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.image as mpimg

# number of amplifiers/segments to be plotted in CCD
num_ch = 16
# number of sensors in one raft
num_CCDs = 9

# function to plot a full raft of CCDs
def plotfullraft(raft, filter_band, ratio_img_list):
    num_ch = 16
    num_CCDs = 9
    fig, axs = plt.subplots(3, 3, figsize=(20, 20),dpi=300) 
    axs = axs.ravel()
    fig.suptitle(str(raft) + ' Differential Ratios, ' + str(filter_band) + ' band', fontsize=16)
    columns = 3
    rows = 3
    ax = []
    for i in range(num_CCDs):
        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax1.axis('off')    # gets rid of subplot borders
        img=mpimg.imread(str(ratio_img_list[i]))
        plt.imshow(img)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])  # gets rid of ALL ticks
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    
    
# function to load and parce through QE data to make a list of combined images, each element of the list is an image for one amplifier/segment in the ITL sensor
def create_combined_ITL(superdarkpath, superbiaspath, qeflatpath, weights, title):    
    superdark = get_ccd_from_id(None, superdarkpath, [])     # load superdark
    superbias = get_ccd_from_id(None, superbiaspath, [])     # load superbias
    
    num_ch = 16
    img_list = []
    MEDIAN = None

    # loops over each amplifier in the CCD
    for ch in range(num_ch):
        arr = []                # defines an empty array for storing new combined flat for each amplifier
        for i, aqeflatpath in enumerate(qeflatpath):
            aqeflat = get_ccd_from_id(None, aqeflatpath, [], bias_frame=superbias)

            img = aqeflat.unbiased_and_trimmed_image(ch + 1).getImage().array

            MEDIAN = numpy.median(img)
            img = img/MEDIAN     # this will normalize your image

            arr.append(img*weights[i])

        img = numpy.sum(arr,axis=0)/numpy.sum(weights)   # NEED THIS LINE TO MAKE COMBINED IN UV BAND (weights function called)
        std = mad_std(img)
        # puts images into a list to use below
        img_list.append(img)  
        
    #plotonesensor_ITL(img_list, title)
    return img_list
    




# function to load and parce through QE data to make a list of combined images, each element of the list is an image for one amplifier/segment in the E2V sensor
def create_combined_E2V(superdarkpath, superbiaspath, qeflatpath, weights, title):    
    superdark = get_ccd_from_id(None, superdarkpath, [])     # load superdark
    superbias = get_ccd_from_id(None, superbiaspath, [])     # load superbias

    img_list = []
    MEDIAN = None

    # loops over each amplifier in the CCD
    for ch in range(num_ch):
        arr = []                # defines an empty array for storing new combined flat for each amplifier
        for i, aqeflatpath in enumerate(qeflatpath):
            aqeflat = get_ccd_from_id(None, aqeflatpath, [], bias_frame=superbias)

            img = aqeflat.unbiased_and_trimmed_image(ch + 1).getImage().array

            MEDIAN = numpy.median(img)
            img = img/MEDIAN     # this will normalize your image

            arr.append(img*weights[i])

        img = numpy.sum(arr,axis=0)/numpy.sum(weights)   # NEED THIS LINE TO MAKE COMBINED IN UV BAND (weights function called)
        std = mad_std(img)
        # puts images into a list to use below
        img_list.append(img)  
        
    plotonesensor_E2V(img_list, title)
    return img_list

 

def create_combined_ITL_pointings(superdarkpath, superbiaspath, qeflatpath, weights):    
    superdark = get_ccd_from_id(None, superdarkpath, [])     # load superdark
    superbias = get_ccd_from_id(None, superbiaspath, [])     # load superbias
    
    num_ch = 16
    img_list = []
    MEDIAN = None

    # loops over each amplifier in the CCD
    for ch in range(num_ch):
        arr = []                # defines an empty array for storing new combined flat for each amplifier
        for i, aqeflatpath in enumerate(qeflatpath):
            aqeflat = get_ccd_from_id(None, aqeflatpath, [], bias_frame=superbias)

            img = aqeflat.unbiased_and_trimmed_image(ch + 1).getImage().array

            MEDIAN = numpy.median(img)
            img = img/MEDIAN     # this will normalize your image

            arr.append(img*weights[i])

        img = numpy.sum(arr,axis=0)/numpy.sum(weights)   # NEED THIS LINE TO MAKE COMBINED IN UV BAND (weights function called)
        std = mad_std(img)
        # puts images into a list to use below
        img_list.append(img)  
    return img_list



# code for normalizing the amps in each sensor between amp edges, between the amp before it, and between amp rows (top and bottom)
# the function which calculates “relative gain” by looking at differences between edges of neighboring amplifiers
def internaladjustment(image_com):
    relativenorm = [1.]*16
    # Firstly, adjusting normalization using both edges on left and right
    for i in range(1,16):
        relativenorm[i] = (numpy.median(image_com[i - 1][:,-10:-5] / image_com[i][:,5:10]))  # this line takes a ratio of a column of the right edge of i-1 amp and the left edge of i amp 
    relativenorm = numpy.array(relativenorm)
    
    # Next, make relative normalizations to be normalized against the first amp
    for i in range(1,16):
        relativenorm[i] *= relativenorm[i - 1]
    print(relativenorm)
    
    # Then, adjust normalization using upper and lower rows
    uplow = []
    for i in range(0,8):
        uplow.append(numpy.median((image_com[i][5:10,] * relativenorm[i]) / (image_com[i + 8][-10:-5,] * relativenorm[i + 8])))   # lower row (top) / upper row (bottom) assuming coarsely determined normalization
    uplow = numpy.median(uplow)
    relativenorm[8:] *= uplow
    
    # function returns this array of normalized data for each amp 
    return relativenorm



# code for displaying one full E2V CCD 
def plotonesensor_E2V(img_list, title, relativenorm=None):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    
    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    image_com = []
    for i in range(num_ch):
        if i < columns:
            image_com.append(numpy.rot90(img_list[i + 8], 2))
        else:
            image_com.append(img_list[num_ch - (i + 1)])

    if relativenorm is None:
        relativenorm = internaladjustment(image_com)
    N = 0.5
    print(relativenorm)
    
    for i in range(0,16):
        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])
        im = plt.imshow(image_com[i]*relativenorm[i], vmin=mean-N*std,vmax=mean+N*std,origin="lower")
        
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return image_com
    
    

# code for displaying one full E2V CCD with a smoothing filter
def plotonesensor_E2V_smooth(img_list, title, relativenorm=None):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2

    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    image_com = []
    for i in range(num_ch):
        if i < columns:
            image_com.append(numpy.rot90(img_list[i + 8], 2))
        else:
            image_com.append(img_list[num_ch - (i + 1)])

    if relativenorm is None:
        relativenorm = internaladjustment(image_com)
    N = 0.3
    sigma=10              # changes width of smoothing
    print(relativenorm)
    
    for i in range(0,16):
        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])
        im = plt.imshow(gaussian_filter(image_com[i]*relativenorm[i],sigma=sigma),vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap=pylab.get_cmap("tab20c"))
    
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return image_com



'''
measure the relative gains from both images, such as a1, a2 

divide them in the same way when you create the differential image a1/a2, 

pass relative gains to plotonesensor_ITL as the third argument relativenorm=a1/a2 
'''


# code for displaying one full ITL CCD    
def plotonesensor_ITL(img_list, title, relativenorm=None):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    
    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    image_com = []  # empty list to store data for COMBINED image
    for i in range(num_ch):
        if i < columns:
            image_com.append(np.flipud(img_list[columns - (i + 1)]))    # top row must be reversed (so 07 ... 00 NOT 00 ... 07)
        else:
            image_com.append(img_list[i])                                # bootom row in normal order (08 ... 15)
    
    if relativenorm is None:
        relativenorm = internaladjustment(image_com)
    N = 0.5
    print(relativenorm)
    
    for i in range(0,16):
        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])
        im = plt.imshow(image_com[i]*relativenorm[i], vmin=mean-N*std,vmax=mean+N*std,origin="lower")
        
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return image_com
    

'''
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
'''
    
    
    
    
# code for displaying one full ITL CCD with a smoothing filter
def plotonesensor_ITL_smooth(img_list, title, relativenorm=None):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    
    image_com = None
    wholepixels = numpy.array(img_list).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    image_com = []  # empty list to store data for COMBINED image
    for i in range(num_ch):
        if i < columns:
            image_com.append(np.flipud(img_list[columns - (i + 1)]))    # top row must be reversed (so 07 ... 00 NOT 00 ... 07)
        else:
            image_com.append(img_list[i])                                # bootom row in normal order (08 ... 15)
    
    if relativenorm is None:
        relativenorm = internaladjustment(image_com)
    N = 0.3
    sigma=10              # changes width of smoothing
    print(relativenorm)
    
    for i in range(0,16):
        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])
        im = plt.imshow(gaussian_filter(image_com[i]*relativenorm[i],sigma=sigma),vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap=pylab.get_cmap("seismic"))
        
    plt.suptitle(title)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return image_com

    
    
    
# plotting one ITL sensor and saving the figure as an image for displaying full raft
def plotonesensor_ITL_andsave(diff_arr, sensor_label, raft, i, relativenorm=None):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    image_com = None
    wholepixels = numpy.array(diff_arr).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    image_com = []
    for j in range(num_ch):
        if j < columns:
            image_com.append(np.flipud(diff_arr[columns - (j + 1)]))    # top row must be reversed (so 07 ... 00 NOT 00 ... 07)
        else:
            image_com.append(diff_arr[j])                               # bootom row in normal order (08 ... 15)

    if relativenorm is None:
        relativenorm = internaladjustment(image_com)
    N = 0.5
    
    for k in range(0,16):
        ax1 = fig.add_subplot(rows, columns, k + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])
        im = plt.imshow(image_com[k]*relativenorm[k], vmin=mean-N*std,vmax=mean+N*std,origin="lower")
        
    plt.suptitle('Differential Ratio between COMBINED and CCOB ' + str(sensor_label[i]))
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(raft) + "_" + str(sensor_label[i]) + ".png")
    return image_com
    
    
    
# plotting one E2V sensor and saving the figure as an image for displaying full raft
def plotonesensor_E2V_andsave(diff_arr, sensor_label, raft, i, relativenorm=None):
    fig=plt.figure(figsize=(5, 5), dpi=150)   
    columns = 8
    rows = 2
    image_com = None
    wholepixels = numpy.array(diff_arr).flatten()
    mean = numpy.mean(wholepixels,dtype=numpy.float64)
    std = numpy.std(wholepixels,dtype=numpy.float64)
    ax = []
    image_com = []
    for j in range(num_ch):
        if j < columns:
            image_com.append(numpy.rot90(diff_arr[j + 8], 2))
        else:
            image_com.append(diff_arr[num_ch - (j + 1)])

    if relativenorm is None:
        relativenorm = internaladjustment(image_com)
    N = 0.5
    
    for k in range(0,16):
        ax1 = fig.add_subplot(rows, columns, i + 1)
        ax.append(ax1)
        ax1.set_xticks([])   
        ax1.set_yticks([])
        im = plt.imshow(image_com[k]*relativenorm[k], vmin=mean-N*std,vmax=mean+N*std,origin="lower")
        
    plt.suptitle('Differential Ratio between COMBINED and CCOB ' + str(sensor_label[i]))
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", anchor=(0,-1))
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(str(raft) + "_" + str(sensor_label[i]) + ".png")
    return image_com

    
# more color maps: 
    # N = 0.5
    #im = plt.imshow(image_com,vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap=pylab.get_cmap("tab20c"))
    
    # sigma=10
    # N = 0.5
    # im = plt.imshow(image_com,vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap='Greys')
    


    
