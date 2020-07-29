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


# number of amplifiers/segments to be plotted in CCD
num_ch = 16


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
        
    plotonesensor_ITL(img_list, title)
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
    
    
    
    
    
# more color maps: 
    # N = 0.5
    #im = plt.imshow(image_com,vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap=pylab.get_cmap("tab20c"))
    
    # sigma=10
    # N = 0.5
    # im = plt.imshow(image_com,vmin=mean-N*std,vmax=mean+N*std,origin="lower",cmap='Greys')
    


    
