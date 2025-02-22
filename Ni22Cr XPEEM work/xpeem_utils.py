import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from tifffile import imwrite, imsave
import datetime
from numba import jit, njit, prange
from functools import partial
import time 
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from numpy.typing import ArrayLike
from scipy.stats import skew

#Array utility functions
def normalize_array(arr: ArrayLike):
    """Normalize the values of an array from 1 to 0"""
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr


#This is an implementation that subtracts masks from the file- from crop_and_norm.ipynb
def mask_subtract(data: str, mask: str, outfile: str):
    """takes a numpy array and multiplies it buy a mask.  Data and mask must be the same dimensions, and mask must be valued 1-0."""
    image = imread(data)
    print(image.shape, image.dtype)
    original_dtype = image.dtype
    mask = imread(str(mask))
    print(mask.shape, mask.dtype)
    #this should normalize the mask to 0-1.  
    mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
    print(np.min(mask), np.max(mask))

    try:
        try:
            for slice in range(image.shape[0]):
                image[slice,:,:] = image[slice,:,:]*mask[:,:,0]
        except:
                for slice in range(image.shape[0]):
                    image[slice,:,:] = image[slice,:,:]*mask[:,:]
    except: 
        print("Mask/Image dimension mismatch, masking skipped")

    image = image.astype(original_dtype)
    imsave(outfile, image)
    return image

def mask_subtract_arr(image: ArrayLike, mask: ArrayLike):
    """takes an array of shape (n,x,y) or (x,y) and multiply it by a mask with shape (x,y)"""
    normed_mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
    masked_image = np.copy(image)
    try:
        try:
            for slice in range(image.shape[0]):
                masked_image[slice,:,:] = image[slice,:,:]*normed_mask[:,:]
        except:
            masked_image[:,:] = image[:,:]*normed_mask[:,:]
    except: 
        print("Mask/Image dimension mismatch, masking skipped")
        return masked_image

    return masked_image

def mask_subtract_file(data: str, mask: str, outfile=None ):
    """takes a numpy array and multiplies it buy a mask.  Data and mask must be the same dimensions, and mask must be valued 1-0."""
    image = imread(data)
    print(image.shape, image.dtype)
    original_dtype = image.dtype
    mask = imread(str(mask))
    print(mask.shape, mask.dtype)
    #this should normalize the mask to 0-1.  
    mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
    print(np.min(mask), np.max(mask))

    try:
        try:
            for slice in range(image.shape[0]):
                image[slice,:,:] = image[slice,:,:]*mask[:,:,0]
        except:
                for slice in range(image.shape[0]):
                    image[slice,:,:] = image[slice,:,:]*mask[:,:]
    except: 
        print("Mask/Image dimension mismatch, masking skipped")

    image = image.astype(original_dtype)
    print(image.dtype)
    if outfile is not None:

        imsave(outfile, image)
    return image

#These are utility functions implementing tiling, plane leveling, and segmentation efficiently using numba - from xpeem_batchv3.ipynb
@partial(jit, nopython=True)
def swap(x, t_args):
    """Executes the np transpose function in a way that numba will understand, there is a bug with how numba understands np.transpose and this is a workaround"""
    return np.transpose(x, t_args)

@njit
def tile_image(imframe: np.ndarray, tilesize: tuple):
    """Returns an array of image tiles with shape (Y, X, n, m) where Y is the tile row number, X is the tile column
    number, n is the tile pixel height and m is the tile pixel width"""
    img_height, img_width = imframe.shape
    tileheight, tilewidth = tilesize
    imcopy = imframe.copy() #workaround for numba reshape not supporting non-contiguous arrays
    tiled_array = imcopy.reshape(img_height // tileheight, tileheight, img_width // tilewidth, tilewidth)
    out_array = swap(tiled_array,(0,2,1,3))
    return out_array

@njit
def reform_image(tilearray: np.ndarray, originalsize: tuple):
    """Reforms an image from a tile array with shape (Y, X, n, m) where Y is the tile row number, X is the tile column
    number, n is the tile pixel height and m is the tile pixel width.  Returns an image with the shape of the original imag"""
    
    reordered = swap(tilearray,(0,2,1,3))
    contiguous = reordered.copy() #numba doesn't support numpy reshaping of non-congiguous arrays so you have to copy it as a workaround
    reformed = contiguous.reshape(originalsize)
    return reformed

@njit
def linalg_test(solvematrix, zmatrix):
    """Test the least squares solver using numba"""
    coefs = np.linalg.lstsq(solvematrix, zmatrix)[0]
    return coefs

@njit
def plane_level_njit(img: np.ndarray):
    """Takes a 2d image (numpy array), calculates the mean plane, and returns that 2d image after subtracting the
    mean plane"""

    """Create x, y, and z (pixel intensity) arrays"""
    dimensions = img.shape
    totalpoints = dimensions[0] * dimensions[1]
    xarray = np.linspace(0, totalpoints - 1, totalpoints) % dimensions[0]
    yarray = np.linspace(0, dimensions[1] - 1, totalpoints)  # a sequence that makes y values that correspond to the z points
    flatimg = img.flatten()+0.0

    """Calculate the best fit plane through the datapoints"""
    cvals = xarray*0+1.0  #make the cvals floats

    xyc = np.column_stack((xarray, yarray, cvals))
    coefs = np.linalg.lstsq(xyc, flatimg)[0]  #njit and lstsq require all the datatypes to be the same

    leveled_flat = flatimg-((coefs[0]*xarray+coefs[1]*yarray+coefs[2])-np.mean(flatimg))

    final_img = leveled_flat.reshape(img.shape)
    return final_img

@njit
def median_prominence_threshold(imgarray, prominence):
    """Takes an input array and thresholds it using median + prominence"""
    brlevel = np.median(imgarray)
    threshold = brlevel + prominence
    thresholded = (imgarray > threshold)
    return thresholded

#This is the main function that levels and segments an image - from xpeem_batchv3.ipynb
@njit(parallel=True)
def batch_level_segment(image: np.array, tileshape: tuple, imageshape: tuple, threshold, progressupdate=True):
    """efficient numba implementation of a function that takes a 3d image as a np array, tiles it, levels it, 
    and segments it """
    segmented_timeseries = np.zeros(image.shape)

    for frameid in range(image.shape[0]):
        
        frame = image[frameid,:,:]
        tiles = tile_image(frame, tileshape)
        segmented = np.zeros(tiles.shape)

        for i in prange(tiles.shape[0]):
            for j in prange(tiles.shape[1]):
                tile = tiles[i,j,:,:]
                leveltile = plane_level_njit(tile)
                segmentedtile = median_prominence_threshold(leveltile,threshold)
                segmented[i,j,:,:] = segmentedtile


        reformed = reform_image(segmented, imageshape)
        segmented_timeseries[frameid,:,:] = reformed.astype(np.uint8)
        if frameid%50 ==0 and progressupdate:
            print("frame: "+str(frameid))

    return segmented_timeseries

#This function finds particles from a segmented image stack by tracking connected groups of pixels between a minsize and maxsize (inclusive)
def findparticles_3d_img(frames: np.array, minsize=5, maxsize = 200, progressupdate=True):
    """
        This function takes a 3d image "frames" with the structure [frame, x, y]

       Create a pandas dataframe, and then find the particles (regions) using skimage's label function.  Label connects
       regions of the same integer value, i.e. segmented regions. In this dataframe, I also save the perimeter, filled fraction,
       and the area.
    """
    features = pd.DataFrame()
    for i in range(frames.shape[0]):
        label_image = label(frames[i,:,:], background=0)
        props = regionprops_table(label_image,properties=("centroid", "area"))
        framedata = pd.DataFrame(props)
        framedata['frame'] = i
        features = features.append(framedata)
        if i%50 ==0 and progressupdate: print(str(datetime.datetime.now())+" frame " + str(i))

    filtered = features[features['area'] >= minsize]
    filtered = filtered[filtered['area'] <= maxsize]
  
    return filtered


#Functions for visualizing the data from the timeseries - from xpeem_timeseries_visualization.ipynb
def removebadframes(data, movienumber):
    """
    This function just removes frames that are visually identified as out of focus.  See 6/16/21 entry in lab notebook
    for which frames are identified as bad

    data is a pandas dataframe that is is generated from trackpybatch or tackpy
    movienumber identifies which movie is being used.
    """

    if movienumber == 1:
        values = [14, 30, 32, 125, 141, 148]
        badframes = [x - 1 for x in values]     #subtract one to account for the fact that python indexes start at 0, and in the notes I took the frame indexes started at 1
        for value in badframes:
            data = data[(data['frame'] != value)]
    elif movienumber == 2:
        values = [9, 10, 31, 94, 120, 230]
        badframes = [x - 1 for x in values]
        for value in badframes:
            data = data[(data['frame'] != value)]
        data = data[(data['frame'] > 220) | (data['frame'] < 172)]
    elif movienumber == 3:
        values = [269, 268]
        badframes = [x - 1 for x in values]
        for value in badframes:
            data = data[(data['frame'] != value)]
        data = data[(data['frame'] > 251) | (data['frame'] < 227)]

    return data

def frame_to_langmuir(frame):
    """Converts frames to langmuirs for movie 3, starting at 20 L and going to 65L"""
    return frame*(45/719)+20

def get_particle_stats(df:pd.DataFrame, frames= 719):
    """This function gets the number of particles, the median size, the frame id, and the number of langmuirs of oxidation that have passed"""
    summary_stats = pd.DataFrame()
    for i in range(frames):
        frame_df = df[df.frame == i]
        if frame_df.empty:
            continue
        else:
            summary_stats = summary_stats.append([{'frequency': frame_df.shape[0], 
                                                'median_size': np.median(frame_df['area']),
                                                'frame': i,
                                                'area': frame_df['area'].sum(),
                                                'langmuir': frame_df['langmuir'].iloc[0],
                                                'skewness': skew(frame_df['area'])
                                                }])

    return summary_stats

def logistic(x, a, b, c, d):
    return a/(1.+np.exp(-c*(x-d)))+b


#Functions for hyperspectral image interpretation
def cosine_similarity_2d(line,reference):
    """Take the cosine similarity of two 1D vectors"""
    cosine = np.dot(line,reference)/(np.linalg.norm(line)*np.linalg.norm(reference))
    return cosine

def cosine_similarity_3d(image,reference):
    """Take the cosine similarity of a hyperspectral image and a reference spectra of the same dimensions"""
    flat_image = (image.reshape((len(image),image.shape[1]*image.shape[2])).T)
    #Take the magnitude of each observation m for use in the cosine calculation
    norm = np.linalg.norm(flat_image, axis=1)
    #Calculate cosine similarity using CS = a.b/(||a||*||b||)
    cosine = np.dot(flat_image,reference)/(norm*np.linalg.norm(reference))
    #reshape the cosine sum into a 2d image
    cosine_2d = cosine.reshape((image.shape[1],image.shape[2]))
    return cosine_2d

def normed_cs_metal_ox(image, ref1, ref2):
    #performs the normalized cosine similarity using the metal and oxide refs, does not deal with NAN's
    image = image[:len(ref1),:,:]
    metal_cs = cosine_similarity_3d(image, ref1)
    oxide_cs = cosine_similarity_3d(image, ref2)
    cosine_sum = metal_cs+oxide_cs
    oxide_cosine_norm = oxide_cs/cosine_sum
    return oxide_cosine_norm