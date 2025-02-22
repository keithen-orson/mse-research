import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from tifffile import imsave
from pathlib import Path
from skimage import io
from os.path import split, join
import matplotlib as mpl
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde

def load_registered_xas():
    """Loads the XAS spectra, masks, and photon energies. 
    Dictionary keys are the element, Cr, Ni, Ti, Al, Fe
    Returns three dictionaries 1 is the images, 2 is the masks, 3 is the photon energies.
    """
    cr_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Cr_shifted.tif')
    ni_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Ni_shifted.tif')
    ti_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Ti_shifted.tif')
    al_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Al_shifted.tif')
    fe_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Fe_shifted.tif')
    xas_elements = {"Cr":cr_xas,"Ni":ni_xas,"Ti":ti_xas,"Al":al_xas,"Fe":fe_xas}

    al_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Al_frame_30_leveled_gaussian_blur.tif')
    cr_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Cr_frame_30_leveled_gaussian_blur.tif')
    fe_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Fe frame 15 gaussian blur only.tif')
    ni_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Ni_frame_14_leveled_gaussian_blur.tif')
    ti_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Ti frame 37 gaussian blur.tif')
    xas_masks = {"Cr":cr_mask,"Ni":ni_mask,"Ti":ti_mask,"Al":al_mask,"Fe":fe_mask}

    xas_photon_energies= {"Cr":np.linspace(570,584,len(cr_xas)), "Ni":np.linspace(848,865,len(ni_xas)), 
                        "Ti":np.linspace(452,470,len(ti_xas)),"Al":np.linspace(75,90,len(al_xas)),
                        "Fe":np.linspace(703,718,len(fe_xas))}
    return xas_elements,xas_masks, xas_photon_energies

def load_xas_imgs():
    """Loads the XAS hyperspectral elements. 
    Dictionary keys are the element, Cr, Ni, Ti, Al, Fe
    Returns: 
        dict: A dictionary with the element name key and the image
    """
    cr_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Cr_shifted.tif')
    ni_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Ni_shifted.tif')
    ti_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Ti_shifted.tif')
    al_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Al_shifted.tif')
    fe_xas = imread('/Users/apple/Sync/Research/Alloy 72/Registration/Fe_shifted.tif')
    xas_elements = {"Cr":cr_xas,"Ni":ni_xas,"Ti":ti_xas,"Al":al_xas,"Fe":fe_xas}

    return xas_elements

def load_phase_masks():
    """
    Loads the phase masks for different phases in the material.
    
    The function reads the masks for various phases from the specified file paths and returns them in a dictionary.
    The phases include Al enriched phase, islands, Ni depleted phase, and L21 phase.
    
    Returns:
        dict: A dictionary with keys as phase names and values as the corresponding masks.
    """
    al_phase = imread("/Users/apple/Sync/Research/Alloy 72/Segmentation/Registered masks/Al enriched phase.tif")
    islands = imread("/Users/apple/Sync/Research/Alloy 72/Segmentation/Registered masks/islands only.tif")
    ni_depleted = imread("/Users/apple/Sync/Research/Alloy 72/Segmentation/Registered masks/ni depleted phase.tif")
    l21 = imread("/Users/apple/Sync/Research/Alloy 72/Segmentation/Registered masks/registered l21 mask.tif")
    inverse_fcc = al_phase+islands+ni_depleted+l21
    fcc = inverse_fcc[inverse_fcc == 0]

    phase_masks = {"Al phase": al_phase, "Islands": islands, "Ni depleted": ni_depleted, "L21": l21, "FCC": fcc}

    return phase_masks

def subtract_background(image: ArrayLike, xas_energy: ArrayLike, bg_endpoints: tuple):
    """
    This function subtracts a linear background from a hyperspectral image array with dimensions (E, X, Y).
    It computes the background based on two energy endpoints and subtracts it from the image.
    Returns both the background-subtracted image and the linear background used.
    
    Parameters:
    - image: A 3D array (E, X, Y), where E is the energy dimension and (X, Y) are spatial dimensions.
    - xas_energy: 1D array containing the energy values corresponding to the E dimension of the image.
    - bg_endpoints: A tuple of two energy values (start, end) used to define the region for background calculation.
    
    Returns:
    - br_subtracted: The background-subtracted image.
    - linear_br: The computed linear background.
    """
    
    # Get the number of energy levels (E) in the hyperspectral image
    len_element = image.shape[0]
    
    # Find the indices corresponding to the linear background energy range defined by bg_endpoints
    left_index = (np.abs(xas_energy - bg_endpoints[0])).argmin()  # Index of the left endpoint
    right_index = (np.abs(xas_energy - bg_endpoints[1])).argmin()  # Index of the right endpoint

    # Reshape the image for easier processing; flatten X and Y dimensions into one dimension
    reshaped = image.reshape(image.shape[0], image.shape[1] * image.shape[2]).T
    
    # Initialize a matrix to store the linear background for each pixel
    linear_br = np.ones(reshaped.shape)
    
    # Loop through each pixel in the reshaped image (each row corresponds to a pixel)
    for xy in range(len(reshaped)):
        # Calculate the average intensity at the left and right endpoints of the background region
        left_end = np.average(reshaped[xy, left_index:left_index+2])  # Left endpoint average
        right_end = np.average(reshaped[xy, right_index-2:right_index])  # Right endpoint average
        
        # Calculate the slope of the background line
        linear_br_slope = (right_end - left_end) / len_element
        
        # Generate the linear background for this pixel using the slope and left endpoint
        for e in range(reshaped.shape[1]):
            linear_br[xy, e] = linear_br_slope * e + left_end
    
    # Transpose the linear background to match the original image shape
    linear_br = linear_br.T
    linear_br = linear_br.reshape(image.shape)  # Reshape back to the original (E, X, Y) shape

    # Subtract the linear background from the original image
    br_subtracted = image.copy() - linear_br

    # Return the background-subtracted image and the linear background
    return br_subtracted, linear_br

def load_reg_XAS_masks():
    """Loads the XAS masks. 
    Dictionary keys are the element, Cr, Ni, Ti, Al, Fe
    returns a dictionary of the masks
    Returns:
        dict: A dictionary with elements as keys holding the registered XAS masks"""

    al_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Al_frame_30_leveled_gaussian_blur.tif')
    cr_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Cr_frame_30_leveled_gaussian_blur.tif')
    fe_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Fe frame 15 gaussian blur only.tif')
    ni_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Ni_frame_14_leveled_gaussian_blur.tif')
    ti_mask = imread('/Users/apple/Sync/Research/Alloy 72/Segmentation/Fiji segmentation/Ti frame 37 gaussian blur.tif')
    xas_masks = {"Cr":cr_mask,"Ni":ni_mask,"Ti":ti_mask,"Al":al_mask,"Fe":fe_mask}

    return xas_masks

def plot_pca_results(pca,scores,e_values=None):
    """
    Plots the results of PCA analysis including scree plot, principal components, and score plots.
    
    Parameters:
    - pca: The PCA object containing the results of the PCA analysis.
    - scores: The scores obtained from the PCA analysis.
    - e_values: Optional, the energy values corresponding to the components. If None, defaults to a range of scores.
    """

    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8))
    if e_values is None:
        e_values = np.arange(scores.shape[1])

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    critical_value = np.amin(np.where(cumulative_variance >= 0.9))

    #Scree Plot
    axs[0,0].plot(pca.explained_variance_ratio_[:20],'bo')
    axs[0,0].set_title("Scree Plot")
    axs[0,0].set_ylabel("Explained Variance")
    axs[0,0].set_xlabel("Principal Component")
    axs[0,0].annotate(str(critical_value)+" PC's needed for 90%",(.27,.85),xycoords='figure fraction')

    #Plot components 1-4
    axs[0,1].set_title("Principal Components")
    axs[0,1].plot(e_values, pca.components_[0])
    axs[0,1].plot(e_values, pca.components_[1])
    axs[0,1].plot(e_values, pca.components_[2])
    axs[0,1].plot(e_values, pca.components_[3])
    axs[0,1].legend(["C1","C2","C3","C4"])
    axs[0,1].set_ylabel("intensity (arb. u.)")
    axs[0,1].set_xlabel("Photon energy (eV)")

    #Do a score plot
    pc1v2 = np.vstack([scores[0,:],scores[1,:]])
    kernel_1v2 = gaussian_kde(pc1v2)
    density_1v2 = np.reshape(kernel_1v2(pc1v2).T, scores[0,:].shape)
    axs[1,0].scatter(scores[0,:],scores[1,:],c=density_1v2)
    axs[1,0].set_xlabel("PC1, (" + str(round(pca.explained_variance_ratio_[0]*100,1))+"%)")
    axs[1,0].set_ylabel("PC2, (" + str(round(pca.explained_variance_ratio_[1]*100,1))+"%)")

    pc3v4 = np.vstack([scores[2,:],scores[3,:]])
    kernel_3v4 = gaussian_kde(pc3v4)
    density_3v4 = np.reshape(kernel_3v4(pc3v4).T, scores[0,:].shape)
    axs[1,1].scatter(scores[2,:],scores[3,:],c=density_3v4)
    axs[1,1].set_xlabel("PC3, (" + str(round(pca.explained_variance_ratio_[2]*100,1))+"%)")
    axs[1,1].set_ylabel("PC4, (" + str(round(pca.explained_variance_ratio_[3]*100,1))+"%)")

def show_score_imgs(pca_scores, num_components,imsize = 1024):
    """
    Displays the PCA score images for the specified number of components.
    
    Parameters:
    - pca_scores: The scores obtained from the PCA analysis.
    - num_components: The number of principal components to display.
    - imsize: The size of the image (default is 1024).
    """

    fig1, axs = plt.subplots(1,num_components)

    for i in range(num_components):
        # Reshape the PCA scores to the original image size
        pc_i = np.reshape(pca_scores[:, i], (imsize, imsize))
        # Display the image with intensity scaling
        axs[i].imshow(pc_i, vmin=np.percentile(pc_i, .5), vmax=np.percentile(pc_i, 99.5))
        axs[i].axis('off')
        axs[i].set_title("Component " + str(i + 1))

def plot_nmf(nmf, e_values=None):
    """
    Plot function designed to plot the components produced by the NMF package in skimage
    """
    
    if e_values is None:
        e_values = np.arange(nmf.components[1].shape[1])
    #Plot nmf components
    component_list = []
    for i in range(nmf.n_components_):
        plt.plot(e_values, normalize_array(nmf.components_[i]))
        component_list.append("C"+str(i))
    plt.title("NMF Components")
    plt.legend(component_list)
    plt.ylabel("intensity(arb.u.)")
    plt.xlabel("Photon energy (eV)")

def plot_ica(ica, n_components = 4, e_values=None):
    """
    Plot function designed to plot the components produced by the ICA package in skimage
    """
    if e_values is None:
        e_values = np.arange(ica.components[1].shape[1])
    #Plot nmf components
    component_list = []
    for i in range(n_components):
        plt.plot(e_values, normalize_array(ica.components_[i]))
        component_list.append("C"+str(i))
    plt.title("NMF Components")
    plt.legend(component_list)
    plt.ylabel("intensity(arb.u.)")
    plt.xlabel("Photon energy (eV)")

def normalize_array(arr: ArrayLike):
    """Normalize the values of an array from 1 to 0"""
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr

def make_16bit(img: ArrayLike):
    """
    Converts an array image into a 16-bit grayscale image by normalizing and scaling to 16-bit.
    
    Parameters:
    - img: The input image array.
    
    Returns:
        np.ndarray: The 16-bit grayscale image.
    """
    normed = normalize_array(img)
    scaled = normed * 65536
    return scaled.astype(np.uint16)

def trim_edges(image: ArrayLike, border=30):
    """
    Trim the edges of an image by removing a border from the x, y dimensions.
    Works for both 2D and 3D images.

    Parameters:
    - image: A 2D or 3D array (shape can be either (x, y) or (n, x, y))
    - border: The width of the border to remove from all sides of the x and y dimensions.

    Returns:
    - out_img: The trimmed image, with the specified border removed.
    """
    # Check if the image is 2d or 3D (n, x, y)
    if image.ndim == 3:
        # Trim the x and y dimensions, but leave the first (n) dimension intact
        out_img = image[:, border:-border, border:-border]
    elif image.ndim == 2:
        # Trim both x and y dimensions for 2D images
        out_img = image[border:-border, border:-border]
    else:
        raise ValueError("Input image must be either 2D or 3D.")
    
    return out_img

def mask_subtract_arr(image: ArrayLike, mask: ArrayLike, return_nan=False):
    """
    Applies a mask to an array of shape (n, x, y) or (x, y) by multiplying it element-wise.
    
    Parameters:
    - image: A 2D or 3D array (shape can be either (x, y) or (n, x, y)).
    - mask: A 2D array (shape (x, y)) to be applied to the image.
    - return_nan: If True, masked regions will be set to NaN. Default is False.
    
    Returns:
    - masked_image: The image after applying the mask.
    """
    # Normalize the mask to the range [0, 1]
    normed_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    
    # Create a copy of the image to apply the mask
    masked_image = np.copy(image)
    
    try:
        try:
            # If the image is 3D, apply the mask to each slice
            for slice in range(image.shape[0]):
                masked_image[slice, :, :] = image[slice, :, :] * normed_mask[:, :]
        except:
            # If the image is 2D, apply the mask directly
            masked_image[:, :] = image[:, :] * normed_mask[:, :]
    except:
        print("Mask/Image dimension mismatch, masking skipped")
        return masked_image
    
    # If return_nan is True, set masked regions to NaN
    if return_nan:
        masked_image[masked_image == 0] = np.nan

    return masked_image

def mask_subtract_file(data: str, mask: str, outfile=None):
    """
    Takes a numpy array and multiplies it by a mask. Data and mask must be the same dimensions, and mask must be valued 1-0.
    
    Parameters:
    - data: The file path to the data image.
    - mask: The file path to the mask image.
    - outfile: Optional; the file path to save the masked image. If None, the image is not saved.
    
    Returns:
    - image: The masked image.
    """
    # Read the image
    image = io.imread(data)
    print(image.shape, image.dtype)
    original_dtype = image.dtype
    
    # Read the image mask
    mask = io.imread(str(mask))
    print(mask.shape, mask.dtype)
    
    # Normalize the mask to the range [0, 1]
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    print(np.min(mask), np.max(mask))

    try:
        try:
            # If the image is 3D, apply the mask to each slice
            for slice in range(image.shape[0]):
                image[slice, :, :] = image[slice, :, :] * mask[:, :, 0]
        except:
            # If the image is 2D, apply the mask directly
            for slice in range(image.shape[0]):
                image[slice, :, :] = image[slice, :, :] * mask[:, :]
    except:
        print("Mask/Image dimension mismatch, masking skipped")

    # Convert the image back to its original data type
    image = image.astype(original_dtype)
    print(image.dtype)
    
    # Save the masked image if an output file path is provided
    if outfile is not None:
        imsave(outfile, image)
    
    return image

def plane_level_njit(img: np.ndarray):
    """
    Takes a 2D image (numpy array), calculates the mean plane, and returns that 2D image after subtracting the mean plane.
    
    Parameters:
    - img: A 2D numpy array representing the image.
    
    Returns:
    - final_img: The image after subtracting the mean plane.
    """
    # Get the dimensions of the image
    dimensions = img.shape
    totalpoints = dimensions[0] * dimensions[1]
    
    # Create x and y arrays corresponding to the pixel positions
    xarray = np.linspace(0, totalpoints - 1, totalpoints) % dimensions[0]
    yarray = np.linspace(0, dimensions[1] - 1, totalpoints)  # a sequence that makes y values that correspond to the z points
    
    # Flatten the image to a 1D array
    flatimg = img.flatten() + 0.0

    # Create a constant array for the linear equation
    cvals = xarray * 0 + 1.0  # make the cvals floats

    # Stack x, y, and cvals arrays into a single 2D array
    xyc = np.column_stack((xarray, yarray, cvals))
    
    # Calculate the coefficients of the best fit plane using least squares
    coefs = np.linalg.lstsq(xyc, flatimg, rcond=None)[0]  # njit and lstsq require all the datatypes to be the same

    # Subtract the mean plane from the flattened image
    leveled_flat = flatimg - ((coefs[0] * xarray + coefs[1] * yarray + coefs[2]) - np.mean(flatimg))

    # Reshape the leveled image back to the original shape
    final_img = leveled_flat.reshape(img.shape)
    
    return final_img

def cosine_similarity_3d(image: np.ndarray, reference: np.ndarray):
    """
    Takes the cosine similarity of a hyperspectral image and a reference spectra of the same dimensions.
    
    Parameters:
    - image: A 3D numpy array (E, X, Y) where E is the energy dimension and (X, Y) are spatial dimensions.
    - reference: A 1D numpy array representing the reference spectra.
    
    Returns:
    - cosine_2d: A 2D numpy array representing the cosine similarity image.
    """
    # Reshape the image to a 2D array where each row is a pixel and each column is an energy value
    flat_image = image.reshape((len(image), image.shape[1] * image.shape[2])).T
    
    # Take the magnitude of each observation for use in the cosine calculation
    norm = np.linalg.norm(flat_image, axis=1)
    
    # Calculate cosine similarity using CS = a.b / (||a|| * ||b||)
    cosine = np.dot(flat_image, reference) / (norm * np.linalg.norm(reference))
    
    # Reshape the cosine similarity result back to a 2D image
    cosine_2d = cosine.reshape((image.shape[1], image.shape[2]))
    
    return cosine_2d