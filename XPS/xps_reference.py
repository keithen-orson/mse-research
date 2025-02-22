import numpy
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import peakfit_functions
import os.path
from os import listdir, chdir
from os.path import isfile, join, getmtime, split

# refspectra = '/Users/apple/Documents/Research/Muri NiCr experiments/NiCr bare metal/5-26/copy20210526-101021_--ESp_Cayman_iXPS--6_1-Detector_Region.txt'
# experimentalspectra = '/Users/apple/Documents/Research/Muri NiCr experiments/Muri Ni22Cr6Mo slow 12-17-21/23c/0/copy_20211217-115137_Muri Ni22Cr6Mo slow--ESp_Cayman_iXPS--3_1-Detector_Region.txt'


def copy_trim_xy(folderpath:str, ke_to_BE=True, norm=False):
    """Trim the metadata from all xy format .txt files in a folder and optionally convert the kinetic energy to binding energy and normalize to 1"""
    mypath = folderpath
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i in onlyfiles:
        if '.txt' not in i:
            filepath = mypath + '/' + i
            print(filepath)
            f = open(filepath)
            file_contents = f.readlines()
            f.close()

        while file_contents[0].startswith('#'):
            del(file_contents[0])

        xy = np.zeros((2, len(file_contents)))
        for j in range(len(file_contents)):
            linedata = file_contents[j].split()
            xy[0, j] = 1486.7-float(linedata[0])
            xy[1, j] = float(linedata[1])

        maxintensity = max(xy[1, :])
        minintensity = min(xy[1,:])
        xy[1,:] = (xy[1,:]-minintensity)/(maxintensity-minintensity)

        for j in range(len(file_contents)):
            file_contents[j] = ''+str(xy[0,j]) + '   ' + str(xy[1,j])+'\n'


        copyname = mypath+'/'+'igor_'+i
        copy = open(copyname, "w")
        for lines in file_contents:
            copy.write('%s' % lines)

        copy.close()
    return

def load_csv(fileoutname, fileinname=True, ):
    """this function is a work in progress, don't use yet"""
    if fileinname == True:
        fileinname = input("Input file name: ")
    print(fileinname)
    xy = pd.read_csv(fileinname)
    xy.to_csv(
        fileoutname,
        index=False, header=False, sep=' ')
    return xy

def load_trimmed_spectra(filename=True, kinetictobinding=True):
    """This function takes a filename and loads a single spectra from it.  The spectra must be just the xy
    data in the format [kinetic energy, intensity] with no header.  It returns the data transformed from
    kinetic energy to binding energy assuming an Al K alpha x-ray source"""

    if filename == True:
        filename = input("Input file name: ")
    print(filename)
    f = open(filename)
    file_contents = f.readlines()

    f.close()
    xy = np.zeros((2, len(file_contents)))
    for j in range(len(file_contents)):
        linedata = file_contents[j].split()

        #stops reading the file at any blank lines
        if not(linedata):
            print("blank lines found, exiting file reader")
            break

        #changes kinetic energy (the default format) to binding energy
        if kinetictobinding:
            xy[0, j] = 1486.7 - float(linedata[0])
            xy[1, j] = float(linedata[1])
        else:
            xy[0, j] = float(linedata[0])
            xy[1, j] = float(linedata[1])

    xy = np.transpose(xy)
    xycopy = xy

    #make sure the binding energies are sorted in ascending order
    xy = xycopy[np.argsort(xycopy[:,0])]
    return xy

def shift_spectra(xy, shift, manualinput=False):
    """shifts a spectra by amount shift.  Takes an array of x,y in either [2, x] or [x, 2]"""

    inputshape = xy.shape
    if inputshape[1] != 2 and inputshape[0] == 2:
        xy = np.transpose(xy)
        print("linear_interpolate() transposed the x,y data")
    if manualinput:
        shift = float(input("Energy Shift: "))


    xy[:,0] = xy[:,0] + shift

    return xy

def spectra_interpolate(xy, step = 0.01, bounds=False):
    """Given an input array of x, y values, this will interpolate the data using scipy's cubic interpolation
    to get a function with the interpolated data and desired boundaries and step"""
    if bounds==False:
        bounds = detect_bounds(xy)
    inputshape = xy.shape
    if inputshape[1] != 2 and inputshape[0] == 2:
        xy = np.transpose(xy)
        print("linear_interpolate() transposed the x,y data")

    f = interp.interp1d(xy[:,0], xy[:,1], kind='linear')
    xnew = np.arange(bounds[0], bounds[1]+step, step)
    ynew = f(xnew)
    xnew = np.array(xnew)
    ynew = np.array(ynew)
    interpolated = np.stack((xnew,ynew))

    return np.transpose(interpolated)

def normalize_to_one(spectra):
    """
    Takes a spectra (energy, intensity in a two column array) and normalizes the max to 1 and min to zero
    :param spectra:
    :return: normalized
    """
    normed_spectra = np.zeros(spectra.shape)
    normed_spectra[:,0] = spectra[:,0]
    spectra_max = np.max(spectra[:, 1])
    spectra_min = np.min(spectra[:, 1])
    normed_spectra[:, 1] = (spectra[:, 1] - spectra_min) / (spectra_max - spectra_min)
    return normed_spectra

def normalize_spectra_to(target, spectra):
    """normalizes the height of one spectra to that of a target spectrum.  Takes x,y data in a numpy array
    and returns an array of x,y column data"""
    target = np.array(target)
    spectra = np.array(spectra)

    #check the dimensions of the input x,y columns to make sure they are shaped the same
    targetshape = target.shape
    if targetshape[1] != 2 and targetshape[0] == 2:
        target = np.transpose(target)
        print("transposed the x,y data for target")

    spectrashape = spectra.shape
    if spectrashape[1] != 2 and spectrashape[0] == 2:
        spectra = np.transpose(spectra)
        print("transposed the x,y data for spectra")



    target_max = np.max(target[:,1])
    target_min = np.min(target[:,1])
    spectra_max = np.max(spectra[:,1])
    spectra_min = np.min(spectra[:,1])

    normed_spectra = np.zeros(spectra.shape)
    normed_spectra[:,0] = spectra[:,0]

    #first normalized intensity of the spectra to the range 0-1
    normed_spectra[:,1] = (spectra[:,1]-spectra_min)/(spectra_max-spectra_min)

    #then normalize the intensityu the spectra to the target spectra
    normed_spectra[:,1] = (normed_spectra[:,1]*(target_max-target_min))+target_min

    return normed_spectra

def subtract_spectra(experimental, reference, wiggle, shirley_subtract=False, testregion=(0,75)):
    """This subtracts a reference spectra (xy columns) from an experimental spectra.  The wiggle parameter
    defines how far the reference spectra can move to account for charging.  Testregion defines the data point
    range for peak alignment process is.  For the Nickel 2p3/2 with step size .05 ev, the metal peaks are
    within the first ~75 data points, or 3.75 eV"""
    #how many steps the reference spectra is allowed to move needs to be an integer
    wiggle = int(wiggle)

    #make sure the arrays are numpy arrays and the arrays are both in ascending order
    expcopy = np.array(experimental)
    refcopy = np.array(reference)

    experimental = expcopy[np.argsort(expcopy[:,0])]
    reference = refcopy[np.argsort(refcopy[:,0])]
    print(detect_bounds(experimental))
    print(detect_bounds(reference))

    fig6 = plt.figure(6)
    plt.plot(experimental[:, 0], experimental[:, 1])
    plt.plot(reference[:, 0], reference[:, 1])

    if (shirley_subtract):
        experimentalbr = peakfit_functions.shirley_calculate(experimental[:,0], experimental[:,1])
        referencebr = peakfit_functions.shirley_calculate(reference[:,0], reference[:,1])
        print("subtracted shirley background from spectra")

        plt.plot(reference[:,0], referencebr)
        plt.plot(experimental[:,0],experimentalbr)
        experimental[:,1] = experimental[:,1]-experimentalbr
        reference[:,1] = reference[:,1]-referencebr

    #normalize the height of the reference spectra to the height of the experimental spectra
    reference = normalize_spectra_to(experimental,reference)


    #trim the reference spectra so it can be wiggled without index errors
    trimmed_reference = reference[0:-wiggle,:]

    residualsum = np.zeros(wiggle)
    for i in range(wiggle):
        trimmed_experimental = experimental[i:-(wiggle-i),:]
        residualsum[i] = np.sum(trimmed_experimental[testregion[0]:testregion[1],:]-trimmed_reference[testregion[0]:testregion[1],:])
        fig5 = plt.figure(5)



    index = np.where(np.absolute(residualsum) == np.amin(np.absolute(residualsum)))
    #index = int(index[0])
    index = 5

    #make a new shifted reference

    shifted_ref = reference[index:-(index+1),:]

    #shift the binding energy
    shifted_ref[:,0] = shifted_ref[:,0]+(index+1)*(shifted_ref[1,0]-shifted_ref[0,0])
    #print(shifted_ref)
    #trim the experimental so they're the same length and can be subtracted, also so they start at the same binding energy
    trimmed_experimental = experimental[2*index+1:,:]
    #print(trimmed_experimental)

    subtracted = trimmed_experimental-shifted_ref
    subtracted[:,0] = trimmed_experimental[:,0]
    plt.plot(subtracted[:,0],subtracted[:,1])
    plt.show()
    return subtracted

def detect_bounds(spectra):
    return np.min(spectra[:,0]), np.max(spectra[:,0])

#These imports are for the shirley_calculate function
from numpy import array, linspace, arange, zeros, ceil, amax, amin, argmax, argmin, abs
from numpy.linalg import norm
def shirley_calculate(x, y, tol=1e-5, maxit=10):
    """ S = specs.shirley_calculate(x,y, tol=1e-5, maxit=10)
    Calculate the best auto-Shirley background S for a dataset (x,y). Finds the biggest peak
    and then uses the minimum value either side of this peak as the terminal points of the
    Shirley background.
    The tolerance sets the convergence criterion, maxit sets the maximum number
    of iterations.

    This function is from Kane O'Donnell
    https://github.com/kaneod/physics/blob/master/python/specs.py
    """

    DEBUG = False
    OPTION = 2

    # Make sure we've been passed arrays and not lists.
    x = array(x)
    y = array(y)

    # Sanity check: Do we actually have data to process here?
    if not (x.any() and y.any()):
        print("specs.shirley_calculate: One of the arrays x or y is empty. Returning zero background.")
        return zeros(x.shape)

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
        print("shirley_calculate reversed the energy values")
    else:
        is_reversed = False

    # Locate the biggest peak.
    maxidx = abs(y - amax(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print("specs.shirley_calculate: Boundaries too high for algorithm: returning a zero background.")
        return zeros(x.shape)

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - amin(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - amin(y[maxidx:])).argmin() + maxidx
    xl = x[lmidx]
    yl = y[lmidx]
    xr = x[rmidx]
    yr = y[rmidx]

    # Max integration index
    imax = rmidx - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    it = 0
    while it < maxit:
        if DEBUG:
            print("Shirley iteration: ", it)
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1]
                                               - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum
        # Calculate new B
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] +
                                                   y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        if norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B

def EI_data_from_snapshot(linedata, kinetictobinding=True):
    """This block of code gets rid of the metadata starting with #, and sums up all of the individual measurements from
    each snapshot, and gets (energy, intensity) data in the numpy array 'dataset'"""

    str_dataset = []
    dataset = []  # this becomes your full dataset by the end, in python list format

    for row in linedata:
        if row.find('#') == -1:
            row_split = row.split()

            new_row = []
            row_str = ''  # accumulators
            intensity_sum = 0

            for element in row_split:
                try:
                    num_element = float(element)
                    if row_split.index(element) == 0:
                        if kinetictobinding:
                            new_row.append(1486.7-num_element)
                            row_str += element + '\t'
                        else:
                            new_row.append(num_element)
                            row_str += element + '\t'
                    else:
                        intensity_sum += num_element
                except:
                    pass
            row_str += str(intensity_sum) + '\n'
            str_dataset.append(row_str)
            new_row.append(intensity_sum)
            dataset.append(new_row)

    data = np.array(dataset)  # converts the data to a numpy array
    return data

"""This is the main function for collecting all the snapshots in a directory and putting them in a dataframe"""

def find_snapshots_in_dir(search_dir):
    """This function gathers the snapshots from each file and returns them as a pandas dataframe"""
    # sort the snapshot files by the snapshot number- sorting by date doesn't work because they were all created by the exporter around the same time
    files = [f for f in listdir(search_dir) if (isfile(join(search_dir, f)) and "DS_Store" not in f)]
    files = [os.path.join(search_dir, f) for f in files] # add path to each file

    masterlist = pd.DataFrame(columns=['filename','data','scantype', 'date','time','scansum'])

    shift = 0
    for i, name in enumerate(files):
        #This loop gathers all the data and aligns the spectra
        if not (".txt" in name):
            continue
        elif ('._' in name):
            continue
        filepath = os.path.join(search_dir, name)
        f = open(filepath)
        lines = f.readlines()
        f.close()

        line1 = lines[0].split(" ")     #get the xps transition type from the metadata
        transition = line1[-2]+ line1[-1]
        transition = transition.replace("\n","")
        pair = os.path.split(name)
        tail = pair[1]
        filedate = tail.split('-')[0]
        filetime = tail.split('-')[1]
        filetime = filetime.split('_')[0]
        data = EI_data_from_snapshot(lines)


        masterlist = masterlist.append([{'filename': name,
                                         'data': data,
                                         'scantype': transition,
                                         'date': filedate,
                                         'time': filetime,
                                         'center': shift,
                                         'scansum': np.sum(data[:,1])
                                         }])

    masterlist = masterlist.reset_index(drop=True)


    return masterlist