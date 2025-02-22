import numpy
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import peakfit_functions as pf
from os import listdir, chdir
from os.path import isfile, join, getmtime, split
from scipy.interpolate import make_interp_spline, splrep, BSpline

refspectra = '/Users/apple/Documents/Research/Muri NiCr experiments/NiCr bare metal/5-26/copy20210526-101021_--ESp_Cayman_iXPS--6_1-Detector_Region.txt'
experimentalspectra = '/Users/apple/Documents/Research/Muri NiCr experiments/Muri Ni22Cr6Mo slow 12-17-21/23c/0/copy_20211217-115137_Muri Ni22Cr6Mo slow--ESp_Cayman_iXPS--3_1-Detector_Region.txt'

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

def spectra_interpolate(xy, step, bounds):
    """Given an input array of x, y values, this will interpolate the data using scipy's cubic interpolation
    to get a function with the interpoolated data and desired boundaries and step """
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

"""Testing the spectra_interpolate() function """
# data = load_trimmed_spectra()
# fig1 = plt.figure(1)
# plt.plot(data[0,:],data[1,:])
#
# xnew, ynew = spectra_interpolate(data, .05, (850,864))
#
# fig2 = plt.figure(2)
# plt.plot(xnew, ynew)
# plt.show()

# """testing the normalize_spectra_to() function"""
# #subtract the shirley backgrounds by hand using the good compound background from kolxpd
# niexpbr = load_trimmed_spectra(kinetictobinding=False, filename="/Volumes/DATA STICK/Muri Ni22Cr6Mo slow 12-17-21/Curve subtraction tests/Ni bck copy.txt")
# niexpbr = np.vstack((niexpbr, niexpbr[-1,:]))
# niexpbr[-1,0] = niexpbr[-1,0]+.05
# niexpbr[:,0] = niexpbr[:,0]+.1
# print(detect_bounds(niexpbr))
# ref = load_trimmed_spectra(filename=refspectra)
# exp = load_trimmed_spectra(filename=experimentalspectra)
# print(detect_bounds(exp))
# refbr = peakfit_functions.shirley_calculate(ref[:,0],ref[:,1])
# exp[:,1] = exp[:,1]-niexpbr[:,1]
# ref[:,1] = ref[:,1]-refbr
#
#
# fig3 = plt.figure(3)
# plt.plot(ref[:,0],ref[:,1])
# plt.plot(exp[:,0],exp[:,1])
#
# normed = normalize_spectra_to(exp, ref)
# refupper = np.average(normed[-5:,1])
# normed = numpy.vstack(([[865,0],[865.1, 0]], normed))
#
# min, max = detect_bounds(exp)
#
# trimmednormed = spectra_interpolate(normed, .05, (min,max))
#
# fig4 = plt.figure(4)
# plt.plot(exp[:,0],exp[:,1])
# plt.plot(trimmednormed[:,0],trimmednormed[:,1])
# plt.show()
#
# finished = subtract_spectra(exp,trimmednormed,20,shirley_subtract=False)
# finaldata = pd.DataFrame(finished)
# finaldata.to_csv('/Users/apple/Documents/Research/Muri NiCr experiments/Muri Ni22Cr6Mo slow 12-17-21/subtracttest.csv', index=False, header=False)


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

def remove_outliers(edgesize, sigma, data):
    """Removes outliers by excluding the edge channels (defined by edgesize) and excluding anything with a value or
    derivative that is sigma standard deviations away from the mean"""
    diffdata = np.gradient(data[:, 1])
    averagediff = np.average(diffdata)
    stddiff = np.std(diffdata)
    outlierlist = []
    averagintensity = np.average(data[:, 1])
    stdintensity = np.std(data[:, 1])

    for i in range(len(diffdata)):
        if (abs(diffdata[i] - averagediff) > sigma * stddiff):
            outlierlist = np.append(outlierlist, int(i))
            data[i,1] = np.NaN
        elif (abs(data[i, 1] - averagintensity) > sigma * stdintensity):
            outlierlist = np.append(outlierlist, int(i))
            data[i,1] = np.NaN
        if ((i <= (edgesize-1)) and len(diffdata) >= edgesize):
            outlierlist = np.append(outlierlist, int(i))
            data[i,1] = np.NaN
        elif ((len(diffdata) - i) <= edgesize):
            outlierlist = np.append(outlierlist, int(i))
            data[i,1] = np.NaN
    return data

def find_ni_peak(x, y, guess=[100, 853, 2, 100, 853, 2]):
    """finds the center of the nickel peak using 1 voigt peak centered around 853"""
    # guess = [ampG1, cenG1, sigmaG1,ampL1, cenL1, widL1]
    paramopt = pf.fit_1Voigt(x, y, guess)

    return paramopt

def gather_snapshots(transition, masterdata,normalize=False):
    """
    Given a transition name (string) and a dataframe containing arrays of data corresponding to a transition, return
    a triangular set of coordinates with format (
    :param transition:
    :param masterdata:
    :return: plotdata
    """
    sorted_data = masterdata[masterdata.scantype == transition]
    plotdata = np.zeros((0,3))
    for j, i in enumerate(sorted_data.data):
        snapshot = (np.array(i))

        if normalize:
            snapshot = normalize_to_one(snapshot)
        indexs = np.zeros((snapshot.shape[0],1))
        indexs[:] = j
        snapshot = np.hstack((indexs,snapshot))
        plotdata = np.vstack((plotdata,snapshot))

    return plotdata

def normalize_group(masterlist):
    """Takes the master list and normalizes the spectra in the master list"""

    for i, name in enumerate(masterlist.index):
        rollingsum = 0
        snapshot = masterlist.iloc[i]
        if snapshot.scantype == "Ni2p3/2":
            j = i+1
            rollingsum +=snapshot.scansum
            scanlist = []
            scanlist = scanlist.append(snapshot.scantype)
            while ("Ni2p3/2" not in scanlist) and ("Mo3d" not in scanlist) and ("Cr2p3/2" not in scanlist):
                current = masterlist.iloc[j]
                if "O1s" in current.scantype:
                    j += 1
                    continue
                elif current.scantype in scanlist:
                    j +=1
                    continue
                rollingsum += current.scansum
                j+=1

            #print(rollingsum)
            rollingsum=0

    return masterlist

def plot_df_spectra(dataframe, hold=False):
    """Goes through a dataframe formatted by the snapshots_from_dir() function from the xps_reference.py file"""
    for index, row in dataframe.iterrows():
        xy = row["data"]
        plt.plot(xy[:,0],xy[:,1])
        plt.xlim(max(xy[:,0]),min(xy[:,0]))
        plt.title(row["scantype"])
        plt.xlabel("Binding Energy (eV)")
        plt.ylabel("Intensity (arb. u.)")
        if not hold:
            plt.show()
    if hold:
        plt.show()


"""This is the start of the main function!!!"""

def snapshots_from_dir(search_dir,sort = False, align = "none"):
    """This function gathers the snapshots from each file and aligns them based off of the nickel peak"""
    # sort the snapshot files by the snapshot number- sorting by date doesn't work because they were all created by the exporter around the same time
    files = [f for f in listdir(search_dir) if isfile(join(search_dir, f))]
    if sort:
        files.sort(key=lambda x: x.split('-')[5])
        files.sort(key=lambda x: len(x))
    files = [join(search_dir, f) for f in files] # add path to each file

    masterlist = pd.DataFrame(columns=['filename','data','scantype', 'date','time','scansum'])
    shift = 0
    signalsum = 0
    for i, name in enumerate(files):
        #This loop gathers all the data and aligns the spectra
        if ".txt" not in name:
            continue
        elif ('._' in name):
            continue
        filepath = join(search_dir, name)
        f = open(filepath)
        lines = f.readlines()
        f.close()

        line1 = lines[0].split(" ")     #get the xps transition type from the metadata
        transition = line1[-2]+ line1[-1]
        transition = transition.replace("\n","")
        pair = split(name)
        tail = pair[1]
        filedate = tail.split('-')[0]
        filetime = tail.split('-')[1]
        filetime = filetime.split('_')[0]
        data = EI_data_from_snapshot(lines)

        #optional outlier removal
        #data = remove_outliers(3, 3)

        center = 0
        #defines an alignment transition, in this case nickel, and aligns all snapshots that are not Ni by the Ni shift
        #shift is updated on every Ni peak
        nicondition = ("Ni2p3/2" in transition)
        isNi = ("Ni2p3/2" in align)
        valencecondition = ("Valence" in transition)
        isValence = ("Valence" in align)
        if nicondition and isNi:
            #If the snapshot is a nickel snapshot, fit the nickel peak with a voigt peak.  if you can successfully fit,
            #update the peak shift parameter
            try:
                fitparams = find_ni_peak(data[:, 0], data[:, 1])
                center = fitparams[1]
                shift = 852.6 - center
            except RuntimeError:
                print("peak finding failed: " + tail)
                centerindex = np.argmax(data[:,1])
                center = data[centerindex,0]
                print("using simple max at: " + str(center))
                shift = 852.6 - center
                #make the shift- this function remembers the last alignment feature
        elif valencecondition and isValence:
            """Do the shift based on the valence band intersection with zero"""
            derivatives = np.gradient(data[:,1])

            #smooth the derivative using a b spline
            flipped_derivatives = np.flip(derivatives)
            flipped_x = np.flip(data[:, 0])
            spline_derivative = splrep(flipped_x, flipped_derivatives, s=10000)
            smoothed_derivative = BSpline(*spline_derivative)(flipped_x)
            valence_edge = (flipped_x[np.argmin(smoothed_derivative)])

            # plt.plot(np.flip(data[:, 0]), BSpline(*spline_derivative)(flipped_x))
            # plt.plot(data[:,0], derivatives)
            # plt.show()
            # #Optionally just use the minimum of the derivative
            # valence_edge = (data[np.argmin(derivatives),0])
            # print(valence_edge)
            shift = 0-valence_edge


        data = shift_spectra(data, shift)
        masterlist = masterlist.append([{'filename': name,
                                         'data': data,
                                         'scantype': transition,
                                         'date': filedate,
                                         'time': filetime,
                                         'center': center,
                                         'scansum': np.sum(data[:,1])
                                         }])

    masterlist = masterlist.reset_index(drop=True)


    return masterlist