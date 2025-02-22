from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
from matplotlib import gridspec
from numpy import array, linspace, arange, zeros, ceil, amax, amin, argmax, argmin, abs
from numpy import polyfit, polyval, seterr, trunc, mean
from numpy.linalg import norm
from scipy.interpolate import interp1d
import pandas as pd

DEBUG = False
OPTION = 2



def _1gaussian(x_array, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))

def _2gaussian(x_array, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))

def _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenG1)**2+widL1**2)) )

def _2Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1, ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )+\
           ((ampG2*(1/(sigmaG2*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG2)**2)/((2*sigmaG2)**2)))) +
              ((ampL2*widL2**2/((x-cenL2)**2+widL2**2)) ))

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

def fit_2Voigt(x, y):
    shirley = shirley_calculate(x, y, maxit=20)
    y_br_subtracted = y-shirley
    popt, pcov = scipy.optimize.curve_fit(_2Voigt, x, y_br_subtracted)
    residual = y-(_2Voigt(x, *popt))

    fig = plt.figure(5, figsize=(4, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    gs.update(hspace=0)

    ax1.plot(x, y, "ro")
    ax1.plot(x, _2Voigt(x, *popt)+shirley, 'k--')  # ,\
    # label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))
    # residual
    ax2.plot(x, residual, "bo")
    # Background
    ax1.plot(x, shirley, '^')
    plt.xlim((max(x), min(x)))
    plt.show()

    return popt

def fit_ni2p32(x, y):
    shirley = shirley_calculate(x, y, maxit=20)
    y_br_subtracted = y - shirley
    guess = [.1,1, 852.9, 1000, 1, 1000, 859.5, 1, 1000, 859.5, 1]
    # lims = ([-np.inf, -np.inf, 852.6, 0, .4, 0, 858, .4, 0, 858, 0],
    #         [np.inf,np.inf, 853.2, np.inf, 1.5, np.inf,861, 2.5, np.inf, 861, np.inf])
    popt, pcov = scipy.optimize.curve_fit(_nickel_ds_voigt, x, y_br_subtracted, p0=guess)
    print(popt)
    residual = y - (_nickel_ds_voigt(x, *popt))-shirley

    fig = plt.figure(5, figsize=(4, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    gs.update(hspace=0)

    ax1.plot(x, y, "ro")
    ax1.plot(x, _nickel_ds_voigt(x, *popt) + shirley, 'k--')
    # residual
    ax2.plot(x, residual, "bo")
    # Background
    ax1.plot(x, shirley, '^')
    plt.xlim((max(x), min(x)))
    plt.show()

    return popt

def _donsunj_conv_gauss(x, a, F, E, amp1, sigma1):
    gauss = amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - E) / sigma1) ** 2)))
    ds = np.cos(np.pi*a/2+(1-a)*np.arctan((x-E)/F))/(F**2+(x-E)**2)**((1-a)/2)
    return gauss+ds

def _nickel_ds_voigt(x, a, F, E, amp1, sigma1, ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2):
    return _donsunj_conv_gauss(x, a, F, E, amp1, sigma1)+_1Voigt(x, ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2)

def fit_1Voigt(x,y,guess):
    """Fits 1 voigt peak to the x and y data and returns the optimal parameters"""
    #guess = [ampG1, cenG1, sigmaG1,ampL1, cenL1, widL1]

    popt_1voigt, pcov_1voigt = scipy.optimize.curve_fit(_1Voigt, x, y, p0=guess)

    return popt_1voigt

def plotall(master, plotrange=(0, 1000)):
    """takes a dataframe formatted by the file xps_plotter and visualizes all of the plots on the same figure"""

    fig = plt.figure()
    ax1 = fig.add_subplot()

    for j in range(int(len(master.columns) / 2)):
        tempx = pd.DataFrame
        tempy = pd.DataFrame

        for i, col in enumerate(master.columns):
            test = (master.iloc[0, i] <= plotrange[1])
            if (int(col.split(" ")[1]) == j) and ('binding' in col):

                tempx = master[col]
            elif (int(col.split(" ")[1]) == j) and ('intensity' in col):

                tempy = master[col]

        if tempy.empty:
            print('array empty')
            continue
        tempx = tempx.dropna()
        tempy = tempy.dropna()
        x = tempx.to_numpy()
        y = tempy.to_numpy()
        ax1.plot(tempx, tempy)
    return fig

#def subtractNi(xarray,yarray)