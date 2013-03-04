#!/usr/bin/python3

## \package processing
# Created on Oct 17, 2012
#
# \author: Trevor Kuhlengel

#import sys
import itertools as it

import numpy as np
import scipy as sp
from scipy import ndimage
#from scipy.ndimage.morphology import *
import matplotlib as mpl




import graphing
import itktest

from numpy.core.defchararray import count

##   \brief: Method that isolates a fish into a smaller volume.  Can use different
#    difference detections to find the edge of the signal from the noise
#    
#    \param[in] nparray3D 3 dimensional Iterator object returning 2D slices of the 3D volume 
#    \param[in] mode String denoting the statistical calculation to use to
#         differentiate fish from non fish.
#    Currently supported modes
#        "variance" or "var"- Variance of a 2D slice. Default option.
#        "stdev" - Difference in standard deviation
#        "absolute" - Absolute difference between minimum and maximum in a slide:
#            Advantage: Very sensitive to single voxel changes
#            Disadvantage: Very easy to fool if the image has high value noise.
#        "median" - Median value of a given slice.
#    \param[in] fractional_diff The difference of the mode from the previous slide that
#        defines the threshold value.        
#    \param[in] step Number of slices to step for each difference check.
#        Good to speed up processing of high density array.
#        Increasing this number decreases precision.
#    \param[in] compareto number of slices before or ahead to compare the difference to.
#        positive values are lookahead, negative are look behind.
#       
#    \note Advantages of this method:
#        Trims away noise from images based on 
#    Drawbacks of this method:
#        - Unreliable for noisy images or images with weak boundaries between 
#          tissue and container
def isolateFishByDifference(nparray3D, mode="variance", fractional_diff=0.05,
                            step=1, compareto= -1):

    shp = np.shape(nparray3D)
    bounds = np.zeros([len(shp), 2])

    # Set the default upper bounds to be the length of each axis
    for i in range(len(shp)):
        bounds[i, 1] = shp[i]
    slices = nparray3D
    # aswaps = np.array([[0, 1], [0, 2]])
    # Primary axis limits
    _isolateFish(slices, mode, fractional_diff, step, compareto)
    # Secondary axis limits
    # Tertiary axis boundaries


def _isolateFish(npdata, mode, frac_diff, step, compareto):
    diff = []
    counter = 0
    breaker = False
    # for i in scaledgrid[0:1047:1,:,:]:
    for i in npdata[::step]:
        diff.append(_isolateFishEngine(i, mode))
        current = diff[ counter ]
        last = diff[ counter + compareto ]
        currdiff = current - last

        # See if the threshold is met this time
        test = currdiff > (frac_diff * last)
        print("Step = {}\nLast Value = {}\nCurrent Difference = {}\nCurrent Value = {}\n".format(counter + 1, last, currdiff, current))

        if test:

            breaker = True

        if breaker:
            print("I think I caught a fish!")
            break
        counter += 1
    return
def _isolateFishEngine(nparray2D, mode):
    result = []
    if type(mode) == type(str()):
        result.append(_isolateFishMode(nparray2D, mode))
    elif type(mode) in [type(list()), type(tuple()), type(np.arange(1))]:
        if len(mode) == 0:
            raise Exception("No mode specified.")
        elif len(mode) == 1:
            result.append(_isolateFishMode(nparray2D, mode[0]))
        else:
            for m in mode:
                result.append(_isolateFishMode(nparray2D, m))
    return result

# #  @brief Returns the desired mode calculation for a single characterization mode. 
#    Mode must be a string matching one of the mode strings described in 
#    isolateFishByDifference docstring.
#    @param[in] nparray2D The 2D NumPy array/matrix containing the slice 
#    @param[in] mode The name of the calculation to use as detailed by
#        processing.isolateFishByDifference
def _isolateFishMode(nparray2D, mode):
    if type(mode) == type(""):
        modelow = mode.tolower()
        if "var" in modelow:
            return np.var(nparray2D, axis=None)
        elif modelow == "stdev":
            return np.std(nparray2D, axis=None)
        elif modelow == "absolute":
            return np.ptp(nparray2D, axis=None)
        elif modelow == "median":
            return np.median(nparray2D, axis=None)
        else:
            raise Exception("Mode parameter does not match one of the available modes")
    elif type(mode) == type(bandpass):
        return mode(nparray2D.flat)
    else:
        raise Exception("Mode parameter does not match one of the available descriptor types")

## \brief Perform high or low bandpass on the passed in array. The default
#  is 3 standard deviations from the mean.  Absolute values can be passed in
#  using min or max.
# \param[in] nparray 
# \param[in] mode    'high', 'low', or both are accepted
# \param[in] max_val    The maximum value to allow inclusive.
# \param[in] min_val    Minimum value to allow inclusive range.
# \param[in] value    Value to set cut out values to. Defaults to 0.0
# \param[in] stdevs Number of standard deviations to use for the bandpass,
#    cutting out all values outside the range indicated.
# \return bandpassed NumPy array that has all values out of the range Set
# to the default value
# \note Processing time can be decreased moderately with the selection of 
#    min and/or maximum values prior to function call. 
def bandpass(nparray, mode="both", max_val=None, min_val=None, value=0.0, stdevs=4.0):
    modes = ["both", "low", "high", "mid"]
    assert mode.lower() in modes, "Argument Error: Mode not a known argument"
    m = modes.index(mode)
    mid = False
    if m == 0:
        low, high = True, True
    elif m == 1:
        low, high = True, False
    elif m == 2:
        low, high = False, True
    elif m == 3:
        low, high, mid = False, False, True

    if (low and min_val is None) or (high and max_val is None):
        mean = np.mean(nparray.flat)
        stdev = np.std(nparray.flat)
    resultarray = nparray.flatten()
    nullarray = np.zeros(resultarray.shape)
    if value != 0.0 and type(value) == type(float()):
        nullarray[:] = value
    if low:
        if min_val is None:
            min_val = mean - stdevs * stdev
        indexmod = resultarray < min_val
        resultarray = np.where(indexmod, 0.0, resultarray)
    if high:
        if max_val is None:
            max_val = mean + stdevs * stdev
        indexmod = resultarray > max_val
        resultarray = np.where(indexmod, nullarray, resultarray)
    del indexmod, nullarray
    return resultarray.reshape(nparray.shape)

## \brief Divides data based on element value, relative to the provided threshold.
#    \param npdata A numpy array of any type.
#    \param threshold A float value above which the values are set to 1.0
#    \param binary Optional parameter indicating if the user only wants the
#        binary mask back. Defaults to false. Skips the actual calculation if used
#    \param mode Optional parameter defining which mode to use for thresholding.
#        Present Options are:
#            'binary' Sets all values in the image to zero or one.
#            'upper'  Sets all values above threshold to upperval, default = 1.0
#            'lower'  Sets all values below or equal to threshold to lowerval, default = 0.0
#    \param upperval Scalar value or numpy array matching shape of npdata to use for upper threshold value
#    \param lowerval Scalar value or numpy array matching shape of npdata to use for upper threshold value  
def threshold(npdata, mask=None, threshold=0.5, upperval=1.0, lowerval=0.0, binary=False, mode="binary", dtype=None):
    '''
    \brief: Takes the input of the object and thresholds the image to 0 and 1. This removes variablility from and image
     and allows for simple algorithms to be used on the data

    '''
    if mask is None:
        mask = npdata > threshold
    if binary:
        return mask
    if dtype is not None:
        result = np.empty(npdata.shape, dtype=dtype)
    else:
        result = np.empty(npdata.shape, dtype=npdata.dtype)
    if "binary" in mode.lower():
        # If the values are greater, set them to one, else zero.
        result = np.where(mask, upperval, lowerval)
    elif "upper" in mode.lower():
        # Sets all values above threshold to one
        result = np.where(mask, upperval, npdata)
    elif "lower" in mode.lower():
        # Sets all values below or equal to threshold to zero
        result = np.where(mask, npdata, lowerval)
    return result


##  \brief Automatically thresholds an array into two segments, by finding local
#    minima on the histogram.  The assumption is that there are two distinct 
#    classes of data in an array, and that there is a measurable difference 
#    between these two data.
#    \param[in] data: numpy array filled with data.
#    \param[in] high: Optional. Value to use for high guesses
#    \param[in] low:
#    \param[in] threshguess: A guess for the threshold value to start with.
#    \param[in] highstd: Number of standard deviations above the mean to use
#        for initial guess. If no high value is given, this is used to generate
#        a guess from statistical analysis of the data. Default = 4.0  
#    \param[in] lowstd: Number of standard deviations above the mean to use
#        for initial guess. If no high value is given, this is used to generate
#        a guess from statistical analysis of the data. Default = 2.0  
#    \param[in] plot: Boolean indicating whether to show a plot of the 
#        calculation at each step.
#    \return Floating point value denoting the theshold that evenly divides
#        the data based on the initial guesses.
def autothreshold(data, threshguess=None, high=None, low=None, highstd=4.0,
                  lowstd=1.0, plot=False):
    '''
    Automatically calculates an optimal threshold value for a work.
    Reference
    2001 - Hu - Automatic Lung Segmentation for Accurate Quantitation of Volume
     X-Ray CT images
    Assumptions in using this method:
    1. There are only 2 types of voxels:
        1) Voxels within the very dense body and chest wall structures
        2) low-density voxels in the lungs or in the air surrounding the body of the subject (non-body voxels)

    This method essentially Divides the Voxel space into these two types of voxels based on threshold.
    Starting off with an educated low and high guess, the equation is as follows:
    Threshold= (low + high)/2

    The space of voxels is then split by voxel value above/below threshold, and the mean of each space is computed.
    The new means are then put into the threshold equation as high and low, respectively.

    Standard guesses are the value for air, and the value for the body voxels. When standardized this translates to -1000 HU

    #Threshhold = 0.00111747210925
    '''
    # Ub is the mean of the body character (high values)
    # Un is the mean of the non-body voxels (low value)

        
    # Make sure that there are both high and low guesses.
    if threshguess is not None:
        thresh = threshguess
    elif (high is None or low is None):
        m = np.mean(data, axis=None)
        st = np.std(data, axis=None)
        
        # If a high/low guess is missing, estimate one.
        if high is None:
            high = m + highstd * st
            ub = "guess of {}".format(high)
        else:
            ub = str(high)
        
        if low is None:
            low = m - lowstd * st
            un = "guess of {}".format(low)
        else:
            un = str(low)
        print("Calculating threshold from {} and {}".format(un, ub))
        print("Ub0 = {}; Un0 = {}".format(high, low))
        thresh = (high + low) / 2
    else:
        thresh = (high + low) / 2
    
    
    print("Threshhold = {}".format(thresh))
    # lastthresh = 1 - thresh
    lastthresh = 0
    if plot:
        graphing.show_hist(data, low, high, thresh,
                           title="Histogram for iteration {}".format(0))
    
    while(lastthresh != thresh and not np.isnan(thresh)):
        
        lastthresh = thresh

        threshmap = data > thresh
        high = np.mean(data[threshmap])
        low = np.mean(data[np.logical_not(threshmap)])
        thresh = (high + low) / 2
        
        print("Low = {:9.7f} High = {:9.7f}".format(low, high))
        print("Threshhold = {}".format(thresh))
        if plot:
            graphing.show_hist(data, low, high, thresh,
                           title="Histogram for iteration {}".format(0))

    return thresh
# # \brief Arbitrary parametric shape hough transform. (Pronounced "Haf" transform)
def hough_transform(image, parametric_eq=r"RO={X}*cos({THETA})+{Y}*sin({THETA})"):
    pass
# # \brief Performs a 2D line detection
def hough_lines(d2image, d2_iscanny=False, pixel_step_size=1.0, angle_samples=181, ro_samples=200):
    # TODO
    if not d2_iscanny:
        data = itktest.canny_edge_detector(d2image, returnNP=True)
    else:
        data = d2image
    coords = get_coords(data)
    # Define the maximum size of ro to be the length of the hypotenuse of the image
    romax = int(
              np.sqrt(
                      float(
                            d2image.shape[0] ** 2 + d2image.shape[1] ** 2
                            )
                      )
              )
    # Because we want to have a variable resolution, divide the output by the pixel step size
    accum = np.zeros((int(romax ), int(angle_samples)), dtype="uint32")
    rospace=np.linspace(0.0, romax, num=ro_samples)
    theta, tstep = np.linspace(0.0, np.pi, num=angle_samples, retstep=True)
    print(tstep)
    accum=np.zeros((1,2), dtype="float32")
    #Here's where the work goes on
    for x, y in coords:
        for ti in range(len(theta)):
            ro = x * np.cos(theta[ti]) + y * np.cos(theta[ti])
            accum_temp=np.column_stack((ro,theta))
        #_update_accumulator(accum, thetabins, ro_step, ro_bins)
        #accum=np.concatenate(accum,accum_temp)
        for i in np.arange(len(ro)):
            accum[ro[i], theta[i]/tstep] += 1
    argarray = np.argsort(accum, axis=None)
    return argarray,accum

##\brief Finds circles in an image.
def hough_circles(d2image, d2_iscanny=False, gridsize=100, **kwargs ):
    if not d2_iscanny:
        data = itktest.canny_edge_detector(d2image, returnNP=True,  **kwargs)
    else:
        data = d2image
    data=np.asanyarray(data,  dtype="bool8")
    coords = get_coords(data)
    #We're going to assume that R is less than 0.5*min(d2image.shape)
    #This is convenient, because it only allows for circles completely contained in the image
    r_max= max(d2image.shape)
    r2_max=r_max**2
    x0, xstep=np.linspace(0.0, d2image.shape[1], endpoint=False, retstep=True)
    y0, ystep=np.linspace(0.0, d2image.shape[0], endpoint=False, retstep=True)
    r2_space=np.square(np.linspace(0.0,r_max, gridsize, endpoint=True))
    #r2_space=np.linspace(0.0,r_max, gridsize, endpoint=True)
    accum=np.zeros((len(r2_space),len(y0),len(x0)),dtype="uint32")
    
    for x,y in coords:
        for yi in range(len(y0)):
            for xi in range(len(x0)):
                #r2=np.sqrt((x-x0[xi])**2+(y-y0[yi])**2)
                r2=(x-x0[xi])**2+(y-y0[yi])**2
                if r2<r2_max:
                    ri=np.searchsorted(r2_space, r2)
                    accum[ri,yi, xi]+=1
    #    for y0i,x0i in it.product(y0,x0):
    #        
    #        ri=np.searchsorted(r2_space, r2)
    #        accum[ri, 
    return accum,(x0,y0,r2_space)

def _update_accumulator(accumulator, additions_np, theta_bins, ro_step,ro_bins):
    pass
       
def get_coords(data):
    '''
    Method to get coordinates of true values in an array.  Uses numpy.indices 
    to generate index spaces.  Very memory efficient, and should be very fast.
    '''
    return np.argwhere(data)

        
def canny_edge_filter(volume):
    # Step 1: Gaussian convolution with 5x5x5
    pass
# #! \brief Finds the centers of discrete objects relative to the image indices from a binary mask. 
def getObjectCenters(mask, labels=None, found_obj=None, structure=np.ones((3, 3, 3))):
    if found_obj is None:
        getObjectSlices(mask, labels=labels, structure=structure)
    
    centers = []
    for i in found_obj:
        zslice, rowslice, colslice = i
        com = ndimage.center_of_mass(mask[i])
        com = (com[0] + zslice.start, com[1] + rowslice.start, com[2] + colslice.start)
        centers.append(com)
            
    return centers

def getObjectSlices(mask, labels=None, structure=np.ones((3, 3, 3))):
    if labels is None:
        labels, labelcount = ndimage.label(mask, structure=structure)
    found_obj = ndimage.find_objects(mask, labels)
    return found_obj

def getLargestSliceIndex(slicelist):
    '''
    '''
    volumes = []
    t = type(slicelist)
    count = 0
    maxv = -1
    maxvi = -1
    if t == list and (slicelist) > 1:
        for slicei in slicelist:
            z = slicei[0].stop - 1 - slicei[0].start
            y = slicei[1].stop - 1 - slicei[1].start
            x = slicei[2].stop - 1 - slicei[2].start
            v = x * y * z
            volumes.append(v)
            if (v > maxv):
                maxv = v
                maxvi = count
            count += 1
    assert maxvi >= 0, "Max index is not greater than the start index, check slice list for negative values"
    return maxvi
        
    
def binaryFillHoles(binarydata, kernel=np.ones((3, 3, 3))):
    
    result = np.zeros_like(binarydata)
    # binary_fill_holes(binarydata, output=result)
    if kernel is None:
        ndimage.binary_fill_holes(binarydata, output=result)
    else:
        ndimage.binary_fill_holes(binarydata, structure=kernel, output=result)
    
    # binary_fill
    return result

def binaryClosing(binarydata, structure=None, iterations=1):
    
    result = np.zeros_like(binarydata)
    if structure is None:
        ndimage.binary_closing(binarydata, iterations=iterations, output=result)
    else:
        ndimage.binary_closing(binarydata, structure=structure, iterations=iterations, output=result)
    return result

def binaryOpening(binarydata, structure=None, iterations=3):
    '''
    Opening [R52] is a mathematical morphology operation [R53] that consists
    in the succession of an erosion and a dilation of the input with the same
    structuring element. Opening therefore removes objects smaller than the
    structuring element.

    Together with closing (binary_closing), opening can be used for noise removal.
    '''
    result = np.empty_like(binarydata)
    
    if structure is None:
        ndimage.binary_opening(binarydata, iterations=iterations, output=result)
    else:
        ndimage.binary_opening(binarydata, structure, iterations=iterations, output=result)
    return result

def projections(volume, func=np.amax):
    # #! \brief Calculates projections along each axis and returns a list
    # filled with numpy arrays of each projection. Generalized to N-dimensional
    # appropriate func values are "numpy.amax", "

    axmax = []

    for axis in np.arange(len(volume.shape)):
        axmax.append(func(volume, axis))

    return axmax

# #@brief Removes the greatest n values of the flattened array
# @param data Numpy Array containing the data you want to sort
# @param count The number of points to change. Optional. Defaults to 0.01% of the data
# @param fraction The fraction of data points to change.  0.01= 1% changed. Default is 0.0001
# @param replacewith The function or value to replace selected values with with
# @returns A copy of the nparray with the greatest <count> values set to the minimum
def removeGreatest(data, count=None, fraction=.0001, mode="max", replacewith=np.min):

    # make a copy of the array in flattened form.
    flat = data.flatten()

    # Count defaults to 0.01% of the total number of voxels
    if count is None:
        count = int(flat.size * fraction)


    # Get an array filled with the indices that would sort the array
    sortedindex = np.argsort(flat, axis=None)

    length = sortedindex.size

    # Make an array of the highest values' indices
    itera = sortedindex[(length - count):length:1]

    # Replace the highest values with the minimum value
    if type(replacewith) in [int, float, str]:
        np.put(flat, itera, replacewith)
    else:
        np.put(flat, itera, replacewith(flat))
    return flat.reshape(data.shape)


def findNthGreatestValue(data, count=None, fraction=.0001, sort_array=None):
    """
    @brief finds the value at the counted distance from the maximum value
    of the array.  pseudo-sorts the array and picks the value a distance
    from the top.

    @param fraction The fraction of the dataset that should be above the value
    @param count    The number of elements that should be greater than the returned value.
    """
    
    # make a copy of the array in flattened form.
    # flat=data.flatten()

    # Count defaults to 0.01% of the total number of voxels
    if count is None:
        count = int(data.size * fraction)


    # Get an array filled with the indices that would sort the array
    if sort_array is None:
        sort_array = np.argsort(data.flat, axis=None)

    length = sort_array.size - 1

    # Make an array of the highest values' indices
    # sortedIndices=sortedindex[(length-count):length:1]

    index = sort_array[length - count]
    return data.flat[index]
# @autojit()
def linewise(image, threshold=0.5):
    # index=np.nonzero(image)
    data = np.copy(image)
    index = image.flat > threshold
    index = index.reshape(image.shape)
    # img = np.copy(image)
    print(data.shape)
    result = []
    counter = 0
    for line in data:

        mini, maxi = 0, len(line)
        print(line)
        if len(np.nonzero(line)) == 0:
            continue
        jc = 0  # jcounter
        nc = 0  # False counter
        for j in range(len(line)):

            if line[j] == True:
                if jc == 0:
                    mini = j
                jc += 1
            elif line[j] == False:
                if jc < 3:
                    jc = 0
                nc += 1

        for j in reversed(range(len(line))):
            if line[j] == True:
                maxi = j

        print("Line = {%4d} Min = {%4d} Max = {%4d}".format(counter, mini, maxi))
        line[mini:maxi] = 1.0
        result.append(line)
        counter += 1
    x = np.concatenate(result)
    print(len(x))
    return x.reshape(image.shape)

#
# _4_connected = numpy.array([[0, 1, 0],
#                             [1, 1, 1],
#                             [0, 1, 0]], dtype=bool)

# def kill_small(binary_image, min_size, structure=_4_connected):
#    label_image, num_objects = sp.ndimage.label(binary_image, structure)
#    # Label 0 is the background...
#    object_labels = np.arange(1, num_objects+1)
#    # Get the area of each labeled object by summing up the pixel  values in the
#    # binary image. (Divide the binary image by its max val to ensure  that the image
#    # consists of 1s and 0s, so that the sum equals the area in pixels.)
#    areas = np.array(np.sum(binary_image / binary_image.max(), label_image, object_labels))
#    big_objects = object_labels[areas >= min_size]
#    # This part will be pretty slow! But I can't think right now how to  speed it up.
#    # (If there are more big objects than small objects, it would be faster to
#    # reverse this process and xor out the small objects from the binary image,
#    # rather than the below, which or's up a new image of just the  large objects.)
#    big_object_image = numpy.zeros(binary_image.shape, dtype=bool)
#    for bo in big_objects:
#        big_object_image |= label_image == bo
#    return big_object_image
x="""
def right_join_py(nestedList, dtype="float32"):
    result = []  # np.array((100,3), dtype="float32")
    #define the columns
    #np.ndarray c1, c2
    for i in range(len(nestedList)):
        if len(nestedList[i][1])==0:
            continue
        try:
            if type(nestedList[i][1][1]) == list :
                #if len(nestedList[i][1])>0:
                nestedList[i][1] = right_join_py(nestedList[i][1])
            #elif type(nestedList[i][1]) == list:
            #   if type(nestedList[i][1][1])==list or type(nestedList[i][1][1]) == np.ndarray:
            #        nestedList[i][1] = right_join_py(nestedList[i][1])
        #elif type(nestedList[i][1]) == np.ndarray:
        #    nestedList[i][1] = right_join(nestedList[i][1])
        except Exception as exc:
            #print(exc)
            pass
        if len(nestedList[i]) > 0:
     
            c2 = np.asarray(nestedList[i][1], dtype=dtype)
            c1 = np.zeros((c2.shape[0],), dtype=dtype)
            c1[:] = nestedList[i][0]
            result.append(np.column_stack((c1, c2)))
    if len(result)>1:
        result=np.concatenate(result)
        
    return result

def get_coordinates_py(image, maxdepth=0, depth=0):
    '''
    Get a list of coordinates indicating trues image mask down to a coordinate list in n dimensions.
    Recursively stacks data and then right joins it.  Should be very memory efficient.
    '''   
    
    #Should only be altered on the first entry.  Assuming we have more than 0 dimensions
    if maxdepth is 0:
        #print(image.shape)
        maxdepth = image.ndim
        assert maxdepth>0, "Error: Max depth was not changed."
    if image.ndim == 1:
        sublist = []
        for i in range(len(image)):
            if image[i]:
                sublist.append(i)
        return sublist
    elif maxdepth > depth:
        sublist = []
        for i in range(image.shape[0]):
            sublist.append([i, get_coordinates_py(image[i], maxdepth=maxdepth, depth=depth + 1)])
    return right_join_py(sublist)
"""
def right_join(nestedList):
    result = []  # np.array((100,3), dtype="float32")
    #define the columns
    #np.ndarray c1, c2
    for listi in nestedList:
        try:
            #if type(listi[1]) == list or type(listi[1][1]) == np.ndarray:
            if type(listi[1][1])==list or type(listi[1][1]) == np.ndarray:
                listi[1] = right_join(listi[1])
            #elif type(listi[1]) == np.ndarray:
            #    listi[1] = right_join(listi[1])
        except Exception as exc:
            #print(exc)
            pass
            
        c2 = np.asarray(listi[1], dtype="float32")
        c1 = np.zeros((c2.shape[0],), dtype="float32")
        c1[:] = listi[0]
        result.append(np.column_stack((c1, c2)))
    return np.concatenate(result)

def get_coordinates(image, maxdepth=0, depth=0):
    '''
    Get a list of coordinates indicating trues image mask down to a coordinate list in n dimensions.
    Recursively stacks data and then right joins it.  Should be very memory efficient.
    '''   
    if maxdepth is 0:
        #print(image.shape)
        maxdepth = image.ndim
    if image.ndim == 1:
        sublist = []
        for i in range(len(image)):
            if image[i]:
                sublist.append(i)
        return sublist
    elif maxdepth > depth:
        sublist = []
        for i in range(image.shape[0]):
            sublist.append([i, get_coordinates(image[i], maxdepth=maxdepth, depth=depth + 1)])
    return right_join(sublist)    

def test():
    # print("Test not yet written, ending program")
    import main, itktest
    # import processing
    testfish = "/scratch/images/J29-33/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd__x218_y232_z1013_orig_10um.nhdr"
    hparts, npdata = main.unpacker(testfish, hdrstring=False)
    npdata = npdata.reshape(hparts["dsize"][::-1])
    
    #Canny Testing
    npcanny = itktest.canny_edge_detector(npdata[85], returnNP=True)
    
    #Hough Lines testing
    #output = hough_lines(npcanny, d2_iscanny=True)
    #print(output)
    
    
    #Coordinate testing
    boolean_data=npdata>18783.5193537
    print(get_coordinates(boolean_data))
    

if __name__ == '__main__':
    test()

else:
    print("Module processing.py loaded successfully")
