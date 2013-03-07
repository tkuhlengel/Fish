#!/usr/bin/python
##
# \file hough.py
# \package src
# \ingroup image
#
# \date 2013
#
# \author Trevor Kuhlengel <tkuhlengel@gmail.com>
## \brief Performs a 2D line detection

#Needed packages
import numpy as np

#Local imports
import processing,itktest



def hough_lines(d2image, d2_iscanny=False, pixel_step_size=1.0, angle_samples=181, ro_samples=200, N_return=5,
                return_all=False, **kwargs):
    if not d2_iscanny:
        data = itktest.canny_edge_detector(d2image, returnNP=True,  **kwargs)
    else:
        data = d2image
    coords=processing.get_coords(data)
    #Number of divisions in 180 degrees to use for angle sampling.
    #Largest value is the hypotenuse of the image dimensions
    romax = np.sqrt(float(data.shape[0] ** 2 + data.shape[1] ** 2))
    #The evenly spaced bins of RO and theta
    rospace=np.linspace(0.0, romax,    num=ro_samples,    endpoint=False, retstep=False)
    theta = np.linspace(0.0, np.pi/2., num=angle_samples, endpoint=False, retstep=False)
    theta_ind=np.arange(0,len(theta),dtype="uint16")
    
    #Generate the accumulator space. X axis
    accum = np.zeros(( len(theta),len(rospace)), dtype="uint32")
    accum_x=np.ravel(accum)
    #Something to store extra results
    
    #For each coordinate, apply theta equation to calculate ro
    ro=[]
    #print(theta_ind)
    for x, y in coords:
        #Perform the transform to this space
        ri=x * np.cos(theta) + y * np.sin(theta)
        ro.append(ri)
        ri_sort=np.searchsorted(rospace, ri)
        #Get the equivalent coordinates for the output
        #Make the 2-D results into 1-D results for easy indexing of the accumulator
        index=np.ravel_multi_index((theta_ind,ri_sort),accum.shape, mode="raise")
        accum_x[index]+=1
    #Index locations where the greatest values are. Returned as numpy arrays of point coordinates
    indices=np.argwhere(
        accum >= processing.findNthGreatestValue(accum, count=N_return)
        )
    
    ro_the=np.column_stack((rospace[indices[:,1]],theta[indices[:,0]]))

    if return_all:
        return ro_the, (accum,rospace,theta)
    else:
        return ro_the

##\brief Finds circles in an image.
#  \param d2image A 2-dimensional numpy image array. This function runs faster if the image is already the
#   canny edge detected image in boolean form. If it is not, it will be run through the detector
#   \see itktest.canny_edge_detector
#  \param d2_iscanny Boolean indicating whether the image input has already undergone canny edge detection.
#  \param radius_samples
#  \param xy_samples
#  \param N_return Number of top candidates to return.
def hough_circles(d2image, d2_iscanny=False, radius_samples=200, xy_samples=[200,200], N_return=5,
                return_all=False, **kwargs ):
    if not d2_iscanny:
        data = itktest.canny_edge_detector(d2image, returnNP=True,  **kwargs)
    else:
        data = d2image
    
    data=np.asanyarray(data,  dtype="bool8")
    coords = processing.get_coords(data)
    #We're going to assume that Radius is less than the hypotenuse of the image 
    #This is convenient, because it only allows for circles completely contained in the image
    r_max= np.sqrt(float(data.shape[0] ** 2 + data.shape[1] ** 2))
    x0=np.linspace(0.0, d2image.shape[1], num=xy_samples[0], endpoint=False, retstep=False)
    y0=np.linspace(0.0, d2image.shape[0], num=xy_samples[1], endpoint=False, retstep=False)
    r2_space=np.square(np.linspace(0.0,r_max, num=radius_samples, endpoint=False))
    #r2_space=np.linspace(0.0,r_max, gridsize, endpoint=True)
    
    #Create a list of index arrays
    x0_ind=np.arange(0, len(x0), dtype="uint16")
    y0_ind=np.arange(0, len(y0), dtype="uint16")
    r2_ind=np.arange(0, len(r2_space),dtype="uint16")

    accum=np.zeros((len(r2_space),len(y0),len(x0)),dtype="uint32")
    #Create a 1-D view
    accum_x=np.ravel(accum)
    
    xy=ndRightJoin(x0,y0)
    xy_ind=ndRightJoin(x0_ind, y0_ind)
    
    #Choosing to split them for the equation
    x0_eq,y0_eq=xy[:,0], xy[:,1]
    x_ind,y_ind=xy_ind[:,0], xy_ind[:,1]
    #Now going to 
    
    
    for x, y in coords:
        #Perform the transform to this space
        r2=(x-x0_eq)**2+(y-y0_eq)**2
        r2_sort=np.searchsorted(r2_space,r2)
        #Get the equivalent coordinates for the output
        #Make the 2-D results into 1-D results for easy indexing of the accumulator
        index=np.ravel_multi_index((r2_sort, y0_ind, x0_ind),accum.shape, mode="raise")
        accum_x[index]+=1
    
    indices=np.argwhere(
        accum >= processing.findNthGreatestValue(accum, count=N_return)
        )
    r2_y_x=np.column_stack((r2_space[indices[:,0]],y0[indices[:,1],x0[indices[:,2]]]))
    
        
        
    if return_all:
        return r2_y_x,(accum, x0,y0,r2_space)
    else:
        return r2_y_x

def ndRightJoin(*vectors):
    '''
    Right Join a sequence of N vectors into an array of length YxN, 
    ''' 
    mesh=np.meshgrid(*vectors, indexing="ij")
    stacked=[matrix.flatten() for matrix in mesh]
    return np.column_stack(stacked)
    
    return np.reshape(stacked,  (-1,len(vectors)))

def hough_circle_setup():
    pass
def hough_circle_work(func, accum, coords, linspace_eq_result, ):
    pass
