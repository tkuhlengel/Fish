import numpy as np
import scipy as sp
from scipy import ndimage

#cython imports
cimport numpy as np

#C++ Boolean 
#from libcpp cimport bool
#Python Boolean
from cpython cimport bool

import sys

#Calculating a 2D center of Mass
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

## \brief Function to create a 3D point cloud from any binary mask.
#  \param image A numpy boolean array (containing only ones and zeros) of some data.  Outermost layer will be ignored if any data is in it.
# \param structure A numpy boolean comparison structure.  Defines the region that a point will be compared to. Defaults to a 3x3x3 with connectivity of one.
# \param buffersize an integer denoting how large the numpy output arrays buffer is.
# \returns A tuple containing the edge points of the volume 
def buildPointCloud(np.ndarray image, np.ndarray structure, int buffersize):
    cdef np.ndarray kernel=np.copy(structure).astype("bool8")
    kernel[1,1,1]=False

    cdef int x=0,y=0,z=0,i=0
    
    shape=image.shape
    cdef int zmax=shape[0]
    cdef int ymax=shape[1]
    cdef int xmax=shape[2]
    
    cdef np.ndarray activeRegion
    cdef np.ndarray compare
    
    cdef bool hasTrue, hasFalse
    cdef np.ndarray copycloud=np.zeros_like(image, dtype="bool8")
    cdef np.ndarray results = np.ndarray((buffersize,3), dtype="float32")
    cdef int counter=0
    for z in range(1,zmax-1):
        for y in range(1,ymax-1):
            for x in range(1,xmax-1):
                if image[z,y,x]:
                    hasTrue=False
                    hasFalse=False
                    
                    
                    activeRegion=image[z-1:z+2,y-1:y+2,x-1:x+2]
                    compare=activeRegion[kernel]
                    for i in range(len(compare)):
                        if hasTrue and hasFalse:
                            results[counter,:]=(z,y,x)
                            copycloud[z,y,x]=True
                            
                            #print(z,y,x)
                            
                            counter+=1
                            #Increment the size of the results table as needed.
                            if counter%buffersize==0:
                                results.resize((counter+buffersize,3))
                                #np.resize(counter+buffersize)
                            break
                        if compare[i] and not hasTrue:
                            hasTrue=True
                        elif not compare[i] and not hasFalse:
                            hasFalse=True
                        
    results=np.trim_zeros(results.flat,trim="b")
    results=np.reshape(results,(-1,3))
    
    return (results,copycloud)

def cy_computer_slice_weight(np.ndarray arr):
    #print("Got to start of cy_weight")
    #Change the weight so all weights are non-negative
    f=arr-arr.min()
    
    #Iterator lengths
    cdef int ysize = f.shape[0]
    cdef int xsize = f.shape[1]
    
    #Sum of the weight
    cdef DTYPE_t xmass_sum=0.0, ymass_sum=0.0
    
    #Sum of the mass*sum
    cdef DTYPE_t xsum = 0.0, ysum = 0.0
    
    #Resulting centers of mass
    cdef DTYPE_t xcom=0.0
    cdef DTYPE_t ycom=0.0
    #cdef DTYPE_t mini = f.min()
    cdef DTYPE_t pt=0.0
    
    cdef DTYPE_t zero=0.0
    cdef int x,y
    
    #Variables for unweighted
    cdef int xcount=0
    cdef int ycount=0
    #Before changing x and y to cdef ints, the speed was
    #32.5 ms per run
    #After, it was 20 ms per run
    
    #Before adding the if statement > 0.0, it was 20 ms per run
    #After, it was 19.5 ms per run
    #print("Got to start of for loop")
    for y in range(ysize):
        for x in range(xsize):
            pt=f[y,x]
            #print("Got to inner for loop")
            if pt>0.0:
                #print("XSum")
                xsum+=pt*x
                xmass_sum+=pt
                ysum+=pt*y
                ymass_sum+=pt
    #print("Got to end of for loop")
    #print("Xsum = {}\nXmass_sum = {}".format(xsum,xmass_sum))
    if xmass_sum != 0.0 and ymass_sum != 0.0:
        xcom=xsum/xmass_sum
        ycom=ysum/ymass_sum
    else:
        xcom=0.0
        ycom=0.0
    #print("Defining RES")
    cdef np.ndarray res=np.zeros((2,),dtype=DTYPE)
    #print("Res Declared")
    res[0]=xcom
    res[1]=ycom
    #print("Got to END of cy_weight")
    return res
#def cy_two_dim_linreg(np.ndarray arr):
#    #M
#    #Cost function = 1/2 sum(hypothesis0*x**i - y**i
#    f=arr-arr.min()
#    output=[]
#    coo=scipy.sparse.coo_matrix(arr)
#    for i in range(len(f)):
#        output=gradient_descent(coo[0],coo[1])
                                
        
    
#LINEAR REGRESSEION ATTEMPT
def feature_normalize(np.ndarray X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    cdef int n_c = X.shape[1]
    cdef int i
    cdef np.ndarray mean_r = np.zeros([n_c,],dtype=DTYPE)
    cdef np.ndarray std_r =  np.zeros([n_c,],dtype=DTYPE)

    cdef np.ndarray X_norm = np.zeros([n_c,],dtype=DTYPE)
    cdef DTYPE_t m,s

    
    for i in range(n_c):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])
        mean_r[i]=m
        std_r[i]=s
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r


def compute_cost(np.ndarray X, np.ndarray y, np.ndarray theta):
    '''
    Compute cost for linear regression
    '''
    #Number of training samples
    cdef int m = y.size

    cdef np.ndarray predictions = X.dot(theta)

    cdef np.ndarray sqErrors = (predictions - y)

    cdef np.ndarray J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradient_descent(np.ndarray X, np.ndarray y,np.ndarray theta, float alpha, int num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    cdef int m = y.size,i=0,it=0
    cdef np.ndarray J_history = np.zeros(shape=(num_iters, 1), dtype=DTYPE)
    cdef np.ndarray temp = np.zeros([X.shape[0],],dtype=DTYPE)
    cdef np.ndarray predictions=np.zeros_like(X.dot(theta),dtype=DTYPE)
    cdef np.ndarray errors_x1

    cdef int theta_size = theta.size

    
    

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp=temp.reshape(m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history
#\LINEAR REGRESSION ATTEMPT

def cy_computer_slice_weight_unweighted(np.ndarray arr):
    #print("Got to start of unweighted")
    #Change the weight so all weights are non-negative
    f=arr-arr.min()
    
    #Iterator lengths
    cdef int ysize = f.shape[0]
    cdef int xsize = f.shape[1]
    
    #Sum of the weight
    
    #Sum of the mass*sum
    cdef DTYPE_t xsum = 0.0
    cdef DTYPE_t ysum = 0.0
    
    #Resulting centers of mass and current point
    cdef DTYPE_t xcom=0.0, ycom=0.0, pt=0.0
    #cdef DTYPE_t ycom=0.0
    #cdef DTYPE_t mini = f.min()
    #cdef DTYPE_t pt=0.0

    cdef int x,y
    
    #Variables for unweighted
    cdef int xcount=0
    cdef int ycount=0
    for y in range(ysize):
        for x in range(xsize):
            pt=f[y,x]
            if pt>0.0:
                xsum+=x
                xcount+=1
                ysum+=y
                ycount+=1   
    if xcount > 0 and ycount>0:
        xcom=xsum/xcount
        ycom=ysum/ycount
    else:
        xcom=0.0
        ycom=0.0
    
    cdef np.ndarray res=np.zeros((2,),dtype=DTYPE)
    res[0]=xcom
    res[1]=ycom
    
    return res

def center_volume(np.ndarray npdata):
    #cdef np.ndarray data
    cdef int i
    #if axis!=0:
    #    data=np.swapaxes(npdata,0,axis)
    #else:
    #    data=npdata
    print(npdata.shape[0],npdata.shape[1],npdata.shape[2])
    cdef int size=npdata.shape[0]
    cdef np.ndarray result = np.zeros((npdata.shape[0],2),dtype=npdata.dtype)
    for i in range(size):
        result[i]=cy_computer_slice_weight(npdata[i])
    return result

cdef np.ndarray filledArray
cdef np.ndarray inputArray
cdef int xmax, ymax

def fill_slice(np.ndarray npdata):
    ##Fills from the corner of each binary slice.  Returns the volume mask where 
    # All negative space in the slice attached to 0,0 is highlighted, and everything
    # not connected to that point is negative space (False).
    # @param npdata: Binary mask of a slice. 
    cdef int xpos=0
    cdef int ypos=0
    global filledArray
    filledArray = np.zeros_like(npdata)
    filledArray[:]=False
    global xmax
    global ymax
    global inputArray
    inputArray=npdata
    xmax=npdata.shape[0]
    ymax=npdata.shape[1]
    sys.setrecursionlimit(100000)
    sl_fill(xpos,ypos)
    return filledArray

#def fill_volume(np.ndarray volume, seedpoint, structure=np.ones((1,1,1),dtype="bool8"):
#    

#cdef recursive_fill(np.ndarray volume, int * pos, np.ndarray structure, int fillvalue):
#    f

def sl_fill(int x, int y):
    global inputArray
    global xmax
    global ymax
    #Overflow case
    if x>=xmax or y >= ymax or y < 0 or x < 0:
        return 0
    #Base case
    if inputArray[x,y]:
        return 0
    #recursive case
    else:
        global filledArray
        filledArray[x,y]=True
        #Up
        sl_fill( x , y + 1 )
        #left
        sl_fill(x - 1 , y  )
        #down
        sl_fill(x , y - 1 )
        #right
        sl_fill(x + 1, y )
        #If adding 3d, just add 2 z alterations
    return 0
    
