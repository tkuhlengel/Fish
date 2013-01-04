'''
Created on Nov 16, 2012

@author: trevor
'''
import Cython

import numpy as np
cimport numpy as np
#Calculating a 2D center of Mass
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
#BTYPE = np.bool8
#ctypedef np.bool_t BTYPE_t
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
    for y in range(ysize):
        for x in range(xsize):
            pt=f[y,x]
            if pt>0.0:
                xsum+=pt*x
                xmass_sum+=pt
                ysum+=pt*y
                ymass_sum+=pt
    if xmass_sum != 0.0 and ymass_sum != 0.0:
        xcom=xsum/xmass_sum
        ycom=ysum/ymass_sum
    else:
        xcom=np.NaN
        ycom=np.NaN
    cdef np.ndarray res=np.zeros((2,),dtype=DTYPE)
    res[0]=xcom
    res[1]=ycom
    return res

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

    cdef int x=0,y=0
    
    #Variables for unweighted
    cdef int xcount=0,ycount=0

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
        xcom=np.NaN
        ycom=np.NaN
    
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

def feature_normalize(np.ndarray X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    cdef int n_c = X.shape[1]
    cdef int i=0
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
            temp.shape[0],temp.shape[1] = m, 1

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

#\LINEAR REGRESSION ATTEMPT