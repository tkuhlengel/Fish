import os, struct, re

import numpy as np
#import scipy as sp
#import matplotlib as mpl



#import main,fish,processing,graphing
#import fish
#import cython
#%load_ext cythonmagic
import gc
gc.enable()
from scipy import ndimage
import cyfuncs


def buildPointCloudCy(image, structure=ndimage.generate_binary_structure(3,1).astype("bool8"), buffersize=100):
    results, copycloud = cyfuncs.buildPointCloud(image, structure, buffersize)
    return results
def buildPointCloudPy(image, structure=ndimage.generate_binary_structure(3,1).astype("bool8"), buffersize=100):

    kernel=np.copy(structure).astype("bool8")
    kernel[1,1,1]=False
    x=y=z=i=0

    zmax,ymax, xmax=image.shape
    
    #cdef np.ndarray activeRegion
    #cdef np.ndarray compare
    
    #cdef bool hasTrue, hasFalse
    
    results = np.ndarray((buffersize,3), dtype="float32")
    counter=0
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
                            counter+=1
                            #Increment the size of the results table as needed.
                            if counter%buffersize==0:
                                results.resize((counter+buffersize,3))
                            break
                        if compare[i] and not hasTrue:
                            hasTrue=True
                        elif not compare[i] and not hasFalse:
                            hasFalse=True
                        
    results=np.trim_zeros(results.flat,trim="b")
    
    
    return results.reshape((-1,3))
                        