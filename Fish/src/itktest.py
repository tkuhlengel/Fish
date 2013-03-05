#!/usr/bin/python2.7

## \brief Module implementing ITK tools.
#Created on Jan 30, 2012
#
#\author Trevor Kuhlengel


import SimpleITK as si
import numpy as np

#===============================================================================
# Decorators
#===============================================================================

def temporaryITK(fn):
    def wrapper(arg1, *args, **kwargs):
        if type(arg1)==np.ndarray:
            itk_volume=si.GetImageFromArray(arg1)
        elif type(arg1) == si.Image:
            itk_volume=arg1
        return fn(itk_volume, *args, **kwargs)
    return wrapper    
        
            
## \brief Wrapper that allows automatic conversion of data into a floating point array
#  Requires that the first argument be the one that needs to be converted.
def npFloatInput(fn):
    def wrapper(arg1, *args, **kwargs):
        #If the object to be converted is of known type
        if type(arg1) in {np.ndarray, list, tuple}:
            modinput=np.asanyarray(arg1, dtype="float32")
        else:
            #Assume the function can handle the type or the error.
            modinput=arg1
        return fn(modinput,*args, **kwargs)
    return wrapper

## \brief Converts the return value
def npBoolReturn(fn):
    def wrapper(*args, **kwargs):

        output=fn(*args, **kwargs)
        if type(output) == si.Image:
            output=si.GetArrayFromImage(output)
        return np.asanyarray(output, dtype="bool8")
    return wrapper

#===============================================================================
# Other useful functions
#===============================================================================
def readFile(filename):
    '''
    Can read Nrrd, insight files without any additional information about type and size.
    Returns an ITK Image class object.
    
    '''
    reader=si.ImageFileReader()
    reader.SetFileName(filename)
    return reader.Execute()

@npBoolReturn
@npFloatInput
@temporaryITK
def canny_edge_detector(volume, returnNP=False, **kwargs):
    '''
    Performs canny edge detection on the input.
    
    @param returnNP indicates whether the return type should be a NumPy volume.
     If false, returns a simpleITK image. 
     
    Options:
    CannyEdgeDetection(Image image1, 
    @keyword inLowerThreshold=0.0 The lower threshold from the sobel filter to contain an edge.
    @keyword inUpperThreshold=0.0 Upper threshold from the sobel filter to mark as an edge 
    @keyword inVariance=std::vector< double >(3, 0.0), 
    VectorDouble inMaximumError=std::vector< double >(3, 0.01)) -> Image
    '''
    
      
    #Make sure we have the right type for ITK
    if type(volume)==np.ndarray:
        
        itk_volume=si.GetImageFromArray(np.float32(volume))
        print(type(itk_volume))
        #returnNP=True
    else:
        itk_volume=volume
        
    #Do the edge detection
    canny_edges_volume=si.CannyEdgeDetection(itk_volume, **kwargs)
    
    if returnNP:
        return si.GetArrayFromImage(canny_edges_volume)
    
    return canny_edges_volume 

#def  Buffer<TImage>::GetArrayFromImage(TImage*) [with TImage = itk::Image<float, 2u>]

if __name__=='__main__':
    import main
    #import processing
    testfish="/scratch/images/J29-33/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd__x218_y232_z1013_orig_10um.nhdr"
    hparts,npdata=main.unpacker(testfish,hdrstring=False)
    npdata=npdata.reshape(hparts["dsize"][::-1])
    canny_edge_detector(npdata, returnNP=True)
    
    