#!/usr/bin/python2.7
## \brief Module implementing ITK tools.
#Created on Jan 30, 2012
#
#\author Trevor Kuhlengel


import SimpleITK as si
import numpy as np


def readFile(filename):
    '''
    Can read Nrrd, insight files without any additional information about type and size.
    Returns an ITK Image class object.
    
    '''
    reader=si.ImageFileReader()
    reader.SetFileName(filename)
    return reader.Execute()

def canny_edge_detector(volume, returnNP=False):
    '''
    Performs canny edge detection on the input.
    
    @param returnNP indicates whether the return type should be a NumPy volume.
     If false, returns a simpleITK image. 
    '''
    
    #Make sure we have the right type for ITK
    if type(volume)==np.ndarray:
        
        itk_volume=si.GetImageFromArray(np.float32(volume))
        print(type(itk_volume))
        #returnNP=True
    else:
        itk_volume=volume
        
    #Do the edge detection
    canny_edges_volume=si.CannyEdgeDetection(itk_volume)
    
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
    
    