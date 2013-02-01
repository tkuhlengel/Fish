#!/usr/bin/python2.7
## \brief Module implementing ITK tools.
#Created on Jan 30, 2012
#
#\author Trevor Kuhlengel


import SimpleITK as si


def readFile(filename):
    '''
    Can read Nrrd, insight files without any additional information about type and size.
    Returns an ITK Image class object.
    
    '''
    reader=si.ImageFileReader()
    reader.SetFileName(filename)
    return reader.Execute()