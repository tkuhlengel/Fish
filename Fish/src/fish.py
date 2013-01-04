'''
Created on Oct 4, 2012

Step1: Descriptive Statistics 
    scipy.ndimage.measurements: center_of_mass()
    Scipy.ndimage.measurements: extrema(input)
\author Trevor Kuhlengel
'''
import numpy as np, scipy as sp
#import teem
#Local Imports
import main
from main import splitNrrdData

class Fish(object):
    '''
    classdocs
    '''


    def __init__(self, filename, datatype="float32", endian="little", axes_shape=[]):
        '''
        Constructor
        
        Sets up the class for use.  Filename of the nrrd or raw file is required
        Axes shape is required for RAW files, and should be a list of integers denoting
        the length of the axes.as is datatype.
        '''
        self.filename=filename
        self.hdrinfo=dict()#Data normally contained in a .nrrd or .nhdr header, ignoring the first line and comments
        self.data=np.arange(0,10,1)
        self.shapegrid=np.arange(0,10,1)
        self.datatype=datatype
        self.fileextension=""
        self.axes=axes_shape
        
        #Processing triggers to ensure things are not called too early.
        self.loaded=False
        self.normalized=False
        
        #Endianness
        if endian.lower().startswith("big"):
            self.endian=">"
        else: #assume little endian
            self.endian="<"

    def load(self):
        '''
        Loads the binary data from the raw image.  Tries to load .nrrd 
        '''
        if self.loaded:
            raise Exception("This fish has already been loaded.  Check to make sure you are not calling Fish.load() more than once.")
        t=type(self.filename)
        if t==type(str()):
            bindata=main.loadFile(self.filename)
            self.fileextension=main.parseExtension(self.filename)
        else:
            #TODO Make it so that it can handle more than one file
            raise Exception("I can't handle more than one file yet")
        
        #Get header data
        if ".nrrd" in self.fileextension or ".nhdr" in self.fileextension:
            self.hdrinfo,raw=splitNrrdData(self.bindata)
            self.axes=[i for i in reversed(self.hdrinfo["dsize"])]
        

        elif ".raw" in self.fileextension:
            raw=bindata
            if len(self.axes)==0:
                self.axes=main.getDimFromFile(self.filename)
                assert len(self.axes!=0), "Could not parse the The number of axes must be greater than zero."
        lineardata=main.unpacker2(raw, offset=0)
        self.shapegrid=np.reshape(lineardata, self.axes, order)
        self.loaded=True
    def normalize(self):
        if self.normalized:
            raise Exception("This fish has already been normalized.  \
            Check to make sure you are not calling Fish.normalize() more than once.")
        elif not self.loaded:
            raise Exception("This fish has not yet been loaded and cannot be normalized.  \
            Check to make sure you are not calling Fish.normalize() before Fish.load().")
        main.scaleToZeroOne(self.shapegrid)

    
    def setDimensions(self, axes=[0]):
        '''
        Sets the fish dimensions from any of a number of data sources
        '''
        
        if len(axes)>0 and axes[0]>0:
            self.data.reshape(axes.reverse())
        else:
            self.fileextension=main.parseExtension(self.filename)
            if self.fileextension==".nrrd" or self.fileextension ==".nhdr":
                #Overwrite axes input
                #TODO See if this is necessary as time goes on.
                self.axes=self.hdrinfo["dsize"]
            elif "raw" in self.fileextension:
                dims= main.getDimFromFile(self.filename)    
                if dims is not None:
                    self.axes=dims
                    self.data.reshape(dims)
        
    def useHdr(self):
        pass
    def getSlice(self, slicenumber, plane=[0,1]):
        pass
    def trimFish(self):
        '''
        Trim down the fish to the needed dimensions, removing all unnecessary data
        '''
    
    
