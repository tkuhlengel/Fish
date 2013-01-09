#!/usr/bin/python2.7
'''
Entry point into the Fish manipulation program
Copyright (C) 2013 Trevor Kuhlengel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@author: Trevor Kuhlengel

Python Struct module formats
Format    C_Type        Python_type    Standard size    Notes
x        pad_byte       no_value          
c        char           string          8     
b        signed_char    integer         8    (3)
B        unsigned_char  integer         8    (3)
?        _Bool          bool            8    (1)
h        short          integer         16    (3)
H        ushort         integer         16   (3)
i        int            integer         32    (3)
I        unsigned_int   integer         32    (3)
l        long           integer         32    (3)
L        unsigned_long  integer         32    (3)
q        long_long      integer         64    (2), (3)
Q        u_long_long    integer         64    (2), (3)
f        float          float           32    (4)
d        double         float           64    (4)
s        char[]         string         None 
p        char[]         string         None 
P        void*         integer         None (5), (3)


Data type     Description
bool         Boolean (True or False) stored as a byte
int          Platform integer (normally either int32 or int64)
int8         Byte (-128 to 127)
int16        Integer (-32768 to 32767)
int32        Integer (-2147483648 to 2147483647)
int64        Integer (9223372036854775808 to 9223372036854775807)
uint8        Unsigned integer (0 to 255)
uint16       Unsigned integer (0 to 65535)
uint32       Unsigned integer (0 to 4294967295)
uint64       Unsigned integer (0 to 18446744073709551615)
float        Shorthand for float64.
float16      Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
float32      Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
float64      Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
complex      Shorthand for complex128.
complex64    Complex number, represented by two 32-bit floats (real and imaginary components)
complex128   Complex number, represented by two 64-bit floats (real and imaginary components)
'''

import os 
import sys
import struct, math
import numpy as np
import re
import string

#Compression options
import gzip



def loadFile(pathname, mode='rb', buffersize=(1024*1024*64)):
    '''
    Loads a file and returns the binary string representing the data
    '''
    if pathname.endswith(".gz"):
        fobj=open(pathname, mode, buffersize);
        gz=gzip.open(pathname,mode,fobj)
        
        data=gz.read()
        fobj.close()
    else:
        fobj=open(pathname, mode,buffersize);
        data=fobj.read();
        fobj.close();
    
        
    return data;
def parseExtension(strFileName):
    if ".nrrd" in strFileName:
        return ".nrrd"
    elif ".raw" in strFileName:
        return ".raw"
    elif ".nhdr" in strFileName:
        return ".nhdr"
    else:
        return None

def processNhdr(rawbinary):
    '''
    Function to shorten getSerialImages
    Processes the header and 
    '''
    
    header=processNhdrHeader(rawbinary)
    
    
    #Also, we're setting the default header files to the first image
    return header, loadFile(header["data file"])

def processNrrd(rawbinary, returnRawHeader=False):
    '''
    Function to shorten getSerialImages
    @summary: Takes nrrd files and extracts header and raw data from them.
    Calls load on any linked files.
    
    @param rawbinary: Raw binary data of the read nrrd file to be processed.
    @param returnRawHeader: Boolean to indicate whether the raw header data
        should be returned with the function.
    
    @return: 
    '''
    rawheader,header,raw=splitNrrdData(rawbinary)
    
    #Prints the number of entries in the dictionary and the number of 
    #points in the dataset.
    print(len(header), len(raw))

    if "data file" in header:
            
        raw=loadFile(header["data file"])
    if returnRawHeader:
        return rawheader,header,raw
    
    #Implicit else
    return header, raw

def splitNrrdData(raw):
    '''
    @summary: Splits a .nrrd file into the header data and the binary data if available.
    '''
    offset=raw.index("\n\n")
    assert offset<500, "Error in splitNrrdData: the nrrd file header cannot be found"
    
    #In the nrrd format, the header is separated from the data by two return characters
    #\n\n  We want to split this.
    
    #Raw header
    rawhdr=raw[0:offset:1]
    
    #Raw Data
    data=raw[(offset+2)::1]
    print("Header size = {}".format(offset))
    print("Data Starts at {}".format(offset+2))
    
    #Processed header
    hdrparts=processNhdrHeader(rawhdr)
    
    return rawhdr,hdrparts,data

def processNhdrHeader(header):
    '''
    Splits out the header components into a dictionary that can be used by other parts of the program
    
    header should be a raw string containing only the header information from a nrrd, or the
        entirety of an NHDR file.
    '''
    assert len(header)<1000, "There is probably an error in reading the file. Maybe there is no header?"
    hparts=dict()
    #Split each line out
    hdrlines=re.split(r"\n", header)
    #Split each line in two parts
    for line in hdrlines[1::]:
        if line.startswith("#"):
            continue
        x=re.split(r": ", line)
        #print(x)
        hparts[x[0]]=x[1]
    if "sizes" in hparts:
        dsize=hparts["sizes"].split(" ")
        dsize2=[string.atoi(i) for i in dsize] #Convert the string to an integer
        hparts["dsize"]=dsize2
        #print(hparts["dsize"])
    if "type" in hparts:
        x=hparts["type"]
        if "float" in x:
            out="float32"
        elif "ushort" in x:
            out="uint16"
        elif "long long" in x:
            out="int64"
        elif "uint" in x or "ulong" in x:
            out="uint32"
        elif "long" in x or "int" in x:
            out="int32"
        elif "double" in x:
            out="float64"
        elif "unsigned short" in x:
            out="uint16"
        else:
            print("Unknown raw type encountered, assuming float32")
            print(header)
            out="float32"
        hparts["dtype"]=out
            
            
    return hparts

    

    
def getDimFromFile(filname):
    '''
    Attempts to get the dimensions from the filename.  
    
    Usually only needed for raw files, and requires that x,y,and z dimensions are included in the file name
    Format for extraction *_x[0-9]
    '''
    x=re.findall(r"\_[xX]([0-9]+)\_", filname, re.IGNORECASE)
    y=re.findall(r"\_[yY]([0-9]+)\_", filname, re.IGNORECASE)
    z=re.findall(r"\_[zZ]([0-9]+)\_", filname, re.IGNORECASE)
    
    if len(z)==0:
        return None
    #Boundary case where i labeled xyNNN instead of separating
    if len(x) == 0 and len(y)==0:
        xy=re.findall(r"\_[xXyY]{2}([0-9]+)\_", filname, re.IGNORECASE)
        x=xy
        y=xy
    return (x,y,z)
    
    
def writeFile(pathname, bindata, mode="wb"):
    '''
    Writes a binary string to a binary file at pathname
    '''
    fil=open(pathname, mode)
    fil.write(bindata)
    fil.close()
    return True

        
def processFile(filename, ext=None):
    if ext is None:           
        ext=parseExtension(filename)
    
    #Manage the file types by extension
    nhdr,nrrd=False,False
    if ext == ".nhdr":
        nhdr=True
    #nrrd files
    elif ext==".nrrd":
        nrrd=True
        
    #Load the file
    f=open(filename, 'rb')
    raw=f.read()
    #If it is a header file, we save the dictionary and the 
    if nhdr: #NHDR format.
        rawheader=raw
        header,data=processNhdr(raw)
        
        
    elif nrrd: #NRRD file format
        header,data=processNrrd(raw)
    
    else: #assume RAW
        header,data=None,raw
    return header,data
        
        
def getSerialImages(imagefilenames, sort=True, ext=None):
    '''
    Load a series of images from a stack of files. Optionally sort them first.
    
    DIMS is a list where one entry axis can be variable with the number of 
    files on the stack, which is denoted by any character string. The 
    contents of the string are ignored.
        
    
    '''

    if sort:
        imagefilenames.sort()
    if ext is None:
        ext=parseExtension(imagefilenames[0])
    hdr=None #The primary header
    data=[] #The result list
    first=True
    #Load the files in order and process them.
    counter=0
    for filei in imagefilenames:
        header,dat=processFile(filei,ext=ext)
        if first:
            first=False
            hdr=header
        data.append(dat)
        counter+=1
    
    return hdr, "".join(data)

def unpacker(filename, dtype=None, endian=None, sort=True, hdrstring=False):
    '''
    Basic Data Loader. Automatically Loads .raw files, .nrrd files and 
        .nhdr lists, and processes headers.
    @param filename: a string or list of filenames. If not sorted in order,
         the filenames will be sorted by default in alphabetical order.
    @param dtype: The expected data type in the file. .nrrd Headers will 
        automatically be used 
    @param endian: Specify the endianness of the data. Little will be assumed
        if this parameter is None.
    @param sort: If filename is a list of files, if sort is True, the filenames
    @return: Tuple containing the header data as a dictionary and a numpy array
        containing the file data.  The header may be empty for raw files.
    '''
    if type(filename)==list or type(filename)==tuple:
        assert (type(filename[0])==type(str) or type(filename[0])==type(string())),\
            "Input problem: all items of filename must be strings."
        header,data=getSerialImages(filename)
    elif type(filename)==str or type(filename)==type(string()):
        header,data=processFile(filename)
        
    else:
        raise Exception("Parameter filename must be a string or a list of strings.")
    
    #Next, we need the data type that is being used.
    if dtype is None:
        if "dtype" in header:
            dtype=header["dtype"]
        else:
            dtype="float32"
    
    #Convert the raw data into a numpy 
    npd=unpacker3(data, dtype=dtype)
    if "dsize" in header:
        shape=header["dsize"]
        print("Dsize = {}".format(shape))
        npd=npd.reshape([i for i in shape].reverse())
    if hdrstring:
        return header,npd, None
    return header,npd

def unpacker2(binarystring, endian="<",  datatype="f", offset=None):
    '''
    Legacy, fallback only
    Unpacks a binary string according to the pixelformat and returns a list containin
    '''
    siz=struct.calcsize(endian+datatype)
    if offset is not None:
        hdr=binarystring[0:offset:1]
        result=struct.unpack(endian+datatype*(len(binarystring[(offset)::1])/siz), binarystring[(offset)::1])
    else:
        binary=binarystring
        hdr=None
    

    
    return hdr,np.array(result, dtype="float32")

def unpacker3(binarystring, dtype="float32"):
    '''
    Unpacks a binary string into a numpy array.  This transforms the string 
    into a 1D array of type <dtype>. Should be more memory efficient than the other method
    '''

    return np.fromstring(binarystring,dtype=dtype) 
  
def scaleToZeroOne(data, valueToZero=None, valueToOne=None ):
    test=data.flat
    if valueToZero is None:
        min1=np.minimum.reduce(test,axis=0)
    else:
        min1=valueToZero
        
    if valueToOne is None:
        max1=np.maximum.reduce(test,axis=0)
    else:
        max1=valueToOne
        
    
    
    
    scaled=(data.flat-min1)*(1.0/(max1-min1))
    
    del test

    return scaled.reshape(data.shape)
    
def packer(array, endian="<", datatype="f"):
    if type(array) == np.ndarray:
        siz=int(array.size)
    else:
        siz=len(array)
    bindata=struct.pack(endian+datatype*siz, *((array.flatten()).tolist()))
    return bindata

    
def nonzero_normalize(arraydata, maxvalue=1.0, minvalue=None):
    '''
    Shifts the values of the data so that the minimum value is just above zero
    and zero values remain unchanged, and the value is normalized to a max value
    passed in with arguments
    '''
    nonzero=[]
    for i in arraydata:
        if i!=0.0:
            nonzero.append(i)
    mini=min(nonzero)
    maxi=max(nonzero)
    print("Min={}\nMax={}".format(mini,maxi))
    diff=maxi-mini
    scaler=maxvalue/diff
    result=[]
    absmin=sys.float_info.min;
    for i in arraydata:
        if i!=0.0:
            result.append((i+math.fabs(mini))*scaler+absmin)
        else:
            result.append(0.0)
    return result

def getnonzeros(array):
    #result=[]
    #for i in array:
    #    if i!=0:
    #        result.append(i)
    return filter(lambda x: x!=0, array)
def nonzero_count(array):
    #result=0
    return len(filter(lambda x: x!=0, array))

def nonzero_mean(array):
    return math.fsum(array)/nonzero_count(array)

def processfile(pathname, endian="<", datatype="f", outputdir=""):
    rawdata=loadFile(pathname)
    #array=unpacker(rawdata, pixelformat=(endian+datatype))
    array=unpacker2(rawdata, endian, datatype)
    result=nonzero_normalize(array, maxvalue=100)
    outputdir="/home/trevor/Scratch/tkk3/Sam29-2011-10/testoutput/"+pathname[-9:]
    writeFile(outputdir, packer(result, endian, datatype))

if __name__ == '__main__':
    #from findeyes import Fish
    #f=Fish("/mnt/scratch/tkk3/joinedImages/resized/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_xy585_z2095_cubic-0-5_5.0um.nrrd")
    #testfish="/mnt/scratch/tkk3/joinedImages/resized/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_xy292_z1047_cubic-0-5_10.0um.nrrd"
    #testfish="/mnt/scratch/tkk3/joinedImages/manipulated/rec_L55_1D_PT_13p8E_30mm_w5D_x1172_y1020_z1539_0.74um.raw"
    testfish="/mnt/scratch/tkk3/joinedImages/rec_AWBwtlerf_14_03k_30SSD_8_1_xyz605_2.5um.nrrd"
    
    header,npdata=unpacker(testfish)
    y='''
    parser=ArgumentParser(add_help="Program designed to align images of fish from both raw and reconstructed images.")
    parser.add_argument("-i","--inputs", dest="inputlist", type=str, help="Comma separated SamXX directories to read from")
    parser.add_argument("-o", "--output", nargs=1, dest="output", type=str, help="Directory to place the registration text file")
    parser.add_argument("-t", "--subdir", dest="subdir", type=str, default="reconstructed", help="Subdirectory where hdf data files are stored")
    #parser.add_help("")
    args=parser.parse_args()
    arg2=dict(vars(args))
    #print(arg2)
    #print(args.inputlist)
    args1=default_args(arg2)
    main(args1)
    '''