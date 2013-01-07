import os, struct, re

import numpy as np
import scipy as sp
import matplotlib as mpl

#Need to get some local imports
#os.chdir("/home/trevor/workspace/Fish/src/")
#os.chdir("/home/winnen/git/Fish/Fish/src/")


import main,fish,processing,graphing
import fish
#import cython
#%load_ext cythonmagic
import gc
gc.enable()
from scipy import ndimage

testfish="/mnt/wd2/workspace/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_xy292_z1047_cubic-0-5_10.0um.nrrd"
threshguess=0.00111747210925
hparts,npdata=main.unpacker(testfish,hdrstring=False)
#print(hparts["dsize"].reverse())
print(hparts["dsize"])
#print(reversed(hparts["dsize"]))
#print(len(hparts["dsize"]))
#print(npdata.shape)

#Reshape the data
npdata=npdata.reshape(hparts["dsize"][::-1])
print(npdata.shape)


#Threshold
thresh=processing.autothreshold(npdata,threshguess=threshguess)