import os, struct, re

import numpy as np
import scipy as sp
import matplotlib as mpl

#Need to get some local imports
os.chdir("../")


import main,fish,processing,graphing
import fish
#import cython
#%load_ext cythonmagic
import gc
gc.enable()

from scipy import ndimage

'''
This is an experiment in the processing of the fish using some of the tools in scipy.ndimage.
For the experimental purpose, see trevor's lab notebook page 17.
'''
#Stage 1. Load Files

#filename="/scratch/joinedImages/resized/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_x292_y292_z1047_cubic-0-5_10.0um.nrrd"
#filename="/scratch/joinedImages/uint16/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_uint16_x586_y586_z2095_5.0um.nrrd"
filename="/scratch/joinedImages/uint16/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_uint16_x293_y293_z1047_10.0um.nrrd"
header,npdata=main.unpacker(filename)
npdata=npdata.reshape(header["dsize"][::-1])

#Automatic thresholding of the file
thresh=processing.autothreshold(npdata)
print(npdata.shape)

mask=npdata>thresh
#threshed=np.where(mask, npdata, npdata.min)

#Do some binary morphology to get rid of excess stuff



result=processing.binaryClosing(mask,structure=None, iterations=15)
result2=processing.binaryOpening(result,kernel=None, iterations=1)


clean_mask=np.logical_and(result2, mask)

full_connectivity=True

#Make some different structuring arrays.  3x3x3, 5x5x5 and 7x7x7
if not full_connectivity:
    structure1=ndimage.morphology.generate_binary_structure(3, 1)

    
else:
    structure1=np.ones((3,3,3), dtype="bool8")

labels1,labelcount1=ndimage.label(mask,structure1)
found_objects=ndimage.find_objects(labels1)




#Requires uint8 or uint16 data
labelmap=ndimage.watershed_ift(npdata, labels1, structure1)








#OUTPUT DATA TO FILE
reload(main)
outputfiledir="/mnt/scratch/tkk3/joinedImages/uint16/"
outputfilename="rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_int32_labelmap_xy293_z1047_10.0um.raw"
f=open((outputfiledir+outputfilename),mode='wb')
f.write(labelmap.tostring())
f.close()