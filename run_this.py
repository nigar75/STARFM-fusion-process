## -*- coding: utf-8 -*-
##
###Originally authored by: Nikolina Mileva, on Thu Dec  6 15:33:49 2018.
###Customised by Abudu & Parvin: Dec 21 2022.
##

import time
import rasterio
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'src' )
sys.path.append( mymodule_dir )
import starfm4py as stp
from parameters import (path, sizeSlices)




start = time.time()



product = rasterio.open('C:/Users/nsp095/Downloads/STARFM/2011_Jan_May_landsat.tif')
profile = product.profile
LandsatT0 = rasterio.open('C:/Users/nsp095/Downloads/STARFM/2011_Jan_May_landsat.tif').read(129)
MODISt0 = rasterio.open('C:/Users/nsp095/Downloads/STARFM/2011_Jan_May_modis.tif').read(129)
MODISt1 = rasterio.open('C:/Users/nsp095/Downloads/STARFM/2011_Jan_May_modis.tif').read(134)
                  #'C:/Users/nsp095/Downloads/STARFM/MODIS_NIGHT_07_MAY_2022_1.tif').read(1)
# Set the path where to store the temporary results
path_fineRes_t0 = 'Temporary1/Tiles_fineRes_t0/'
path_coarseRes_t0 = 'Temporary1/Tiles_coarseRes_t0/'
path_coarseRes_t1 = 'Temporary1/Tiles_fcoarseRes_t1/'

# Flatten and store the moving window patches
fine_image_t0_par = stp.partition(LandsatT0, path_fineRes_t0)
coarse_image_t0_par = stp.partition(MODISt0, path_coarseRes_t0)
coarse_image_t1_par = stp.partition(MODISt1, path_coarseRes_t1)

print ("Done partitioning this block, please wait!")

# Stack the the moving window patches as dask arrays
S2_t0 = stp.da_stack(path_fineRes_t0, LandsatT0.shape)
S3_t0 = stp.da_stack(path_coarseRes_t0, MODISt0.shape)
S3_t1 = stp.da_stack(path_coarseRes_t1, MODISt1.shape)

shape = (sizeSlices, LandsatT0.shape[1])

print ("Done stackingthis block, please wait!")

# Perform the prediction with STARFM
for i in range(0, LandsatT0.size-sizeSlices*shape[1]+1, sizeSlices*shape[1]):
    
    fine_image_t0 = S2_t0[i:i+sizeSlices*shape[1],]
    coarse_image_t0 = S3_t0[i:i+sizeSlices*shape[1],]
    coarse_image_t1 = S3_t1[i:i+sizeSlices*shape[1],]
    prediction = stp.starfm(fine_image_t0, coarse_image_t0, coarse_image_t1, profile, shape)
    
    if i == 0:
        predictions = prediction
        
    else:
        predictions = np.append(predictions, prediction, axis=0)
  

# Write the results to a .tif file   
print ('Writing product...')
profile = product.profile
profile.update(dtype='float64', count=1) # number of bands
file_name = path + 'landsat_ 14May_11.tif'

result = rasterio.open(file_name, 'w', **profile)
result.write(predictions, 1)
result.close()


end = time.time()
print ("Fusion is completed in:", (end - start)/60.0, "minutes!")

# Display input and output
plt.imshow(LandsatT0)
plt.gray()
plt.show()
plt.imshow(MODISt0)
plt.gray()
plt.show()
plt.imshow(MODISt1)
plt.gray()
plt.show()	
plt.imshow(predictions)
plt.gray()
plt.show()
