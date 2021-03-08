import os
import numpy as np
import mat73


if __name__ == "__main__":
    # Load new Matlab 7.3 data format (hdf5 file)

    # go to file location and use mat73 package to get data
    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    data = mat73.loadmat('Pentaerythritol_1000ug_CD-01.mat')
    # how to properly read the hypercube, wns and badPix 
    cube = np.array(data['cube']['betterRefl'])
    wn = np.array(data['cube']['wn'])
    badPix = np.array(data['cube']['badPix'])
