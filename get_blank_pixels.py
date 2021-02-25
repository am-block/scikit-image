import scipy.io as sio
import numpy as np
import os
import pickle

if __name__ == "__main__":

    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    data = sio.loadmat('paper_grid')
    cube = np.array(data['cube']['badPix'][0][0])

    badPix = []
    for y in range(cube.shape[0]):
        for x in range(cube.shape[1]):
            if cube[y][x] != 0.0:
                badPix.append([y, x])

    with open('badPix.pkl', 'wb') as fp:
        pickle.dump(badPix, fp)
