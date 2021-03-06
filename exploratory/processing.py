import scipy.io as sio
import numpy as np
import pickle
import mat73
from scipy import ndimage as ndi


def get_cube(file):
    try: 
        data = sio.loadmat(file)
        cube = np.array(data['cube']['betterRefl'][0][0])
        # cube = cube[:32][:64]  # model streaming format
        wn = np.array(data['cube']['wn'][0][0][20:280])
    except:
        data = mat73.loadmat(file)
        cube = np.array(data['cube']['betterRefl'])
        wn = np.array(data['cube']['wn'])
    return cube, wn


def impute_blanks(data):
    with open('../badPix.pkl', 'rb') as fp:
    # with open('./badPix.pkl', 'rb') as fp:
        badPix = pickle.load(fp)

    print(data.shape)
    print(max(badPix[0]), max(badPix[1]))

    for pixel in badPix:
        data[pixel[0]][pixel[1]] = (
                                    data[pixel[0]+1][pixel[1]] +
                                    data[pixel[0]-1][pixel[1]] +
                                    data[pixel[0]][pixel[1]+1] +
                                    data[pixel[0]][pixel[1]-1]) / 4

    return data


def normalize(data):
    # fancy normalized data
    smushed_data = data.reshape(data.shape[0]*data.shape[1],
                                data.shape[2])

    min_scale = smushed_data.min()
    max_scale = np.percentile(smushed_data, 99)  # 98.5
    # /99 max_scale = smushed_data.max()
    scaled = (smushed_data - min_scale) / (max_scale - min_scale)

    # scaled = smushed_data / np.percentile(smushed_data, 99)
    orig_shape_data = scaled.reshape(data.shape[0], data.shape[1],
                                     data.shape[2])
    return orig_shape_data


def no_nans(cube):
    cube[np.isnan(cube)] = 0
    return cube


def remove_zero(cube):
    cube[np.isnan(cube)] = 0
    # fill in blank pixels
    filled_in_data = impute_blanks(cube)
    # remove blank wns
    nonzero_data = filled_in_data[:, :, :]
    # nonzero_data = filled_in_data[10:120, :, 20:280]  # y: 10:120

    return nonzero_data


def preprocess(data):
    # boxcar smoothing
    smoothed_data = ndi.uniform_filter(data, size=3, mode='constant')
    # normalize the data
    normal_data = normalize(smoothed_data)
    return normal_data


def get_truth_name():
    data = sio.loadmat('truth')
    names = []
    for i in range(8):
        names.append(str(
            data['truth_map']['name'][0][i][0][4:]
        ))
    return names


def get_axis(cube):
    maxes = np.max(cube, axis=2)
    lower = np.percentile(maxes, 10)
    upper = np.percentile(maxes, 90)
    return lower, upper
