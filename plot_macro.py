import scipy.io as sio
import numpy as np
import seaborn as sns
import os  # , sys
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from skimage import segmentation
from collections import defaultdict
from scipy import ndimage as ndi
from timeit import default_timer as timer
# from sklearn.preprocessing import StandardScaler


def impute_blanks(data):
    with open('./badPix.pkl', 'rb') as fp:
        badPix = pickle.load(fp)

    for pixel in badPix:
        data[pixel[0]][pixel[1]] = (
                                    data[pixel[0]+1][pixel[1]] +
                                    data[pixel[0]-1][pixel[1]] +
                                    data[pixel[0]][pixel[1]+1] +
                                    data[pixel[0]][pixel[1]-1]) / 4

    return data


def get_cube(file):
    data = sio.loadmat(file)
    cube = np.array(data['cube']['betterRefl'][0][0])
    cube[np.isnan(cube)] = 0
    wn = np.array(data['cube']['wn'][0][0])
    return cube, wn


def normalize(data):
    # fancy normalized data
    # normal = StandardScaler()
    # normal_data = normal.fit_transform(data)
    return data  # normal_data


def preprocess(cube):
    # fill in blank pixels
    filled_in_data = impute_blanks(cube)
    # boxcar smoothing
    smoothed_data = ndi.uniform_filter(filled_in_data, size=3, mode='constant')
    # smoothed_data = smooth_data(filled_in_data)
    # normalize the data
    normal_data = normalize(smoothed_data)
    return normal_data


def get_segments(data):
    # do segmentation here
    time1 = timer()
    labels = segmentation.slic(data, n_segments=20, compactness=0.5,
                               convert2lab=False, slic_zero=False,
                               start_label=1, max_iter=10)
    time2 = timer()
    # segmentation.slic(data, n_segments=20, compactness=1, convert2lab=False,
    #    slic_zero=False, start_label=1, max_iter=25)
    # time3 = timer()
    # segmentation.slic(data, n_segments=20, compactness=1,
    #     convert2lab=False, slic_zero=True, start_label=1, max_iter=35)
    # time4 = timer()

    print('SLIC Algs: ', time2 - time1)  # , time3 - time2, time4 - time3)

    return labels


def get_spectra(segment_labels, cube, wn):
    # get avg spectra for each segment
    segment_spectra = defaultdict(list)
    num_labels = np.max(segment_labels) + 1  # python offset
    for y, col in enumerate(segment_labels):
        for x, row in enumerate(col):
            segment_spectra[segment_labels[y][x]].append(cube[y][x])

    spectra = defaultdict(list)
    for cluster in range(1, num_labels):
        spectra[cluster] = np.mean(segment_spectra[cluster], axis=0)

    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn  # [0]
    multispectra.set_index('wn')

    return multispectra


def get_axis(cube):
    maxes = np.max(cube, axis=2)
    print(np.percentile(maxes, 0))
    print(np.percentile(maxes, 100))
    lower = np.percentile(maxes, 10)
    upper = np.percentile(maxes, 90)
    return lower, upper


def create_plots(image_data, segment_data, spectra_data, lower, upper):
    # create 3 plots, save out output

    # looking at median value for best 'image'
    image = np.median(image_data, axis=2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))
    # ,sharey=True)

    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    sns.heatmap(image, vmin=lower, vmax=upper, ax=ax1)
    num_seg = len(np.unique(segment_data))
    cmap = sns.color_palette("hls", num_seg)
    ax2.tick_params(labelsize='small')
    sns.heatmap(segment_data, cmap=cmap, ax=ax2)
    ax3.tick_params(labelsize='medium')
    keys = list(spectra_data.columns)
    for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
        ax3.plot(spectra_data['wn'], spectra_data[key], label=key, linewidth=5,
                 color=cmap[c])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig("multiplot_test_0.png")


if __name__ == "__main__":
    # go to data, and get data
    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    # cube, wn = get_cube('SetD_sample16_03-HP-02')  #for debugging purposes
    # cube, wn = get_cube('SetD_sample16_01-HP-02')  #for debugging purposes
    cube, wn = get_cube('SetD_sample18_02-HP-02')  # for debugging purposes
    # cube, wn = get_cube('paper_grid')  # for debugging purposes
    # cube, wn = get_cube(sys.argv[1])
    lower, upper = get_axis(cube)

    img_data = preprocess(cube)
    segment_data = get_segments(img_data)
    spectra_data = get_spectra(segment_data, impute_blanks(cube), wn)

    create_plots(img_data, segment_data, spectra_data, lower, upper)
