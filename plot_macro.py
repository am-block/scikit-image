import scipy.io as sio
import numpy as np
import seaborn as sns
import os, sys, pickle
import pandas as pd
import pysptools.distance as dis  # this is for evluation only
from matplotlib import pyplot as plt  # only for evaluation?
from skimage import segmentation
from collections import defaultdict
from scipy import ndimage as ndi
from timeit import default_timer as timer  # only for evaluation


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
    wn = np.array(data['cube']['wn'][0][0][20:280])
    return cube, wn


def normalize(data, factor):
    # fancy normalized data
    smushed_data = data.reshape(data.shape[0]*data.shape[1],
                                data.shape[2])

    scaled = smushed_data * factor / np.percentile(smushed_data, 100)
    orig_shape_data = scaled.reshape(data.shape[0], data.shape[1],
                                     data.shape[2])
    return orig_shape_data


def remove_zero(cube):
    cube[np.isnan(cube)] = 0
    # fill in blank pixels
    filled_in_data = impute_blanks(cube)
    # remove blank wns
    nonzero_data = filled_in_data[:, :, 20:280]  # y: 10:120

    return nonzero_data


def preprocess(data, factor):
    # boxcar smoothing
    smoothed_data = ndi.uniform_filter(data, size=3, mode='constant')
    # normalize the data
    normal_data = normalize(smoothed_data, factor)
    return normal_data


def get_segments(data):
    # do segmentation here
    # time1 = timer()
    labels = segmentation.slic(data, n_segments=20, compactness=0.5,
                               convert2lab=False, slic_zero=False,
                               start_label=1, max_iter=10)
    # time2 = timer()
    # segmentation.slic(data, n_segments=20, compactness=1, convert2lab=False,
    #    slic_zero=False, start_label=1, max_iter=25)
    # time3 = timer()
    # segmentation.slic(data, n_segments=20, compactness=1,
    #     convert2lab=False, slic_zero=True, start_label=1, max_iter=35)
    # time4 = timer()

    # print('SLIC Algs: ', time2 - time1)  # , time3 - time2, time4 - time3)
    return labels


def spec_distance(spectra, num_labels, skip):
    spec_sid = []

    for cluster in range(1, num_labels):
        sid = []
        if cluster in skip:
            sid.append(0)
            continue
        for cluster2 in range(1, num_labels):
            # if (cluster2 in skip):
            #     continue
            if (cluster == cluster2):
                continue
            sam = dis.SAM(spectra[cluster], spectra[cluster2])
            sid.append(dis.SID(spectra[cluster],
                       spectra[cluster2]) * np.tan(sam))
        spec_sid.append(sid)
    combined_sid = [np.nansum(x) for x in spec_sid]
    for cluster in skip:
        combined_sid.insert(cluster-1, 0)
    sorted_sid = sorted(range(1, len(combined_sid) + 1),
                        key=lambda k: combined_sid[k-1], reverse=True)
    topquart = int(len(sorted_sid)/4)
    topquart_sid = sorted_sid[:topquart]
    return topquart_sid


def get_spectra(segment_labels, cube, wn):
    # get avg spectra for each segment

    cube[cube == 0.0] = np.nan
    segment_spectra = defaultdict(list)
    num_labels = np.max(segment_labels) + 1  # python offset
    for y, col in enumerate(segment_labels):
        for x, row in enumerate(col):
            segment_spectra[segment_labels[y][x]].append(cube[y][x])
    spectra = defaultdict(list)
    skip = []
    multiplier = len(cube[0][0])
    for cluster in range(1, num_labels):
        size = len(segment_spectra[cluster])
        nans = np.count_nonzero(np.isnan(segment_spectra[cluster]))
        if nans/(size*multiplier) > 0.25:
            skip.append(cluster)
        spectra[cluster] = np.nanmean(segment_spectra[cluster], axis=0)
    sid = spec_distance(spectra, num_labels, skip)
    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn  # [0]
    multispectra.set_index('wn')
    return multispectra, sid


def get_axis(cube):
    maxes = np.max(cube, axis=2)
    lower = np.percentile(maxes, 10)
    upper = np.percentile(maxes, 90)
    return lower, upper


def create_plots(image_data, segment_data, spectra_data, scores):
    # create 3 plots, save out output

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))

    # median value for 'best image'
    image = np.median(image_data, axis=2)
    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    lower, upper = get_axis(image_data)
    sns.heatmap(image, vmin=lower, vmax=upper, ax=ax1)

    num_seg = len(np.unique(segment_data))
    cmap = sns.color_palette("hls", num_seg)
    ax2.tick_params(labelsize='small')
    sns.heatmap(segment_data, cmap=cmap, ax=ax2)

    ax3.tick_params(labelsize='medium')
    keys = list(spectra_data.columns)
    for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
        style = ':'  # make the distinct spectra stand out
        if key in scores:
            style = 'solid'
        ax3.plot(spectra_data['wn'], spectra_data[key], label=key, linewidth=5,
                 color=cmap[c], linestyle=style)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


if __name__ == "__main__":
    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    try:
        factor = float(sys.argv[1])
    except IndexError:
        factor = 0.5

    samples = [line.rstrip() for line in open('samples.txt', 'r')]
    # samples = [line.rstrip() for line in open('samples_test.txt', 'r')]

    for sample in samples:
        cube, wn = get_cube(sample)
        nonzero_cube = remove_zero(cube)
        img_data = preprocess(nonzero_cube, factor)
        segment_data = get_segments(img_data)
        spectra_data, sid = get_spectra(segment_data, nonzero_cube, wn)
        create_plots(img_data, segment_data, spectra_data, sid)
        plt.savefig('output/'+sample+"_c"+str(factor)+"_test.png")
