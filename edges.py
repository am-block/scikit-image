import numpy as np
import seaborn as sns
import os
import re
from matplotlib import pyplot as plt  # only for evaluation?
from processing import get_cube, get_truth_name, preprocess, remove_zero
from timeit import default_timer as timer  # only for evaluation
import pandas as pd
import warnings
from skimage.segmentation import felzenszwalb, slic
from sklearn.cluster import KMeans
from rxanomaly import rx_detector, remove_blanks, fill_zeros, impute_blanks, save_plt, split
# from plot_macro import get_segments
from segment_rx import get_spectra, get_segments
from scipy import ndimage as ndi
from skimage import feature, filters


def create_plots(image_data, segments, edges, similar):
    # create 3 plots, save out output
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 10), sharex=True)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20), sharex=True, sharey=True)

    # median value for 'best image'
    image = np.median(image_data, axis=2)
    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    # lower, upper = get_axis(image_data)
    sns.heatmap(image, ax=ax1)

    num_seg = len(np.unique(segments))
    mask = np.zeros(segments.shape)
    for label in similar:
        # future warning of indexing - MUST EDIT if upgrade python
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            mask[[segments == label]] = 1

    cmap = sns.color_palette("hls", num_seg)
    ax2.tick_params(labelsize='small')
    ax2.set_facecolor('black')
    sns.heatmap(segments, cmap=cmap, ax=ax2, mask=mask)

    ax3.imshow(edges, cmap=plt.cm.gray)
    # ax3.tick_params(labelsize='medium')
    # keys = list(spectra.columns)
    # ax3.plot(spectra['wn'], spectra['truth'], label='truth',
    #             linewidth=3, color='black', linestyle='solid')
    # keys.remove('truth')
    # for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
    #     style = ':'  # make the distinct spectra stand out
    #     if key in similar:
    #         style = 'solid'
    #     ax3.plot(spectra['wn'], spectra[key], label=key, linewidth=5,
    #              color=cmap[c], linestyle=style)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def get_edges(img, sig = 1.):
    time1 = timer() 
    edges = feature.canny(img, sigma=sig)
    # edges = filters.laplace(img, ksize=5)
    time2 = timer()
    print('edges: ', time2 - time1)
    return edges


def pooled_img(img, factor=2):
    pooled = []
    for y in range(0, img.shape[0], factor):
        pool_y = []
        for x in range(0, img.shape[1], factor):
            pool_y.append(np.nanmean(img[y:y+factor][:, x:x+factor], axis=(0, 1)))
        pooled.append(pool_y)
    pool_array = np.asarray(pooled)
    pool_array[np.isnan(pool_array)] = 0.0
    return pool_array


def scale(data):
    robust_scaler = RobustScaler(quantile_range=(0, 100))
    scaled_data = robust_scaler.fit_transform(data)
    return scaled_data


if __name__ == "__main__":
    os.chdir('/home/block-am/Documents/SLIC Test Data/')

    samples = [line.rstrip() for line in open('samples.txt', 'r')]
    truth_names = get_truth_name()
    names = [name[8:] for name in truth_names]
    for sample in samples:
        if (sample == 'paper_grid'):
            continue
        else:
            setD = re.search('D', sample)
            if setD:
                current = re.search('\d+(_\d+)?', sample).group()
                if current in names:
                    cube, wn = get_cube(sample)
                    nonzero_cube = remove_blanks(cube)
                    img_data = preprocess(nonzero_cube)
                    split_data = split(img_data)
                    for i, sp in enumerate(split_data):

                        # pooled_data = pooled_img(img_data)
                        pooled_data = pooled_img(sp, 3)
                        rx_anomalies = rx_detector(pooled_data)
                        # rx_anomalies = rx_detector(img_data)
                        # rx_anomalies = rx_detector(sp)
                        edges = get_edges(rx_anomalies, 1)
                        segments = get_segments(rx_anomalies)
                        spectra, similar = get_spectra(segments, cube, wn, current)
                        create_plots(nonzero_cube, segments, edges, similar)
                    # edges = get_edges(rx_anomalies)
                    # segments = get_segments(rx_anomalies)
                    # spectra, similar = get_spectra(segments, cube, wn, current)
                    # create_plots(nonzero_cube, segments, edges, similar)
                        outdir = 'output/edge_detection/canny_split32_3x3'
                        save_plt(outdir, sample, i)
