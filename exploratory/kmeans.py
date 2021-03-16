import numpy as np
import seaborn as sns
import os
import re
from collections import defaultdict
from matplotlib import pyplot as plt  # only for evaluation?
from processing import get_cube, get_truth_name, preprocess, remove_zero
import spectral as spy  # can try spectral or pysptools
from timeit import default_timer as timer  # only for evaluation
import pandas as pd
from pysptools import skl  # ended support in 2018
import warnings
from sklearn.cluster import KMeans
from rxanomaly import rx_detector, remove_blanks, fill_zeros, impute_blanks, pooled_img


def save_plt(outdir, sample, i=-1):
    if i >= 0:
        sample = sample + '-' + str(i)
    try:
        plt.savefig(outdir+'/'+sample+".png")
    except OSError:
        os.mkdir(outdir)
        plt.savefig(outdir+'/'+sample+".png")


def create_plots(image_data, segments, spectra):
    # create 3 plots, save out output
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(90, 30))

    # median value for 'best image'
    image = np.median(image_data, axis=2)
    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    # lower, upper = get_axis(image_data)
    sns.heatmap(image, ax=ax1)

    num_seg = len(np.unique(segments))
    cmap = sns.color_palette("hls", num_seg)
    ax2.tick_params(labelsize='small')
    sns.heatmap(segments, cmap=cmap, ax=ax2)

    ax3.tick_params(labelsize='medium')
    keys = list(spectra.columns)
    print(num_seg, len(keys), keys)
    for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
        style = 'solid'
        try:
            ax3.plot(spectra['wn'], spectra[key], label=key, linewidth=5,
                     color=cmap[c], linestyle=style)
        except:
            pass
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def get_spectra(segment_labels, cube, wn):
    # get avg spectra for each segment
    cube[cube == 0.0] = np.nan
    segment_spectra = defaultdict(list)
    num_labels = np.max(segment_labels) + 1  # python offset
    for y, col in enumerate(segment_labels):
        for x, row in enumerate(col):
            segment_spectra[segment_labels[y][x]].append(cube[y][x][20:280])
    spectra = defaultdict(list)
    for cluster in range(num_labels):
        spectra[cluster] = np.nanmean(segment_spectra[cluster], axis=0)
    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn  # [0]
    multispectra.set_index('wn')
    return multispectra


def kmeans_spy(img):
    time1 = timer()
    (m, n) = spy.kmeans(img, 3, 10)  # spectral /spy (data, clusters, iters)
    time2 = timer()
    print('kmeans time: ', time2 - time1)
    return m


def kmeans_pysp(img):
    time1 = timer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        km = skl.KMeans()
        m = km.predict(img, n_clusters=3, n_jobs=-1)  # n_jobs = #CPUs
        time2 = timer()
        print('kmeans time: ', time2 - time1)
        return m


def kmeans_skl(cube):
    nonzero_cube = remove_blanks(cube)
    img_data = preprocess(nonzero_cube)
    pooled = pooled_img(img_data)
    rxvals = rx_detector(pooled)
    rx_reshaped = rxvals.reshape(rxvals.shape[0]*rxvals.shape[1]).reshape(-1, 1)
    #    smushed_data = data.reshape(data.shape[0]*data.shape[1],
    #                             data.shape[2])
    time1 = timer()
    km = KMeans(n_clusters=5, max_iter=10, n_jobs=-1)
    m = km.fit_predict(rx_reshaped)
    m = m.reshape(rxvals.shape[0], rxvals.shape[1])
    time2 = timer()
    print('kmeans time: ', time2 - time1)
    return m


if __name__ == "__main__":
    os.chdir('/home/block-am/Documents/SLIC Test Data/')

    samples = [line.rstrip() for line in open('samples.txt', 'r')]
    truth_names = get_truth_name()
    names = [name[8:] for name in truth_names]
    for sample in samples:
        if (sample == 'paper_grid'):
            current = 0.0
        else:
            setD = re.search('D', sample)
            if setD:
                current = re.search('\d+(_\d+)?', sample).group()
                cube, wn = get_cube(sample)
                nonzero_cube = remove_zero(cube)
                # img_data = preprocess(nonzero_cube)
                # segments = kmeans_spy(img_data)
                # segments = kmeans_pysp(img_data)
                segments = kmeans_skl(cube)
                spectra = get_spectra(segments, cube, wn)
                if current in names:
                    create_plots(nonzero_cube, segments, spectra)
                    outdir = 'output/kmeans/sklearn-rx/c5-10x-pooled'
                    save_plt(outdir, sample)
