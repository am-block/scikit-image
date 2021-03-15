import scipy.io as sio
import numpy as np
import seaborn as sns
import os
import re
import sys
import pandas as pd
import pysptools.distance as dis  # this is for evluation only
from matplotlib import pyplot as plt  # only for evaluation?
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from collections import defaultdict
from timeit import default_timer as timer  # only for evaluation
from processing import get_axis, get_cube, get_truth_name, impute_blanks, preprocess, remove_zero
from rxanomaly import rx_detector, remove_blanks, save_plt
import warnings
import spectral as spy
from sklearn.preprocessing import RobustScaler
from plot_macro import truth_sid, get_truth_spectra, spec_distance


def save_plt(outdir, sample, nseg=0):
    try:
        plt.savefig(outdir+'/'+sample+"c"+str(factor)+".png")
    except OSError:
        os.mkdir(outdir)
        plt.savefig(outdir+'/'+sample+"c"+str(factor)+".png")
    plt.close()


def get_spectra(segments, cube, wn, name):
    # get avg spectra for each segment
    cube[cube == 0.0] = np.nan
    segment_spectra = defaultdict(list)
    num_labels = np.max(segments) + 1  # python offset
    for y, col in enumerate(segments):
        for x, row in enumerate(col):
            segment_spectra[segments[y][x]].append(cube[y][x][20:280])
    spectra = defaultdict(list)
    skip = []
    multiplier = len(cube[0][0])
    #don't care about clusters with a lot of nans AKA edges
    for cluster in range(1, num_labels):
        size = len(segment_spectra[cluster])
        nans = np.count_nonzero(np.isnan(segment_spectra[cluster]))
        if nans/(size*multiplier) > 0.25:
            skip.append(cluster)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            spectra[cluster] = np.nanmean(segment_spectra[cluster], axis=0)
    true_spec, tsid = truth_sid(spectra, name, skip)
    most_similar = spec_distance(spectra, skip)
    truth_dict = {'truth': true_spec}
    spectra.update(truth_dict)
    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn  # [0]
    multispectra.set_index('wn')
    return multispectra, most_similar


def create_plots(image_data, segments, spectra, similar):
    # create 3 plots, save out output
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))

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

    ax3.tick_params(labelsize='medium')
    keys = list(spectra.columns)
    ax3.plot(spectra['wn'], spectra['truth'], label='truth',
                linewidth=3, color='black', linestyle='solid')
    keys.remove('truth')
    for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
        style = ':'  # make the distinct spectra stand out
        if key in similar:
            style = 'solid'
        ax3.plot(spectra['wn'], spectra[key], label=key, linewidth=5,
                 color=cmap[c], linestyle=style)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def scale(data):
    robust_scaler = RobustScaler(quantile_range=(50, 100))
    scaled_data = robust_scaler.fit_transform(data)
    return scaled_data


def get_segments(rxvals, nseg=40, c=3.0, miter=100):
    time1 = timer()
    # scale to [0, 1]
    scaled_vals = scale(rxvals)
    segments = slic(scaled_vals, n_segments=nseg, compactness=c, convert2lab=False,
                    slic_zero=False, start_label=1, max_iter=miter, enforce_connectivity=False)
    # segments = quickshift(scaled_vals, ratio=.75, max_dist=7, 
    #                       return_tree=False, sigma=0.5, convert2lab=False, random_seed=15)
    # segments = slic(scaled_vals, n_segments=20, compactness=0.75, convert2lab=False,
    # segments = slic(scaled_vals, n_segments=10, compactness=0.3, convert2lab=False,
    #                 slic_zero=False, start_label=1, max_iter=100, enforce_connectivity=False)
    # segments = felzenszwalb(scaled_vals, scale=200, sigma=0.5, min_size=100)
    # segments = felzenszwalb(scaled_vals, scale=100, sigma=0.5, min_size=50)
    # segments = watershed(scaled_vals, markers=30, connectivity=1, offset=None, mask=None,
    #                      compactness=.01, watershed_line=True)
    time2 = timer()
    print('segment time: ', time2-time1)
    return segments


if __name__ == "__main__":
    os.chdir('/home/block-am/Documents/SLIC Test Data/')

    samples = [line.rstrip() for line in open('samples.txt', 'r')]
    truth_names = get_truth_name()
    names = [name[8:] for name in truth_names]
    nsegs = [10, 20, 30, 35, 40]
    factors = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    miters = [50, 100]

    for sample in samples:
        if (sample == 'paper_grid'):
            continue
        else:
            setD = re.search('D', sample)
            if setD:
                current = re.search('\d+(_\d+)?', sample).group()
                if current in names:

                    for nseg in nsegs:
                        for factor in factors:
                            for miter in miters:

                                cube, wn = get_cube(sample)
                                nonzero_cube = remove_blanks(cube)
                                img_data = preprocess(nonzero_cube)
                                rx_anomalies = rx_detector(img_data)
                                segments = get_segments(rx_anomalies, nseg, factor, miter)
                                spectra, similar = get_spectra(segments, cube, wn, current)
                                create_plots(nonzero_cube, segments, spectra, similar)
                                outdir = 'output/segment_rx/slic-nocon/'+sample+"_"+str(nseg)+'x'+str(miter)
                                save_plt(outdir, sample, factor)