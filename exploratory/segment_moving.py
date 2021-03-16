import mat73
import numpy as np
import seaborn as sns
import os
import re
import sys
import pandas as pd
import pysptools.distance as dis  # this is for evluation only
from matplotlib import pyplot as plt  # only for evaluation?
from skimage import segmentation
from collections import defaultdict
from timeit import default_timer as timer  # only for evaluation
from processing import get_axis, get_cube, get_truth_name, preprocess, remove_zero
import warnings


def most_dissimilar_spec(spectra, skip):
    # return the most dissimilar cluster label
    spec_sid = []
    num = len(spectra) + 1
    for cluster in range(1, num):
        sid = []
        if cluster in skip:
            sid.append(0)
            continue
        for cluster2 in range(1, num):
            if (cluster == cluster2):
                continue
            sam = dis.SAM(spectra[cluster], spectra[cluster2])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                sid.append(dis.SID(spectra[cluster],
                           spectra[cluster2]) * np.tan(sam))
        spec_sid.append(sid)
    combined_sid = [np.nansum(x) for x in spec_sid]
    for cluster in skip:
        combined_sid.insert(cluster - 1, 0)
    sorted_sid = sorted(range(1, len(combined_sid) + 1),
                        key=lambda k: combined_sid[k-1], reverse=True)
    most_dissimilar = sorted_sid[:1]
    return most_dissimilar


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
    #don't care about clusters with a lot of nans AKA edges
    for cluster in range(1, num_labels):
        size = len(segment_spectra[cluster])
        nans = np.count_nonzero(np.isnan(segment_spectra[cluster]))
        if nans/(size*multiplier) > 0.2:  # 0.25
            skip.append(cluster)
        # say spectra is the mean of the pixels spectras
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            spectra[cluster] = np.nanmean(segment_spectra[cluster], axis=0)
    dissim = most_dissimilar_spec(spectra, skip)
    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn  # [0]
    multispectra.set_index('wn')
    return multispectra, dissim


def create_plots(image_data, segment_data, spectra_data, scores):
    # create 3 plots
    fig, (ax1, ax2, ax3) = plt.subplots(3)  # , figsize=(60, 20))

    # median value for 'best image'
    image = np.median(image_data, axis=2)
    sns.set(font_scale=1)
    ax1.tick_params(labelsize=5)
    lower, upper = get_axis(image_data)
    sns.heatmap(image, vmin=lower, vmax=upper, ax=ax1)

    num_seg = len(np.unique(segment_data))
    # mask = np.zeros(segment_data.shape)
    # for label in scores:
        # future warning of indexing - MUST EDIT if upgrade python
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore', category=FutureWarning)
    #         mask[[segment_data == label]] = 1

    cmap = sns.color_palette("hls", num_seg )
    ax2.tick_params(labelsize=5)
    # ax2.set_facecolor("black")
    sns.heatmap(segment_data, cmap=cmap, ax=ax2)  # , mask=mask)

    ax3.tick_params(labelsize=5)
    keys = list(spectra_data.columns)
    for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
        style = ':'  # make the distinct spectra stand out
        # if key in scores:
        #     style = 'solid'
        ax3.plot(spectra_data['wn'], spectra_data[key], label=key, linewidth=2,
                 color=cmap[c], linestyle=style)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def look_up(name):

    table = {
        '14-01' : [1330, 2425],
        '14-02' : [1735, 2700], 
        '02-01' : [1550, 2000],
        '03-01' : [1700, 2100],
        '04-01' : [1700, 2400],
        '07-01' : [1700, 2550],
        '08-01' : [500, 1000],
        '09-01' : [1650, 2450],
        '01-01' : [1450, 2400],
        '5B-01' : [1750, 2400],
        '5W-01' : [1600, 2200],
        '6C-01' : [1500, 2400],
        '6T-01' : [1500, 2400],
        '17-01' : [1475, 2200],
        '15B-0' : [1550, 2350],
        '15P-0' : [1425, 2175],
        '16P-0' : [1500, 2300],
        '10-01' : [1850, 2550], 
        '11-01' : [1850, 2550],
        '12-01' : [1800, 2550],
        '13-01' : [1250, 2450],
    }
    try:
        start, end = table[name]
    except KeyError:
        print('Please select start/end points for sample ', name)
        sys.exit(1)

    return start, end


def get_mini_cubes(cube, name):
    print(name)
    start, end = look_up(name)
    subsize = 400
    list_of_minicubes = [cube[:, x:x+subsize, :] for x in range(start, end, subsize)]
    # length = cube.shape[1]
    # parts = 500
    # many_minis = int(length/parts)
    # list_of_minicubes = [cube[:, x*parts:(x+1)*parts, :] for x in range(many_minis)]
    # list_of_minicubes.append(cube[:, many_minis*parts:, : ])
    return list_of_minicubes


def get_segments(data, factor=1.0, nseg=40):
    time1 = timer()
    labels = segmentation.slic(data, n_segments=nseg, compactness=factor,
                               convert2lab=False, slic_zero=False,
                               start_label=1, max_iter=10, enforce_connectivity=True)
                               # start_label=1, max_iter=10, enforce_connectivity=False)
    time2 = timer()
    print('SLIC Alg: ', time2 - time1)
    return labels


def save_plt(outdir, sample, factor, part):
    try:
        plt.savefig(outdir+'/'+sample[:-4]+"_c"+str(factor)+"_part"+part+".png")  # remove '.mat'
    except OSError:
        os.mkdir(outdir)
        plt.savefig(outdir+'/'+sample[:-4]+"_c"+str(factor)+"_part"+part+".png")
    plt.close()


if __name__ == "__main__":
    os.chdir('/home/block-am/Documents/substrates/on_move_testing')

    samples = [line.rstrip() for line in open('full_samples.txt', 'r')]
    
    # factors = [0.60, 0.70, 1.3, 1.4]
    factors = [0.80, 0.90, 1.0, 1.1, 1.2]
    # factors = [0.75, 1.0]
    # factors = [.50, 0.75, 1.0, 1.25]
    # factors = [.60, .70, .75, .80, .90, 1.0, 1.1, 1.2]
    nsegs = [5]
    for nseg in nsegs:
        for factor in factors:
            print('trying ', nseg, ' segments with compactness ', factor)
            for sample in samples:
                cube, wn = get_cube(sample)
                cube[np.isnan(cube)] = 0
                minis = get_mini_cubes(cube, sample[7:12])
                for m, mini in enumerate(minis):
                    img_data = preprocess(mini)
                    segment_data = get_segments(img_data, factor, nseg)
                    spectra_data, sid = get_spectra(segment_data, mini, wn)
                    create_plots(img_data, segment_data, spectra_data, sid)
                    outdir = 'output/hand_isolated_400_5clusters/'+sample[:-4]
                    save_plt(outdir, sample, factor, str(m))
