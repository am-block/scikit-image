import scipy.io as sio
import numpy as np
import seaborn as sns
import os
import re
import sys
import pandas as pd
import pysptools.distance as dis  # this is for evluation only
from matplotlib import pyplot as plt  # only for evaluation?
from skimage.segmentation import slic
from collections import defaultdict
from timeit import default_timer as timer  # only for evaluation
from processing import get_axis, get_cube, get_truth_name, impute_blanks, preprocess, remove_zero
import warnings


def get_truth_spectra(sample):
    data = sio.loadmat('truth')
    spect = ''
    for i in range(8):
        if re.search(str(sample), data['truth_map']['name'][0][i][0]):
            spect = data['truth_map']['spect'][0][i][0]
            break
    # if spect == '':
    #     sys.exit("Couldn't find truth spectrum of ", sample)
    return spect[20:280]


def truth_sid(spectra, name, skip):
    spec_sid = []
    num = len(spectra) + 1
    truth_spect = get_truth_spectra(name)
    spec_sid = []

    for cluster in range(1, num):
        sid = []
        if cluster in skip:
            sid.append(1.5)
            continue
        sam = dis.SAM(spectra[cluster], truth_spect)
        sid.append(dis.SID(spectra[cluster], truth_spect) * np.tan(sam))
        spec_sid.append(sid)

    sorted_spectra = sorted(range(1, len(spec_sid) + 1),
                            key=lambda k: spec_sid[k-1])
    most_similar = sorted_spectra[:1]
    # sid_score = min(spec_sid)
    # sid_index = spec_sid.index(min(spec_sid)) + 1
    # print("closest sid_score: ", sid_score, " for ", sid_index)

    return truth_spect, most_similar


def spec_distance(spectra, skip):
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
            sid.append(dis.SID(spectra[cluster],
                       spectra[cluster2]) * np.tan(sam))
        spec_sid.append(sid)
    combined_sid = [np.nansum(x) for x in spec_sid]
    for cluster in skip:
        combined_sid.insert(cluster-1, 0)
    sorted_sid = sorted(range(1, len(combined_sid) + 1),
                        key=lambda k: combined_sid[k-1], reverse=True)
    # topquart = int(len(sorted_sid)/4)
    topquart_sid = sorted_sid[:1]  # topquart]
    return topquart_sid


def get_spectra(segment_labels, cube, wn, name, truth):
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
        if nans/(size*multiplier) > 0.25:
            skip.append(cluster)
        # say spectra is the mean of the pixels spectras
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            spectra[cluster] = np.nanmean(segment_spectra[cluster], axis=0)
    tsid = 0.0
    sid = spec_distance(spectra, skip)
    # save out truth spectra and the closest by sid
    if truth:
        true_spec, tsid = truth_sid(spectra, name, skip)
        truth_dict = {'truth': true_spec}
        spectra.update(truth_dict)
    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn  # [0]
    multispectra.set_index('wn')
    return multispectra, sid


def create_plots(image_data, segment_data, spectra_data, scores, truth):
    # create 3 plots, save out output

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))

    # median value for 'best image'
    image = np.median(image_data, axis=2)
    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    lower, upper = get_axis(image_data)
    sns.heatmap(image, vmin=lower, vmax=upper, ax=ax1)

    num_seg = len(np.unique(segment_data))
    mask = np.zeros(segment_data.shape)
    for label in scores:
        # future warning of indexing - MUST EDIT if upgrade python
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            mask[[segment_data == label]] = 1

    cmap = sns.color_palette("hls", num_seg)
    ax2.tick_params(labelsize='small')
    ax2.set_facecolor("black")
    sns.heatmap(segment_data, cmap=cmap, ax=ax2, mask=mask)

    ax3.tick_params(labelsize='medium')
    keys = list(spectra_data.columns)
    if truth:
        ax3.plot(spectra_data['wn'], spectra_data['truth'], label='truth',
                 linewidth=3, color='black', linestyle='solid')
        keys.remove('truth')
    for c, key in enumerate(keys[:-1]):  # remove 'wn' at end
        style = ':'  # make the distinct spectra stand out
        if key in scores:
            style = 'solid'
        ax3.plot(spectra_data['wn'], spectra_data[key], label=key, linewidth=5,
                 color=cmap[c], linestyle=style)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def get_segments(data, factor, nseg, miter):
    # time1 = timer()
    labels = slic(data, n_segments=nseg, compactness=factor,
                               convert2lab=False, slic_zero=False,
                               start_label=1, max_iter=miter)
    # time2 = timer()
    # print('SLIC Algs: ', time2 - time1, time3 - time2, time4 - time3)
    return labels


def save_plt(outdir, sample, factor):
    try:
        plt.savefig(outdir+'/'+sample+"c"+str(factor)+".png")
    except OSError:
        os.mkdir(outdir)
        plt.savefig(outdir+'/'+sample+"c"+str(factor)+".png")
    plt.close()



if __name__ == "__main__":
    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    try:
        factor = float(sys.argv[1])
    except IndexError:
        factor = 0.5

    samples = [line.rstrip() for line in open('samples.txt', 'r')]
    truth_names = get_truth_name()
    names = [name[8:] for name in truth_names]
    factors = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    # factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    #            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # factors = [1.6, 1.7, 1.8, 1.9]
    nsegs = [10, 20, 40]
    miters = [10, 30, 50]
    for nseg in nsegs:
        for miter in miters:
            for factor in factors:
                for sample in samples:
                    if (sample == 'paper_grid'):
                        current = 0.0
                    else:
                        setD = re.search('D', sample)
                        if setD:
                            current = re.search('\d+(_\d+)?', sample).group()
                            cube, wn = get_cube(sample)
                            nonzero_cube = remove_zero(cube)
                            img_data = preprocess(nonzero_cube)
                            segment_data = get_segments(img_data, factor, nseg,
                                                        miter)
                            if current in names:
                                spectra_data, sid = get_spectra(segment_data,
                                                                nonzero_cube, wn,
                                                                current, True)
                                create_plots(img_data, segment_data, spectra_data,
                                             sid, True)
                                outdir = 'output/normalize98p5_'+str(nseg)+'x'+str(miter)
                                save_plt(outdir, sample, factor)
                                # try:
                                #     plt.savefig(outdir+'/'+sample+"c"+str(factor)+".png")
                                # except OSError:
                                #     os.mkdir(outdir)
                                #     plt.savefig(outdir+'/'+sample+"c"+str(factor)+".png")
                                # plt.close()
        # else:
#     spectra_data, sid = get_spectra(segment_data, nonzero_cube,
#                                     wn, sample, False)
#     create_plots(img_data, segment_data, spectra_data, sid, False)
#     plt.savefig('output/num_cluster_scan/'+sample+"c"+str(factor)+"x15.png")
#     plt.close()
#     continue
# spectra_data, sid = get_spectra(segment_data, nonzero_cube, wn)