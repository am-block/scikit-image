import numpy as np
import seaborn as sns
import os
import re
from matplotlib import pyplot as plt  # only for evaluation?
from timeit import default_timer as timer  # only for evaluation
from processing import get_cube, get_truth_name, preprocess, impute_blanks
import spectral as spy
from scipy.stats import chi2
from sklearn.preprocessing import RobustScaler


def rx_detector(img):
#     time1 = timer()
    rxvals = spy.rx(img)
#     time2 = timer()
#     print("rx: ", time2 - time1)  # 0.07 +/- 0.01
    return rxvals


def pooled_img(img):
    pooled = []
    for y in range(0, img.shape[0], 3):
        pool_y = []
        for x in range(0, img.shape[1], 3):
            pool_y.append(np.nanmean(img[y:y+3][:, x:x+3], axis=(0, 1)))
        pooled.append(pool_y)
    pool_array = np.asarray(pooled)
    pool_array[np.isnan(pool_array)] = 0.0
    return pool_array


def create_plots(image_data, anomalies):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(90, 60))

    # median value for 'best image'
    image = np.median(image_data, axis=2)
    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    nbands = image_data.shape[-1]
    # P = chi2.ppf(0.99, nbands)
    P = chi2.ppf(0.999, nbands)
    # lower, upper = get_axis(image_data)
    sns.heatmap(image, ax=ax1)
    ax2.tick_params(labelsize='small')
    sns.heatmap((anomalies > P), ax=ax2)
    # sns.heatmap(anomalies, ax=ax2)
    sns.histplot(data=anomalies, ax=ax3)


def fill_zeros(cube):
    norm = np.nanmean(cube, axis=(0, 1))
    missing = np.argwhere(np.isnan(cube))
    for pixel in missing:
        cube[pixel[0]][pixel[1]][pixel[2]] = norm[pixel[2]]
    return cube


def remove_blanks(cube):
    # fill in blank pixels
    filled_in_data = impute_blanks(cube)
    nonzero_data = filled_in_data[19:115, 11:125, 20:280]  # y: 10:120, 10:
    no_zeroes = fill_zeros(nonzero_data)
    return no_zeroes


def split(data):
    # y_length = int(data.shape[0]/5)
    # dataA = data[:y_length]
    # dataB = data[y_length:y_length*2]
    # dataC = data[y_length*2:y_length*3]
    # dataD = data[y_length*3:y_length*4]
    # dataE = data[y_length*4:]
    # return dataA, dataB, dataC, dataD, dataE
    dataA = data[:32]
    dataB = data[32:64]
    dataC = data[64:]
    return dataA, dataB, dataC
    # return data


def save_plt(outdir, sample, i=-1):
    if i >= 0:
        sample = sample + '-' + str(i)
    try:
        plt.savefig(outdir+'/'+sample+".png")
    except OSError:
        os.mkdir(outdir)
        plt.savefig(outdir+'/'+sample+".png")
    plt.close()


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
                if current in names:
                    cube, wn = get_cube(sample)
                    nonzero_cube = remove_blanks(cube)
                    img_data = preprocess(nonzero_cube)
                    rx_anomalies = rx_detector(img_data)
                    create_plots(nonzero_cube, rx_anomalies)
                    outdir = 'output/rx-hist'
                    save_plt(outdir, sample)
