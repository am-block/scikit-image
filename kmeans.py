import scipy.io as sio
import numpy as np
import seaborn as sns
import os, re, sys, pickle
from matplotlib import pyplot as plt  # only for evaluation?
from timeit import default_timer as timer  # only for evaluation
from processing import get_cube, get_truth_name, preprocess, impute_blanks
import spectral as spy


def kmeans(img):
    # (m, n) = spy.kmeans(img, 20, 30)
    # print(m, n)
    ...