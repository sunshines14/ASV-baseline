#!/usr/bin/python
#-*- coding: utf-8 -*-

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import os
import numpy 
import argparse
import pdb
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description = "eval");
parser.add_argument('--ground_truth', type=str, default=None);
parser.add_argument('--prediction', type=str, default=None);
parser.add_argument('--positive', type=int, default=1, help='1 if higher is positive; 0 is lower is positive');
opt = parser.parse_args();

def read_score(filename):
    with open(filename) as f:
        scores = f.readlines()
        scores = [float(x.split()[0]) for x in scores] 
        return scores

def calculate_metrics(y, y_score, pos):
    # y: groundtruth scores,
    # y_score: prediction scores.
    
    # confusion matrix 
    #
    # in/out      true          false
    # true   true-positive  false-positive
    # false  false-negative true-negative
    
    # compute equal error rate (EER)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    # compute normalized minimum detection cost function (minDCF)
    c_miss = 10
    fnr = 1 - tpr
    p_target=0.01
    c_fa=1
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    m_dcf = c_det/c_def
    return eer, thresh, m_dcf

if __name__ == '__main__':
    y = read_score(opt.ground_truth)
    y_score = read_score(opt.prediction)
    eer, thresh, m_dcf = calculate_metrics(y,y_score,opt.positive)
    print('EER : %.4f%%'%(eer*100))
    print('thresh : %.4f'%(thresh))
    print('minDCF : %.4f'%(m_dcf))