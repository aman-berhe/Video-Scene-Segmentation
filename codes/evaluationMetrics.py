#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:00:20 2018

@author: berhe
"""
from sklearn.metrics import recall_score,precision_score,f1_score
#from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,accuracy_score
import csv
import pandas as pd
import numpy as np
import os

"""
recall, prcision and f1 measure are used  from the standard recall and precision functions of skilearn metrics
due some encoding problems if the length of the hypothesis and references are not equal then we set the
length to the minimu length of the two
"""
def recall(ref, hyp,rType='binary'):
    if len(ref)>len(hyp):
        ref=ref[:len(hyp)]
    else:
        hyp=hyp[:len(ref)]

    return recall_score(ref,hyp,average=rType)

"""
Recall from scratch
"""
def recall2(X_test, y_test):
	tp =0
	fp = 0
	for i in range(len(y_test)):
	    if y_test[i] == 1:
	        if X_test[i] == 1:
	            tp += 1
	        else:
	            fp +=1

	tn =0
	fn = 0
	for i in range(len(y_test)):
	    if y_test[i] == 0:
	        if X_test[i] == 0:
	            tn += 1
	        else:
	            fn +=1

	return (tp/(tp+fn))


def precision(ref, hyp,rType='binary'):
    if len(ref)>len(hyp):
        ref=ref[:len(hyp)]
    else:
        hyp=hyp[:len(ref)]

    return precision_score(ref,hyp,average=rType)

"""
Precision from scratch
"""
def precision2( ref, hyp):
    success =0
    fail = 0
    length=0
    if len(ref)<len(hyp):
    	length=len(ref)
    else:
    	length=len(hyp)

    for i in range(length):
        if hyp[i] == '1':
            if ref[i]== '1':
                success += 1
            else:
                fail +=1

    return (success/(success+fail))
"""
F-measuere : F1
"""
def f1(ref,hyp):
    if len(ref)>len(hyp):
        ref=ref[0:len(hyp)]
    else:
        hyp=hyp[:len(ref)]

    return f1_score(ref,hyp)

"""
pk value it takes the refrence and hypothesis: The lower value the best result
"""
def pk(ref, hyp, k=None, boundary=1):

    if k is None:
        #print(ref.count(boundary))
        k = int(round(len(ref) / (ref.count(boundary) * 2.)))

    err = 0
    for i in range(len(ref)-k +1):
        r =ref[i:i+k].count(boundary) >  0
        h = hyp[i:i+k].count(boundary) > 0
        #print(ref.count(boundary),hyp.count(boundary),boundary)
        #print (h)
        #print (r)
        if r != h:
           err += 1
    return err / (len(ref)-k +1.)

"""Evaluation technique windowsDiff: it takes the segments' values,  it takes the refrence and hypothesis: The lower value the best result
"""
def windowdiff(ref, hyp, k=None, boundary=1):

    if k is None:
        #print(ref.count(boundary))
        k = int(round(len(ref) / (ref.count(boundary) * 2.)))

    #if len(seg1) != len(seg2):
       # raise ValueError("Segmentations have unequal length")
    wd = 0
    for i in range(len(ref) - k):
        wd += abs((ref[i:i+k+1].count(boundary)) - (hyp[i:i+k+1].count(boundary)))
    return (wd/(float(len(ref)-k)))

"""
Evaluations For Clustering
"""

def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    if len(y_true)>len(y_pred):
        y_true=y_true[:len(y_pred)]
    else:
        y_pred=y_pred[:len(y_true)]

    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # set the number of bins to be n_classes+2 so that
    # count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_labeled_voted[y_pred==cluster] = winner

    return accuracy_score(y_true, y_labeled_voted)

def coverage(y_true,y_pred):

     if len(y_true)>len(y_pred):
        y_true=y_true[:len(y_pred)]
     else:
        y_pred=y_pred[:len(y_true)]

     y_labeled_voted = np.zeros(y_pred.shape)
     labels = np.unique(y_pred)

     bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)


     for cluster in np.unique(y_true):
         hist,_=np.histogram(y_pred[y_true==cluster],bins=bins)

         winner=np.argmax(hist)
         y_labeled_voted[y_true==cluster]=winner

     return accuracy_score(y_pred, y_labeled_voted)

def scene_overflow(scenes, gt, i):
	metr = lambda x: minus(x, gt[i]).__len__() * min(1, intersection(x, gt[i]).__len__())
	val = sum(map(metr, scenes))
	if i == 0:
		base = gt[i+1].__len__()
	elif i == len(gt) - 1:
		base = gt[i-1].__len__()
	else:
		base = gt[i-1].__len__() + gt[i+1].__len__()
	return val / (0.0 + base)
	
def overflow(scenes, gt):
	result = 0.0
	tc = 0.0
	for i in range(0, len(gt)):
		result += scene_overflow(scenes, gt, i) * gt[i].__len__()
		tc += gt[i].__len__()
	return result / tc

"""
normalized mutual information score:
"""
def nmi(y_true,y_pred):
    if len(y_true)>len(y_pred):
        y_true=y_true[:len(y_pred)]
    else:
        y_pred=y_pred[:len(y_true)]

    return normalized_mutual_info_score(y_true,y_pred)
