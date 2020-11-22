#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 22:27:11 2020

@author: u1876024
"""

from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import staintools

from numpy import logical_and as land, logical_not as lnot, logical_or as lor
from skimage.color import rgb2hsv
from skimage.morphology import (
    disk,
    binary_opening,
    binary_closing,
    remove_small_holes,
    remove_small_objects,
)

from tqdm import tqdm
def patchify(I,ps=(32,32),Mask=None,mthresh = 0.25):
    Z = []
    C = []
    for i in range(I.shape[0]//ps[0]):
        for j in range(I.shape[1]//ps[1]):
            if Mask is not None:
                if np.mean(Mask[i*ps[0]:(i+1)*ps[0],j*ps[0]:(j+1)*ps[0]])<mthresh:
                    continue
                    
                Z.append(I[i*ps[0]:(i+1)*ps[0],j*ps[0]:(j+1)*ps[0]])
                C.append([i,j])
    return np.array(Z),np.array(C)


from sklearn.preprocessing import normalize
#%%
def read_data():
    data_dir = './bisque-20200824.120952/Breast Cancer Cells'
    flist = glob(data_dir+'/*.tif')
    lbl_dir = {'malignant':1.0,'benign':0.0}
    target = staintools.read_image("target.png")
    normalizer = staintools.StainNormalizer(method="vahadane")  # "macenko" is much faster
    normalizer.fit(target)
    B = []
    YY = []
    CC = []
    for f in tqdm(flist):
        lbl = lbl_dir[os.path.split(f)[-1].split('_')[-2][:-1]]
        I = imread(f)
        nI = normalizer.transform(I)
        M = rgb2hsv(nI)[:,:,-1]<0.9
        P,C = patchify(nI,Mask = M)
        Pf = np.array([p.flatten() for p in P])/(255)
        B.append(Pf)
        CC.append(C)
        YY.append(lbl)
        #plt.figure();plt.imshow(I)
        #plt.figure();plt.imshow(nI)
    return B,np.array(YY),CC
#%%
    
#%%
# fig, axs = plt.subplots(24, 28)    
# plt.subplots_adjust(wspace=0, hspace=0)
# for i,p in enumerate(P):
#     ax = axs[tuple(C[i])]
#     plt.sca(ax)
#     plt.imshow(p)

if __name__=='__main__':
    B,YY,CC = read_data()