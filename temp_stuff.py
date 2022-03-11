import torch.nn as nn
from torchaudio import datasets
from mk_graph import mk_graph
from pathlib import Path
import torch
import numpy as np
from gmil_edgeconv_cpath_old import GIN, getVisData, showGraphDataset, learnable_sig, change_pixel
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import pandas as pd
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'


def plot_hist(g):
    scores=g.v.cpu().numpy()
    mpl.pyplot.figure()
    df=pd.DataFrame(scores,columns=['s1','s2'])
    #ax=df['s1'].plot.hist(bins=20, density=True, edgecolor='w', linewidth=0.5)
    ax=df['s1'].plot.density(color='g', alpha=0.5)
    ax.set_xlim(left=-0.1,right=1.1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    df['s2'].plot.density(color='r', alpha=0.5, ax=ax)
    mpl.pyplot.show()

to_use=['Nucleus: Area µm^2', 'Nucleus: Circularity', 'Nucleus: Solidity', 'Nucleus: Max diameter µm', 'Nucleus: Min diameter µm', 'Cell: Area µm^2', 'Cell: Circularity', 'Cell: Solidity', 'Cell: Max diameter µm',
 'Cell: Min diameter µm', 'Nucleus/Cell area ratio', 'Hematoxylin: Nucleus: Mean', 'Hematoxylin: Nucleus: Median', 'Hematoxylin: Nucleus: Std.Dev.', 'Hematoxylin: Cytoplasm: Mean', 'Hematoxylin: Cytoplasm: Median', 'Hematoxylin: Cytoplasm: Std.Dev.', 'Hematoxylin: Cell: Mean', 'Hematoxylin: Cell: Median', 'Hematoxylin: Cell: Std.Dev.', 'Eosin: Nucleus: Mean', 'Eosin: Nucleus: Median', 'Eosin: Nucleus: Std.Dev.', 'Eosin: Cytoplasm: Mean', 'Eosin: Cytoplasm: Median', 'Eosin: Cytoplasm: Std.Dev.', 'Eosin: Cell: Mean', 'Eosin: Cell: Median', 'Eosin: Cell: Std.Dev.', 'Smoothed: 50 µm: Nucleus: Area µm^2', 'Smoothed: 50 µm: Nucleus: Circularity', 'Smoothed: 50 µm: Nucleus: Solidity', 'Smoothed: 50 µm: Nucleus: Max diameter µm', 'Smoothed: 50 µm: Nucleus: Min diameter µm', 'Smoothed: 50 µm: Cell: Area µm^2', 'Smoothed: 50 µm: Cell: Circularity', 'Smoothed: 50 µm: Cell: Solidity', 'Smoothed: 50 µm: Cell: Max diameter µm', 'Smoothed: 50 µm: Cell: Min diameter µm', 'Smoothed: 50 µm: Nucleus/Cell area ratio',
 'Smoothed: 50 µm: Hematoxylin: Nucleus: Mean', 'Smoothed: 50 µm: Hematoxylin: Nucleus: Median', 'Smoothed: 50 µm: Hematoxylin: Nucleus: Std.Dev.', 'Smoothed: 50 µm: Hematoxylin: Cytoplasm: Mean', 'Smoothed: 50 µm: Hematoxylin: Cytoplasm: Median', 'Smoothed: 50 µm: Hematoxylin: Cytoplasm: Std.Dev.', 'Smoothed: 50 µm: Hematoxylin: Cell: Mean', 'Smoothed: 50 µm: Hematoxylin: Cell: Median', 'Smoothed: 50 µm: Hematoxylin: Cell: Std.Dev.', 'Smoothed: 50 µm: Eosin: Nucleus: Mean', 'Smoothed: 50 µm: Eosin: Nucleus: Median', 'Smoothed: 50 µm: Eosin: Nucleus: Std.Dev.', 'Smoothed: 50 µm: Eosin: Cytoplasm: Mean', 'Smoothed: 50 µm: Eosin: Cytoplasm: Median', 'Smoothed: 50 µm: Eosin: Cytoplasm: Std.Dev.', 'Smoothed: 50 µm: Eosin: Cell: Mean', 'Smoothed: 50 µm: Eosin: Cell: Median', 'Smoothed: 50 µm: Eosin: Cell: Std.Dev.', 'Smoothed: 50 µm: Nearby detection counts', 'ROI: 0.44 µm per pixel: OD Sum: Mean', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Mean', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Std.dev.',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Angular second moment (F0)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Contrast (F1)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Correlation (F2)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum of squares (F3)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Inverse difference moment (F4)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum average (F5)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum variance (F6)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum entropy (F7)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Entropy (F8)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Difference variance (F9)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Difference entropy (F10)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Information measure of correlation 1 (F11)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Information measure of correlation 2 (F12)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Mean',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Std.dev.', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Angular second moment (F0)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Contrast (F1)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Correlation (F2)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum of squares (F3)',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Inverse difference moment (F4)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum average (F5)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum variance (F6)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum entropy (F7)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Entropy (F8)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Difference variance (F9)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Difference entropy (F10)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Information measure of correlation 1 (F11)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Information measure of correlation 2 (F12)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Mean', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Std.dev.', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Angular second moment (F0)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Contrast (F1)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Correlation (F2)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum of squares (F3)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Inverse difference moment (F4)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum average (F5)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum variance (F6)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum entropy (F7)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Entropy (F8)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Difference variance (F9)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Difference entropy (F10)',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Information measure of correlation 1 (F11)'] 

dataset,Y, slide, used_feats=mk_graph(to_use=to_use, use_res=False)
#model = torch.load(f'D:\\Results\\TMA_results\\models\\n11_fold_1.pt')
model = torch.load(f'D:\\Results\\TMA_results\\models\\good\\fold_0.pt')

G,_=getVisData(dataset,model,'cuda')
#G=showGraphDataset(G,mode='masks')

core_list=['5-A','6-A','9-C']

for core in core_list:
    plot_hist(G[core])

print('Done!')
    

