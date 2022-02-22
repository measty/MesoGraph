#from pyexpat import model
import torch.nn as nn
from torchaudio import datasets
from mk_graph import mk_graph
from pathlib import Path
import torch
import numpy as np
from gmil_edgeconv_cpath_old import GIN, getVisData, showGraphDataset, learnable_sig, change_pixel
from torch.utils.data import DataLoader
#from skimage.segmentation import flood, flood_fill
from PIL import Image
import cv2
import pandas as pd
from split_detections import split_det
import copy
import matplotlib.pyplot as plt
import sys
from graph_utils import label_dets_from_heatmap

sys.setrecursionlimit(10**4)

do_vis=True
base_paths=list(Path(r'E:\Meso_TCGA\Slides_tiled').glob('*'))
lab_df=pd.read_csv(r'D:\TCGA_Data\TCGA_WSI_labels_DX.csv')
lab_df.set_index('WSI-ID', inplace=True)

to_use=None
to_use=['Nucleus: Area µm^2', 'Nucleus: Circularity', 'Nucleus: Solidity', 'Nucleus: Max diameter µm', 'Nucleus: Min diameter µm', 'Cell: Area µm^2', 'Cell: Circularity', 'Cell: Solidity', 'Cell: Max diameter µm',
 'Cell: Min diameter µm', 'Nucleus/Cell area ratio', 'Hematoxylin: Nucleus: Mean', 'Hematoxylin: Nucleus: Median', 'Hematoxylin: Nucleus: Std.Dev.', 'Hematoxylin: Cytoplasm: Mean', 'Hematoxylin: Cytoplasm: Median', 'Hematoxylin: Cytoplasm: Std.Dev.', 'Hematoxylin: Cell: Mean', 'Hematoxylin: Cell: Median', 'Hematoxylin: Cell: Std.Dev.', 'Eosin: Nucleus: Mean', 'Eosin: Nucleus: Median', 'Eosin: Nucleus: Std.Dev.', 'Eosin: Cytoplasm: Mean', 'Eosin: Cytoplasm: Median', 'Eosin: Cytoplasm: Std.Dev.', 'Eosin: Cell: Mean', 'Eosin: Cell: Median', 'Eosin: Cell: Std.Dev.', 'Smoothed: 50 µm: Nucleus: Area µm^2', 'Smoothed: 50 µm: Nucleus: Circularity', 'Smoothed: 50 µm: Nucleus: Solidity', 'Smoothed: 50 µm: Nucleus: Max diameter µm', 'Smoothed: 50 µm: Nucleus: Min diameter µm', 'Smoothed: 50 µm: Cell: Area µm^2', 'Smoothed: 50 µm: Cell: Circularity', 'Smoothed: 50 µm: Cell: Solidity', 'Smoothed: 50 µm: Cell: Max diameter µm', 'Smoothed: 50 µm: Cell: Min diameter µm', 'Smoothed: 50 µm: Nucleus/Cell area ratio',
 'Smoothed: 50 µm: Hematoxylin: Nucleus: Mean', 'Smoothed: 50 µm: Hematoxylin: Nucleus: Median', 'Smoothed: 50 µm: Hematoxylin: Nucleus: Std.Dev.', 'Smoothed: 50 µm: Hematoxylin: Cytoplasm: Mean', 'Smoothed: 50 µm: Hematoxylin: Cytoplasm: Median', 'Smoothed: 50 µm: Hematoxylin: Cytoplasm: Std.Dev.', 'Smoothed: 50 µm: Hematoxylin: Cell: Mean', 'Smoothed: 50 µm: Hematoxylin: Cell: Median', 'Smoothed: 50 µm: Hematoxylin: Cell: Std.Dev.', 'Smoothed: 50 µm: Eosin: Nucleus: Mean', 'Smoothed: 50 µm: Eosin: Nucleus: Median', 'Smoothed: 50 µm: Eosin: Nucleus: Std.Dev.', 'Smoothed: 50 µm: Eosin: Cytoplasm: Mean', 'Smoothed: 50 µm: Eosin: Cytoplasm: Median', 'Smoothed: 50 µm: Eosin: Cytoplasm: Std.Dev.', 'Smoothed: 50 µm: Eosin: Cell: Mean', 'Smoothed: 50 µm: Eosin: Cell: Median', 'Smoothed: 50 µm: Eosin: Cell: Std.Dev.', 'Smoothed: 50 µm: Nearby detection counts', 'ROI: 0.44 µm per pixel: OD Sum: Mean', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Mean', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Std.dev.',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Angular second moment (F0)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Contrast (F1)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Correlation (F2)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum of squares (F3)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Inverse difference moment (F4)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum average (F5)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum variance (F6)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum entropy (F7)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Entropy (F8)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Difference variance (F9)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Difference entropy (F10)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Information measure of correlation 1 (F11)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Information measure of correlation 2 (F12)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Mean',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Std.dev.', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Angular second moment (F0)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Contrast (F1)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Correlation (F2)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum of squares (F3)',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Inverse difference moment (F4)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum average (F5)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum variance (F6)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum entropy (F7)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Entropy (F8)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Difference variance (F9)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Difference entropy (F10)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Information measure of correlation 1 (F11)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Information measure of correlation 2 (F12)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Mean', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Std.dev.', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Angular second moment (F0)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Contrast (F1)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Correlation (F2)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum of squares (F3)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Inverse difference moment (F4)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum average (F5)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum variance (F6)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum entropy (F7)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Entropy (F8)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Difference variance (F9)', 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Difference entropy (F10)',
 'Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Information measure of correlation 1 (F11)'] 

for n,base_path in enumerate(base_paths[0:1]):
    if n==8 or n==29:
        continue
    print(f'processing slide {n}: {base_path}')
    #base_path=Path(r'E:\Meso_TCGA\Slides_tiled\TCGA-UT-A88C-01Z-00-DX1.A3206848-4B03-49CB-AF3C-720272E97AEE')
    base_path=Path(r'E:\Meso_TCGA\Slides_tiled\TCGA-SC-A6LN-01Z-00-DX1.379BF588-5A65-4BF8-84CF-5136085D8A47')
    if do_vis:
        save_path=base_path.joinpath('outputs_g')
        save_path.mkdir()
        mask_path=base_path.joinpath('masks')
        flist=list(mask_path.glob('*.tiff'))
        fnames,tcoords={},{}
        for f in flist:
            fparts=f.name.split('_')
            fnames[fparts[0]]=f.name
            tcoords[fparts[0]]=[int(s) for s in f.stem.split('_')[1].split('-')]

    #det_path=
    df=pd.read_csv(base_path.joinpath('dets.csv'),sep='\t')
    df['label']={'Epithelioid': 'E', 'Biphasic': 'B', 'sarcomatoid': 'S'}[lab_df.loc[base_path.name+'.svs'].labels]
    #set column denoting if tumor or not
    df=label_dets_from_heatmap(Path(f'D:\TCGA_Data\Temp\mapsTCGA\{base_path.name}.svs'),df,0.5)
    dfs=split_det(df,'core',0)
    dfs=[dfs[key] for key in dfs.keys() if key!='Image']


    dataset, Y, slide, _ = mk_graph(dfs[0:], mode='WSI', to_use=to_use)
    #loader = DataLoader(dataset, shuffle=False)

    #model = GIN(dim_features=dataset[0].x.shape[1], dim_target=1, layers=[10,10,10,10,10,10],dropout = 0,pooling='mean',eps=100.0,train_eps=False, do_ls=True)
    model = torch.load(r'D:\Results\TMA_results\models\good\fold_0.pt')
    #model.load_state_dict(state_dict)

    G,_=getVisData(dataset,model,'cuda')
    G=showGraphDataset(G,mode='masks')

    if do_vis:
        tcarray=[tcoords[key] for key in G.keys()]
        tcarray=np.array(tcarray)
        top_left=np.min(tcarray,axis=0)
        mpp=0.5015   #need to get auto

    X_stack, W_stack,c_stack,it_stack=[],[],[],[]
    cum_pts=[0]
    for key in G.keys():
        Xn = G[key].coords.detach().cpu().numpy()
        Wn= np.array([e for e in G[key].edge_index.t().detach().cpu().numpy()])
        if do_vis:
            Xn_global=Xn/mpp - top_left
            X_stack.append(Xn_global)
            Xn=Xn/mpp - np.array(tcoords[key])   # get coordinates
        else:
            X_stack.append(Xn)
        W_stack.append(Wn+cum_pts[-1])
        cum_pts.append(cum_pts[-1]+Xn.shape[0])
        c=G[key].c[:-1,0:2]
        istumor=G[key].istumor[0]
        it_stack.append(istumor[:,None])
        c_stack.append(c)   #c rgb: r-sarc score, g-ep score, b=0
        core=G[key].core[0]
        if do_vis:
            fmask=Image.open(mask_path.joinpath(fnames[key]))
            im=np.array(fmask, dtype=np.float32)
            im[im==1]=255
            im[im==2]=0
            #im=255-im    #cells will now be 0
            alpha=(im==255)*255
            #im=np.flipud(im)
            oob=np.all(Xn<(im.shape[0]-1),axis=1)
            Xn=Xn[oob,:] #remove
            c=c[oob,:]
            istumor=istumor[oob]
            cvals=[]
            for i in range(len(Xn)):
                if istumor[i]:
                    cvals.append(0.5+(c[i,0]**2-c[i,1]**2)/2)
                else:
                    cvals.append(253)
                cval=cvals[-1]
                #change_pixel(im, int(Xn[i,1]), int(Xn[i,0]), cval)
                # Fill a square near the middle with value 127, starting at index (76, 76)
                cv2.floodFill(im, None, seedPoint=(int(Xn[i,0]), int(Xn[i,1])), newVal=cvals[-1])
            #save im
            tum_mask=im==253
            print(f'saving {key}')
            im[im>=253]=0
            #im=Image.fromarray((im*255/np.max(im)).astype('uint8'))
            im=cv2.applyColorMap((im*255/np.max(im)).astype('uint8'),cv2.COLORMAP_TURBO)
            im[tum_mask,:]=254
            im=np.dstack((im,alpha.astype('uint8')))
            cv2.imwrite(str(save_path.joinpath(fnames[key]).with_suffix('.png')),im)
            #im.save(save_path.joinpath(key+'.jpg'))
        
    np.savez(base_path.joinpath('graph.npz'),X=np.vstack(X_stack),W=np.vstack(W_stack),c=np.vstack(c_stack),cum_pts=np.array(cum_pts),it=np.vstack(it_stack))

print('done!')
#model = model.to('CUDA')
#model.eval()

#for data in loader:
    #data = data.to('CUDA')
    #output,xx = model(data)







'''
from tiatoolbox.models.abc import ModelABC
class Model(ModelABC):
    def __init__(self):
        # your code here
        pass

    @staticmethod
    def infer_batch(self):
        # your code here
        pass

    @staticmethod
    def preproc():
        # your code here
        pass

    @staticmethod
    def postproc():
        # your code here
        pass

    def forward(self):
        # your code here
        pass
    '''