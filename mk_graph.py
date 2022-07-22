from scipy.spatial import distance_matrix, Delaunay
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import StandardScaler
from split_detections import map_ind

USE_CUDA = torch.cuda.is_available()
device = {True:'cuda',False:'cpu'}[USE_CUDA] 
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))

def toGeometricWW(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

def connectClusters(C,w=[],core_node=False,dthresh = 3000):
    
    if len(w)==0:
        #W =  NearestNeighbors(radius=15).fit(C).radius_neighbors_graph(C).todense()
        #W =  NearestNeighbors(n_neighbors=11).fit(C).kneighbors_graph(C).todense()
        W =  NearestNeighbors(n_neighbors=11).fit(C).kneighbors_graph(C, mode='distance')
        W[W>50]=0
        W[W>0]=1
        W=W.todense()
    else:
        #dist = DistanceMetric.get_metric('wminkowski', p=2, w=w)
        r=(20/500)*0.5
        W =  NearestNeighbors(radius=r,metric='wminkowski', metric_params={'p': 2, 'w': w}).fit(C).radius_neighbors_graph(C).todense()
    
    if core_node:
        print('connecting virtual node..')
        W=np.vstack((W,np.zeros((1,W.shape[1])))) # zeros only connect TO core node - be a core rep but dont broadcast?
        W=np.hstack((W,np.ones((W.shape[0],1))))  
    np.fill_diagonal(W,0)

    return W
    
    # tess = Delaunay(Cc)
    # neighbors = defaultdict(set)
    # for simplex in tess.simplices:
    #     for idx in simplex:
    #         other = set(simplex)
    #         other.remove(idx)
    #         neighbors[idx] = neighbors[idx].union(other)
    # nx = neighbors    
    # W = np.zeros((Cc.shape[0],Cc.shape[0]))
    # for n in nx:
    #     nx[n] = np.array(list(nx[n]),dtype = np.int)
    #     nx[n] = nx[n][KDTree(Cc[nx[n],:]).query_ball_point(Cc[n],r = dthresh)]
    #     W[n,nx[n]] = 1.0
    #     W[nx[n],n] = 1.0        
    # return W # neighbors of each cluster and an affinity matrix

def asblob(graphs,Y,slide):
    feat_stack, edge_stack, labels=[],[],[]

    for i,g in enumerate(graphs):
        feat_stack.append(g.x)
        edge_stack.append(g.edge_index)
        labels.append(Y[i]*np.ones((g.x.shape[0],1)))


def slide_fold(slide_inds):
    slides=np.unique(slide_inds)
    for s in slides:
        yield [i for i, x in enumerate(slide_inds) if x!=s], [i for i, x in enumerate(slide_inds) if x==s]

def set_core_origin(X,core,core_cents,core_node=False, core_width=2854):
    um_per_pix=0.4415
    
    '''core_cents=[]
    for i in range(4,8):
        TMA=pd.read_csv(Path(f'D:\QuPath_Projects\Meso_TMA\Dearrayed\MESO_{i}\TMA results.txt'), sep='\t')
        core_cents.append(TMA[['Name', 'Centroid X µm', 'Centroid Y µm']])
    core_cents=pd.concat(core_cents,ignore_index=True)
    core_cents.to_csv(Path('D:\QuPath_Projects\Meso_TMA\Dearrayed\core_cents.csv'))'''

    cent=core_cents[core_cents['Name']==core][['Centroid X µm', 'Centroid Y µm']].to_numpy()
    top_left=cent-(core_width/2)*um_per_pix
    X=(X-top_left)/um_per_pix
    if core_node:
        #add extra node at center
        X=np.vstack((X,np.array([int(core_width/2),int(core_width/2)])))

    return X


def mk_graph(dataset='meso', mode='TMA',to_use=None, use_res=False):
    if dataset == 'meso':
        p=Path(r'D:\QuPath_Projects\Meso_TMA\per_core_dets')
        use_JL=True
        core_width = 2854
    else:
        p=Path(r'D:\Mesobank_TMA\per_core_dets')
        use_JL=False
        core_width = 1462
    if not isinstance(p, Path):
        df=p[0]
        read_files=False
    else:
        det_list=list(p.glob('*.csv'))
        df=pd.read_csv(det_list[0])
        read_files=True
    core_node=True
    keep_level=1

    if mode=='TMA':
        #core_cents=pd.read_csv(Path('D:\QuPath_Projects\Meso_TMA\Dearrayed\core_cents.csv'))
        core_cents=pd.read_csv(p.parent/'core_cents.csv')
        JL_labels=pd.read_csv(Path('D:\Meso\JL_labels.csv'))
        pres=pd.read_csv(Path('D:\All_cores\core_labels_pres.csv'))
    else:
        TCGA_df=pd.read_csv(r'D:\TCGA_Data\TCGA_WSI_labels_DX.csv')
        use_JL=False

    if to_use==None:
        columns=df.columns
        ind_preamble=list(columns).index('Centroid Y µm')
        to_use=columns[ind_preamble+1:-1]
        to_use=[col for col in to_use if 'label' not in col]
        #to_use=[col for col in to_use if 'Circularity' not in col]
        #to_use=[col for col in to_use if 'diameter' not in col]
        to_use=[col for col in to_use if 'Length' not in col]
        to_use=[col for col in to_use if 'Delaunay' not in col]
        #to_use.append('Nucleus: Circularity')
        to_use=[col for col in to_use if 'Detection probability' not in col]
        #to_use=[col for col in to_use if 'Smoothed' not in col]
        #to_use=[col for col in to_use if 'Median' not in col]
        to_use=[col for col in to_use if 'Cluster' not in col]
        #to_use=[col for col in to_use if 'Hematoxylin' not in col]
        #to_use=[col for col in to_use if 'Eosin' not in col]
        #to_use=[col for col in to_use if 'OD Sum' not in col]
        #to_use.append('Smoothed: 50 µm: Nearby detection counts')
        #to_use=[col for col in to_use if 'Cell' not in col]
        #to_use=[col for col in to_use if 'Cytoplasm' not in col]
        #to_use=[col for col in to_use if 'Diameter' not in col]
        #to_use=[col for col in to_use if 'Haralick' not in col]
        '''to_use=['Nucleus: Circularity',
            'Nucleus: Area µm^2',
            'Hematoxylin: Nucleus: Mean',
            #'ROI: 0.44 µm per pixel: OD Sum: Mean',
            'Eosin: Nucleus: Mean',
            'Smoothed: 50 µm: Nearby detection counts',
            #'Nucleus: Length µm'
            'Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Entropy (F8)'
            ]'''
    print(f'using {len(to_use)} features: ')
    print(to_use)
    #to_use=res_cols

    use_res=True
    if use_res:
        res_cols=[f'res{i}' for i in range(512)]
        to_use=to_use+res_cols

    if read_files:
        dfs=[]
        for dets in det_list:
            #slide.append(map_ind(dets.stem))
            dfs.append(pd.read_csv(dets))
            if use_res:
                res_feats=np.load(p.parent/'det_res_feats_snorm'/(dfs[-1].Parent.iloc[0]+'.npy'))
                dfs[-1][res_cols]=res_feats[:,0:-2]       
    else:
        dfs=p
    all_dets=pd.concat(dfs,ignore_index=True)
    norm=StandardScaler().fit(all_dets[to_use].to_numpy())
    
    graphs,Y,slide=[],[],[]
    for df in dfs:
        if mode=='TMA' and dataset == 'meso' and ((df.Parent.iloc[0] not in pres.Core.values) or pres[pres.Core==df.Parent.iloc[0]].Quality.values[0]>keep_level):
            continue
        if mode=='TMA':
            df=df[['Parent','Centroid X µm','Centroid Y µm']+to_use+['label']]#.dropna()
            df=df.fillna(df.mean())
        else:
            df=df[['Parent','Centroid X µm','Centroid Y µm']+to_use+['label','istumor']]#.dropna()
            df=df.fillna(df.mean())
        #if mode=='TMA': slide.append(map_ind(df.Parent.iloc[0]))
        X_xy=df[['Centroid X µm','Centroid Y µm']].to_numpy()/500

        if use_JL and df.Parent.iloc[0] in JL_labels.Core.values:
            df.label.iloc[0]=JL_labels[JL_labels.Core==df.Parent.iloc[0]].JL.values[0]

        X=norm.transform(df[to_use].to_numpy())
        if core_node:
            #add virtual node feats as mean of core
            X=np.vstack((X,np.mean(X, axis=0)))

        W=connectClusters(df[['Centroid X µm','Centroid Y µm']].to_numpy(),core_node=core_node)
        alpha=0.25
        w=(1-alpha)*np.array([0.25,0.25]+[0.5*alpha/len(to_use)]*len(to_use))
        #W=connectClusters(np.concatenate((X_xy,X),axis=1),w,core_node=core_node)

        if False:
            with open(Path('/home/marke/MESO/clust_temp.pkl'), 'wb') as output:
                pickler = pickle.Pickler(output, -1)
                pickler.dump(W)
                output.close()

        if sum(sum(np.isnan(X)))!=0:
            print('eek')

        y={'E': 0, 'B': 1, 'S': 2}[df.label.iloc[0]]
        Y.append(y)
        g=toGeometricWW(X,W,y)
        g.core=df.Parent.iloc[0]
        g.type_label=df.label.iloc[0]
        g.feat_names=to_use
        if mode=='TMA':
            g.coords = toTensor(set_core_origin(df[['Centroid X µm','Centroid Y µm']].to_numpy(),g.core,core_cents, core_node, core_width))
        else:
            g.coords = toTensor(df[['Centroid X µm','Centroid Y µm']].to_numpy())
            g.istumor=df['istumor'].to_numpy()
        graphs.append(g)
        print(f'Done graph for core {g.core}')

    return graphs, Y, slide, to_use

if __name__=='__main__':
    mk_graph()



