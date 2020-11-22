#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:18:29 2020

@author: u1876024
"""


from GNN import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
#import umap
from skimage import exposure
from sklearn.decomposition  import PCA

def plotGraph(Xn, Wn, c=(0,0,0),node_size=50,edge_alpha=1.0,axlim=None): 
    """
    Makes & shows a networkx graph based on node positions Xn, Adjacency Wn and color c
    """
    G=nx.Graph()
    G.add_nodes_from(list(range(Xn.shape[0])))
    G.add_edges_from(Wn)   
    pos = dict(zip(range(Xn.shape[0]),Xn))    
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width = 0.5);    
    nx.draw_networkx_nodes(G, pos, node_color=c,node_size=node_size)
    if axlim is not None:
        axes = plt.gca()
        
        axes.set_xlim([axlim[0],axlim[1]]) 
        axes.set_ylim([axlim[2],axlim[3]]) 
        axes.set_yticks(np.round(np.linspace(axlim[2], axlim[3], 3), 2))
        axes.set_xticks(np.round(np.linspace(axlim[0], axlim[1], 3), 2))

        plt.grid()

def showGraph(g,axlim=None):
    """
    Show single pytorch graph object 
    G.x: features
    G.v: features to be used for visualization
    G.c: used for colors (based on G.v or G.x)
    G.coords: position coordinates
    G.egde_index
    G.y
    """
    # try:
    #     coords = toNumpy(g.coords)     # get coordinates
    # except:
    #     coords = g.x[:,:2]  #if no coordinates use color specs
    if hasattr(g,'c'):
        c = g.c
    else:
        c=np.array([[0,0,0]])
    plotGraph(toNumpy(g.coords),[tuple(e) for e in toNumpy(g.edge_index.t())],c=c,node_size=20,edge_alpha=0.25,axlim=axlim)
    
def showGraphDataset(G,max_figs = np.inf):
    """
    Visualize a graph dataset through dimensionality reduction
    """
    G = deepcopy(G)
    try: #if visualization data is available
        X = np.vstack([toNumpy(g.v) for g in G])
    except: #otherwise use node features
        X = np.vstack([toNumpy(g.x) for g in G])
    L = np.cumsum([0]+[g.x.shape[0] for g in G])
    Y = [g.y for g in G]
    pos = np.sum([toNumpy(g.y)==1 for g in G])
    neg = len(G)-pos
    # coordinates - if more than 2D then PCA is used to reduce dim
    try:
        Coords = np.vstack([toNumpy(g.coords) for g in G])
    except:
        Coords = np.vstack([toNumpy(g.x) for g in G])
    if Coords.shape[1]>2:
        tx = PCA()
        Coords = tx.fit_transform(StandardScaler().fit_transform(Coords))[:,[0,1]]
    if Coords.shape[1]<2:
        CC = np.zeros((Coords.shape[0],2))
        CC[:,1]=Coords[:,0]
        CC[:,0]=Coords[:,0]
        Coords = CC
    axlim = [np.min(Coords[:,0])-0.01,np.max(Coords[:,0])+0.01,np.min(Coords[:,1])-0.01,np.max(Coords[:,1])+0.01]
    #import pdb;pdb.set_trace()
    #Determine node coloring based on visualization data
    X = StandardScaler().fit_transform(X)
    if X.shape[1]>3:        
        tx = PCA()#umap.UMAP(n_components=3,n_neighbors=6,min_dist = 0.0)#
        Xp = tx.fit_transform(X)[:,[0,1,2]]
    else:
        Xp = np.zeros((X.shape[0],3))
        Xp[:,:X.shape[1]]=X    
    Xp = np.clip(MinMaxScaler().fit_transform(Xp), 1e-4, 1.0-1e-4)
    #import pdb;pdb.set_trace()
    #for i in range(min(Xp.shape[1],X.shape[1])):
    #    Xp[:,i] = exposure.equalize_hist(Xp[:,i])
    #Xp[:,1]=1-Xp[:,0]
    #Xp = (1-Xp)**2
    #Xp = (1-Xp[:,[2,1,0]])**2
    #fig, axs = plt.subplots(2, max(pos,neg))
    
    fig, axs = plt.subplots(2, min(max_figs,max(pos,neg)))
    plt.subplots_adjust(wspace=0, hspace=0)

    counts = [0,0]
    for i,g in enumerate(G):    
        g.c = Xp[L[i]:L[i+1]]
        g.coords = Coords[L[i]:L[i+1]]
        y = int(g.y)
        if counts[y]>=max_figs:
            continue
        ax = axs[y,counts[y]]
        counts[y]+=1
        plt.sca(ax)
        #import pdb;pdb.set_trace()
        
        showGraph(g,axlim)
        plt.title("{:.2f}".format(toNumpy(g.z)[0]),fontsize= 8)
        
    return fig


    
def getVisData(data,model,device,color = 'x', coord = 'h'):
    """
    Get a pytorch dataset for node representation based on model output
    The node feaures of the input data are replaced with model based node repn
    color: x/h/z 
    coord: x/h/z
    x: original features
    h: layer output features
    z: prediction score
    """
    Z,Y,ZXn = decision_function(model,data,device,outOnly=False)    
    from platt import PlattScaling    
    ps = PlattScaling()
    Z = ps.fit_transform(toNumpy(Y),toNumpy(Z))**2
    #Z = 1/(1 + np.exp(-6*(2*Z-1.0))) 
    Z = toTensor(Z,requires_grad = False)
    G = []    
    for i, d in enumerate(data):
        (zn,xn) = ZXn[i]
        zn = toTensor(ps.transform(toNumpy(zn)),requires_grad = False)
        vd={'x':d.x,'h':xn,'z':zn}
        
        if hasattr(d, 'coords'): 
            cc = d.coords
        else:
            cc = vd[coord]           
        if hasattr(d, 'v'):
            vv = d.v
        else:
            vv = vd[color]
        G.append(Data(x=d.x,v=vv, edge_index=d.edge_index,y=d.y,coords=cc,z=Z[i]))
    return G
#%%
def showImageDataset(G):
    X = np.vstack([toNumpy(g.v) for g in G])
    b,s = np.mean(X),np.std(X)
    pos = np.sum([toNumpy(g.y)==1 for g in G])
    neg = len(G)-pos
    max_figs = 10
    fig, axs = plt.subplots(4, min(max_figs,max(pos,neg)))
    plt.subplots_adjust(wspace=0, hspace=None)
    

    counts = [0,0]
    for i,g in enumerate(G):
        
        y = int(g.y)
        if counts[y]>=max_figs:
            continue
        ZZ,Z0 = showImage(g,b,s)
        ax = axs[2*y,counts[y]]        
        plt.sca(ax)        
        plt.imshow(Z0,origin='lower')
        #import pdb;pdb.set_trace()
        plt.title("{:.2f}".format(toNumpy(g.z)[0]))
        
        ax = axs[2*y+1,counts[y]]        
        plt.sca(ax)        
        plt.imshow(ZZ,origin='lower')
        counts[y]+=1
def showImage(g,b=0,s=1):
    vn = toNumpy(g.v)
    vn = (vn-0.8)*(vn>0.8)/0.2
    ZZ = np.zeros((24*32,28*32,3))
    Z0 = np.zeros((24*32,28*32,3))
    for k,z in enumerate(g.x):
        i,j = np.asarray(toNumpy(g.coords[k]),dtype=np.int)
        x = toNumpy(g.x[k].reshape(32,32,3))     
        Z0[i*32:(i+1)*32,j*32:(j+1)*32,:]=x
        ZZ[i*32:(i+1)*32,j*32:(j+1)*32,:]=x*(vn[k])
    return ZZ,Z0