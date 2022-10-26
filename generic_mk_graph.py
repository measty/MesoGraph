from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import StandardScaler
import pickle

core_node = True

USE_CUDA = torch.cuda.is_available()
device = {True: "cuda", False: "cpu"}[USE_CUDA]


def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v


def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))


def toGeometricWW(X, W, y, tt=0):
    return Data(
        x=toTensor(X, requires_grad=False),
        edge_index=(toTensor(W, requires_grad=False) > tt).nonzero().t().contiguous(),
        y=toTensor([y], dtype=torch.long, requires_grad=False),
    )


def mk_graph(
    label_df=pd.read_csv(r"D:\TestGCN\heba\GraphLabels.csv"),
    adj_base=Path(r"D:\TestGCN\heba\adjacency"),
    attr_base=Path(r"D:\TestGCN\heba\Attributes"),
    to_use=None,
    core_node=True,
):

    Y_orig = label_df["p4EBP1"].values
    graphs = []
    Y = []

    if True:
        with open(Path(r"D:\TestGCN\heba\clust_temp.pkl"), "rb") as f:
            unpickler = pickle.Unpickler(f)
            graphs = unpickler.load()
            to_use = graphs[0].feat_names
            for i, fname in enumerate(label_df["FileName"]):
                y = {False: 0, True: 1}[Y_orig[i]]
                Y.append(y)
        return graphs, Y, to_use
    Xs = []
    for i, fname in enumerate(label_df["FileName"]):
        attr = pd.read_csv(attr_base.joinpath(fname[1:-1]), header=None)
        Xs.append(attr.to_numpy())

    Xmat = np.vstack(Xs)
    keep = np.logical_not(np.any(np.isnan(Xmat), 0))
    Xmat = Xmat[:, keep]
    norm = StandardScaler().fit(Xmat)

    for i, fname in enumerate(label_df["FileName"]):
        parts = fname.split("_")
        fname_mod = "_".join(parts[0:3]) + "_" + ("   " + parts[3])[-8:]
        adj = pd.read_csv(adj_base.joinpath(fname_mod[1:-1]), header=None)

        W = adj.to_numpy()
        if core_node:
            print("connecting virtual node..")
            W = np.vstack(
                (W, np.zeros((1, W.shape[1])))
            )  # zeros only connect TO core node - be a core rep but dont broadcast?
            W = np.hstack((W, np.zeros((W.shape[0], 1))))  # ones to connect to
        np.fill_diagonal(W, 0)

        X = norm.transform(Xs[i][:, keep])
        if core_node:
            # add virtual node feats as mean of core
            X = np.vstack((X, np.mean(X, axis=0)))
        to_use = [f"Feat{k}" for k in range(X.shape[1])]
        # y={0: 2, 1: 2, 2: 1, 3:0, 4:0}[Y_orig[i]]
        y = {False: 0, True: 1}[Y_orig[i]]
        Y.append(y)
        g = toGeometricWW(X, W, y)
        g.core = fname[1:-5]
        g.coords = toTensor(np.ones((X.shape[0], 2)))
        g.type_label = Y_orig[i]
        g.feat_names = to_use
        graphs.append(g)
        print(f"Done graph for core {g.core}")

    if True:
        with open(Path(r"D:\TestGCN\heba\clust_temp.pkl"), "wb") as output:
            pickler = pickle.Pickler(output, -1)
            pickler.dump(graphs)
            output.close()

    return graphs, Y, to_use
