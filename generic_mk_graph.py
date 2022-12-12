from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import StandardScaler
import pickle

"""Generic graph generation from adajcancy matrix, features,
and label datatframe for MesoGraph"""

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
    core_node=True,
    load_path=None,
    label_column = "p4EBP1",
    label_map = {False: 0, True: 1},
):
    """Make a graph from a metadata dataframe, an adjacency matrix, and
    a feature matrix.
    
    Args:
        label_df (pd.DataFrame): dataframe with metadata for each graph.
            Must contain a column named "FileName" with the name of the
            .csv file for each graph (same name in both adjacancy folder and
            attributes folder).
        adj_base (Path): path to directory containing adjacency matrices
        attr_base (Path): path to directory containing feature matrices
        core_node (bool): if True, appends a core node to each graph with features
            equal to the mean of all other nodes in the graph.
        load_path (Path): if not None, loads graphs from this path instead of
            generating them.
        label_column (str): name of column in label_df to use as label
        label_map (dict): dictionary mapping label values to 0/1

    Returns:
        graphs (list): list of torch_geometric.data.Data objects
        Y (list): list of labels for each graph
        to_use (list): list of feature names

    """
    Y_orig = label_df[label_column].values
    graphs = []
    Y = []

    if load_path is not None:
        with open(load_path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            graphs = unpickler.load()
            to_use = graphs[0].feat_names
            for i in range(len(graphs)):
                y = label_map[Y_orig[i]]
                Y.append(y)
        return graphs, Y, to_use
    Xs = []
    for i, fname in enumerate(label_df["FileName"]):
        attr = pd.read_csv(attr_base.joinpath(fname[1:-1]), header=None)
        Xs.append(attr.to_numpy())

    Xmat = np.vstack(Xs)
    # remove any features with nan
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
        y = label_map[Y_orig[i]]
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
