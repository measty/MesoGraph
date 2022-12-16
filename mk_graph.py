import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import StandardScaler
from utils import map_ind
from tiatoolbox.annotation.storage import SQLiteStore
from utils import toTensor
import pickle

"""Graph generation from detected cells/objects for MesoGraph"""

USE_CUDA = torch.cuda.is_available()
device = {True: "cuda", False: "cpu"}[USE_CUDA]


def toGeometricWW(X, W, y, tt=0):
    return Data(
        x=X,
        edge_index=(toTensor(W, requires_grad=False) > tt).nonzero().t().contiguous(),
        y=toTensor([y], dtype=torch.long, requires_grad=False),
    )


def connectClusters(C, w=[], core_node=False, dthresh=3000):

    if len(w) == 0:
        W = NearestNeighbors(radius=60).fit(C).radius_neighbors_graph(C).todense()
        # W =  NearestNeighbors(n_neighbors=11).fit(C).kneighbors_graph(C).todense()
        # W =  NearestNeighbors(n_neighbors=9).fit(C).kneighbors_graph(C, mode='distance')
        # W[W>50]=0    #to use nearest neighbors with dist cutoff
        # W[W>0]=1
        # W=W.todense()
    else:
        # dist = DistanceMetric.get_metric('wminkowski', p=2, w=w)
        r = (20 / 500) * 0.5
        W = (
            NearestNeighbors(
                radius=r, metric="wminkowski", metric_params={"p": 2, "w": w}
            )
            .fit(C)
            .radius_neighbors_graph(C)
            .todense()
        )

    if core_node:
        print("connecting virtual node..")
        W = np.vstack(
            (W, np.zeros((1, W.shape[1])))
        )  # zeros only connect TO core node - be a core rep but dont broadcast?
        # W=np.hstack((W,np.ones((W.shape[0],1))))
        W = np.hstack(
            (W, np.zeros((W.shape[0], 1)))
        )  # dont connect the core node at all. Just use as source of mean feats
    np.fill_diagonal(W, 0)

    return W


def slide_fold(slide_inds):
    slides = np.unique(slide_inds)
    for s in slides:
        yield [i for i, x in enumerate(slide_inds) if x != s], [
            i for i, x in enumerate(slide_inds) if x == s
        ]


def set_core_origin(X, core, core_cents, core_node=False, core_width=2854):
    um_per_pix = 0.4415

    """core_cents=[]
    for i in range(4,8):
        TMA=pd.read_csv(Path(f'D:\QuPath_Projects\Meso_TMA\Dearrayed\MESO_{i}\TMA results.txt'), sep='\t')
        core_cents.append(TMA[['Name', 'Centroid X µm', 'Centroid Y µm']])
    core_cents=pd.concat(core_cents,ignore_index=True)
    core_cents.to_csv(Path('D:\QuPath_Projects\Meso_TMA\Dearrayed\core_cents.csv'))"""

    cent = core_cents[core_cents["Name"] == core][
        ["Centroid X µm", "Centroid Y µm"]
    ].to_numpy()
    top_left = cent - (core_width / 2) * um_per_pix
    X = (X - top_left) / um_per_pix
    if core_node:
        # add extra node at center
        X = np.vstack((X, np.array([int(core_width / 2), int(core_width / 2)])))

    return X


def graph_from_db(db_path, to_use, core_node=True, use_res=False):
    """Construct graph from an annotation store of cell detections
    from a TMA core.
    """
    SQ = SQLiteStore(db_path)
    props = [ann.properties for ann in SQ.values()]
    df = pd.DataFrame(props)

    if use_res:
        res_cols = [f"res{i}" for i in range(512)]
        # to_use=to_use+res_cols
        res_feats = np.load(
            db_path.parent.parent / "det_res_feats_snorm" / (f"{db_path.stem}.npy")
        )
        # df[res_cols]=res_feats[:,0:-2]
        df = pd.concat(
            [df, pd.DataFrame(res_feats[:, 0:-2], index=df.index, columns=res_cols)],
            axis=1,
        )

    df = df[["Centroid X µm", "Centroid Y µm"] + to_use]
    df = df.fillna(df.mean())

    X = df[to_use].to_numpy()
    if core_node:
        # add virtual node feats as mean of core
        X = np.vstack((X, np.mean(X, axis=0)))

    W = connectClusters(
        df[["Centroid X µm", "Centroid Y µm"]].to_numpy(), core_node=core_node
    )
    y = {"E": 0, "B": 1, "S": 2}[db_path.stem.split("_")[1]]
    g = toGeometricWW(X, W, y)
    g.core = db_path.stem.split("_")[0]
    g.type_label = db_path.stem.split("_")[1]
    g.feat_names = to_use
    coords = df[["Centroid X µm", "Centroid Y µm"]].to_numpy()
    if core_node:
        coords = np.vstack((coords, np.mean(coords, axis=0)))
    g.coords = toTensor(coords)
    print(f"Done graph for core {g.core}")
    return g, to_use


def mk_graphs(dataset="meso", to_use=None, use_res=True, load_graphs=None):
    """Construct graphs from annotation stores of cell detections, or load
    pre-constructed graphs from pickle files.
    
    Args:
        dataset (str, optional): Dataset to use. Defaults to "meso".
        to_use (list, optional): List of features to use. If not provided, will
            use all features.
        use_res (bool, optional): Whether to use resnet features.
        load_graphs (str, optional): Path to folder containing pre-constructed
            graphs. If not provided, will construct graphs from scratch.
    Returns:
        graphs (list): List of graphs.
        slide (list): List of slide indices.
        Y (list): List of labels.
        to_use (list): List of features used.
    """
    core_node = True
    if dataset == "meso":
        p = Path(r"D:\QuPath_Projects\Meso_TMA\detections\stores")
    elif dataset == "mesobank":
        p = Path(r"D:\Mesobank_TMA\mesobank_proj\detections\stores")
    if load_graphs:
        graphs = []
        slide, Y = [], []
        needs_load = True
        gr_list = list(load_graphs.glob("*.pkl"))
        if len(gr_list) == 1:
            # all the graphs are in one file
            with open(gr_list[0], "rb") as f:
                gr_list = pickle.load(f)
                needs_load = False
        for gr in gr_list:
            if needs_load:
                with open(gr, "rb") as f:
                    g = pickle.load(f)
            else:
                g = gr
            graphs.append(g)
            if "meso" in dataset:
                slide.append(map_ind(g.core.split("_")[0], dataset))
            Y.append(g.y[0].cpu().item())
        return graphs, slide, Y, g.feat_names

    db_paths = list(p.glob("*.db"))
    SQ = SQLiteStore(db_paths[0])
    props = SQ.pquery("*", unique=False)
    df = pd.DataFrame.from_dict(props, orient="index")

    df_columns_to_ignore = ["label", "Length", "Delaunay", "Detection probability", "Cluster", "Centroid", "Cell"]
    if to_use == None:
        columns = df.columns
        to_use = columns
        for ignore in df_columns_to_ignore:
            to_use = [col for col in to_use if ignore not in col]

    if use_res:
        res_cols = [f"res{i}" for i in range(512)]
        to_use = to_use + res_cols

    print(f"using {len(to_use)} features: ")
    print(to_use)

    graphs, slide, Y = [], [], []
    for db_path in db_paths:
        g, to_use = graph_from_db(db_path, to_use, core_node=core_node, use_res=use_res)
        graphs.append(g)
        slide.append(map_ind(db_path.stem.split("_")[0], dataset))
        Y.append(g.y[0].cpu().item())

    # normalize feats
    X_stack = []
    for g in graphs:
        X_stack.append(g.x[::2,:]) # sample half the nodes for memory
    X_stack = np.vstack(X_stack)
    norm = StandardScaler().fit(X_stack)
    X_stack = None
    for g in graphs:
        g.x = toTensor(norm.transform(g.x), requires_grad=False)

    if not load_graphs:
        # save graphs
        by_core = True
        if not by_core:
            with open(p.parent / "graphs.pkl", "wb") as f:
                pickle.dump(graphs, f)
        else:
            # save the graphs in separate files per core
            (p.parent / "graphs_").mkdir(exist_ok=True)
            for g in graphs:
                with open(p.parent / "graphs_" / f"{g.core}.pkl", "wb") as f:
                    pickle.dump(g, f)

    return graphs, slide, Y, to_use


if __name__ == "__main__":
    mk_graphs()
