import torch.nn as nn
from torchaudio import datasets
from mk_graph import mk_graphs
from pathlib import Path
import torch
import math
import numpy as np
from meso_models import MesoBranched, MesoSep
from torch.utils.data import DataLoader

# from skimage.segmentation import flood, flood_fill
from PIL import Image
import cv2
import pandas as pd
from split_detections import split_det
import copy
import pickle
import matplotlib.pyplot as plt
import sys
from torch_geometric.nn.models import GNNExplainer
from bokeh.layouts import row
from bokeh.io import show
from bokeh.plotting import figure
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_regression, f_regression
from bokeh.io import export_svg


"""Functions to facilitate GNNexplainer interpretation of MesoGraph models"""

sys.setrecursionlimit(10**4)


def sort_feats(xdf, feat_names, stat=mutual_info_regression):
    feat_sel = SelectKBest(stat, k=20)
    feat_sel.fit(xdf[feat_names].values, xdf["score"].values)
    f_scores = feat_sel.scores_
    inds = np.argsort(f_scores)
    sort_feats = np.flip(feat_names[inds])
    return list(sort_feats)


def explain_net(core, node_idx=None, N=10, type="feature"):
    explainer = GNNExplainer(
        model, epochs=200, num_hops=3, return_type="raw", feat_mask_type=type
    )
    # core=dataset[5]
    x = core.x
    edge_index = core.edge_index
    # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)
    if type == "scalar":
        return node_feat_mask
    if type == "feature":
        topN = np.flipud(np.argsort(node_feat_mask.to("cpu")))[:N]
        topN_score = node_feat_mask.cpu().numpy()[topN]
    elif type == "individual features":
        topN = np.fliplr(np.argsort(node_feat_mask.to("cpu"), axis=1))[:, :N]
        topN_score = np.vstack(
            [node_feat_mask.cpu().numpy()[i, topN[i, :]] for i in range(topN.shape[0])]
        )
    # node_imp=[np.mean(edge_mask[np.any((edge_index.T==n).cpu().numpy(), axis=1)].cpu().numpy()) for n in range(x.shape[0])]

    # im=plt.imread(f'D:\All_cores\{core.core}.jpg')
    # plt.imshow(im)
    # ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, threshold=0.5)
    # plt.show()

    return topN, topN_score, node_feat_mask.cpu().numpy()  # , node_imp


def get_top_feats(feat_imp, used_feats):
    feat_order = np.fliplr(np.argsort(feat_imp)[None, :])
    imp_sorted = feat_imp[np.squeeze(feat_order)[10:0:-1]]
    top_feats = np.array(used_feats)[np.squeeze(feat_order)[10:0:-1]]
    return imp_sorted, top_feats, feat_order


def make_chart(feat_imp, used_feats, txt, type="box"):
    if type == "bar":
        feat_imp = np.mean(feat_imp, axis=0)
        vals, top_feats, _ = get_top_feats(feat_imp, used_feats)

        p = figure(
            y_range=top_feats,
            height=500,
            width=400,
            title=f"Top feats: {txt}",
            toolbar_location=None,
            tools="",
        )

        p.hbar(y=top_feats, right=vals, height=0.5)

        # p.xgrid.grid_line_color = None
        p.x_range.start = 0.6 * vals[0]
        p.yaxis.major_label_text_font_size = "24px"

        show(p)
    else:
        # box plot
        mean_feat_imp = np.mean(feat_imp, axis=0)
        vals, top_feats, inds = get_top_feats(mean_feat_imp, used_feats)
        feat_imp = feat_imp[:, np.squeeze(inds)[10:0:-1]]
        qs = np.percentile(feat_imp, [25, 50, 75], axis=0)
        iqr = qs[2, :] - qs[0, :]
        upper = np.minimum(qs[2, :] + 1.5 * iqr, np.max(feat_imp, axis=0))
        lower = np.maximum(qs[0, :] - 1.5 * iqr, np.min(feat_imp, axis=0))

        p = figure(
            tools="",
            title=f"Top feats: {txt}",
            background_fill_color="#efefef",
            y_range=top_feats,
            toolbar_location=None,
            height=500,
            width=400,
            output_backend="svg",
        )

        # stems
        p.segment(upper, top_feats, qs[2, :], top_feats, line_color="black")
        p.segment(lower, top_feats, qs[0, :], top_feats, line_color="black")

        # boxes
        p.hbar(
            top_feats, 0.7, qs[1, :], qs[2, :], fill_color="#E08E79", line_color="black"
        )
        p.hbar(
            top_feats, 0.7, qs[0, :], qs[1, :], fill_color="#3B8686", line_color="black"
        )

        # whiskers (almost-0 height rects simpler than segments)
        p.rect(lower, top_feats, 0.01, 0.2, line_color="black")
        p.rect(upper, top_feats, 0.01, 0.2, line_color="black")
        return p


to_use = None
to_use = [
    "Nucleus: Area µm^2",
    "Nucleus: Circularity",
    "Nucleus: Solidity",
    "Nucleus: Max diameter µm",
    "Nucleus: Min diameter µm",
    "Cell: Area µm^2",
    "Cell: Circularity",
    "Cell: Solidity",
    "Cell: Max diameter µm",
    "Cell: Min diameter µm",
    "Nucleus/Cell area ratio",
    "Hematoxylin: Nucleus: Mean",
    "Hematoxylin: Nucleus: Median",
    "Hematoxylin: Nucleus: Std.Dev.",
    "Hematoxylin: Cytoplasm: Mean",
    "Hematoxylin: Cytoplasm: Median",
    "Hematoxylin: Cytoplasm: Std.Dev.",
    "Hematoxylin: Cell: Mean",
    "Hematoxylin: Cell: Median",
    "Hematoxylin: Cell: Std.Dev.",
    "Eosin: Nucleus: Mean",
    "Eosin: Nucleus: Median",
    "Eosin: Nucleus: Std.Dev.",
    "Eosin: Cytoplasm: Mean",
    "Eosin: Cytoplasm: Median",
    "Eosin: Cytoplasm: Std.Dev.",
    "Eosin: Cell: Mean",
    "Eosin: Cell: Median",
    "Eosin: Cell: Std.Dev.",
    "Smoothed: 50 µm: Nucleus: Area µm^2",
    "Smoothed: 50 µm: Nucleus: Circularity",
    "Smoothed: 50 µm: Nucleus: Solidity",
    "Smoothed: 50 µm: Nucleus: Max diameter µm",
    "Smoothed: 50 µm: Nucleus: Min diameter µm",
    "Smoothed: 50 µm: Cell: Area µm^2",
    "Smoothed: 50 µm: Cell: Circularity",
    "Smoothed: 50 µm: Cell: Solidity",
    "Smoothed: 50 µm: Cell: Max diameter µm",
    "Smoothed: 50 µm: Cell: Min diameter µm",
    "Smoothed: 50 µm: Nucleus/Cell area ratio",
    "Smoothed: 50 µm: Hematoxylin: Nucleus: Mean",
    "Smoothed: 50 µm: Hematoxylin: Nucleus: Median",
    "Smoothed: 50 µm: Hematoxylin: Nucleus: Std.Dev.",
    "Smoothed: 50 µm: Hematoxylin: Cytoplasm: Mean",
    "Smoothed: 50 µm: Hematoxylin: Cytoplasm: Median",
    "Smoothed: 50 µm: Hematoxylin: Cytoplasm: Std.Dev.",
    "Smoothed: 50 µm: Hematoxylin: Cell: Mean",
    "Smoothed: 50 µm: Hematoxylin: Cell: Median",
    "Smoothed: 50 µm: Hematoxylin: Cell: Std.Dev.",
    "Smoothed: 50 µm: Eosin: Nucleus: Mean",
    "Smoothed: 50 µm: Eosin: Nucleus: Median",
    "Smoothed: 50 µm: Eosin: Nucleus: Std.Dev.",
    "Smoothed: 50 µm: Eosin: Cytoplasm: Mean",
    "Smoothed: 50 µm: Eosin: Cytoplasm: Median",
    "Smoothed: 50 µm: Eosin: Cytoplasm: Std.Dev.",
    "Smoothed: 50 µm: Eosin: Cell: Mean",
    "Smoothed: 50 µm: Eosin: Cell: Median",
    "Smoothed: 50 µm: Eosin: Cell: Std.Dev.",
    "Smoothed: 50 µm: Nearby detection counts",
    "ROI: 0.44 µm per pixel: OD Sum: Mean",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Mean",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Std.dev.",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Angular second moment (F0)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Contrast (F1)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Correlation (F2)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum of squares (F3)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Inverse difference moment (F4)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum average (F5)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum variance (F6)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Sum entropy (F7)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Entropy (F8)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Difference variance (F9)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Difference entropy (F10)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Information measure of correlation 1 (F11)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: OD Sum: Haralick Information measure of correlation 2 (F12)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Mean",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Std.dev.",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Angular second moment (F0)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Contrast (F1)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Correlation (F2)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum of squares (F3)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Inverse difference moment (F4)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum average (F5)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum variance (F6)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Sum entropy (F7)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Entropy (F8)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Difference variance (F9)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Difference entropy (F10)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Information measure of correlation 1 (F11)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Hematoxylin: Haralick Information measure of correlation 2 (F12)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Mean",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Std.dev.",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Angular second moment (F0)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Contrast (F1)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Correlation (F2)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum of squares (F3)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Inverse difference moment (F4)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum average (F5)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum variance (F6)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Sum entropy (F7)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Entropy (F8)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Difference variance (F9)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Difference entropy (F10)",
    "Circle: Diameter 50.0 µm: 0.44 µm per pixel: Eosin: Haralick Information measure of correlation 1 (F11)",
]

base_path = Path(
    r"E:\Meso_TCGA\Slides_tiled\TCGA-SC-A6LN-01Z-00-DX1.379BF588-5A65-4BF8-84CF-5136085D8A47"
)

"""    
df=pd.read_csv(base_path.joinpath('dets.csv'),sep='\t')
df['label']={'Epithelioid': 'E', 'Biphasic': 'B', 'sarcomatoid': 'S'}[lab_df.loc[base_path.name+'.svs'].labels]
#set column denoting if tumor or not
df=label_dets_from_heatmap(Path(f'D:\TCGA_Data\Temp\mapsTCGA\{base_path.name}.svs'),df,0.4)
dfs=split_det(df,'core',0)
dfs=[dfs[key] for key in dfs.keys() if key!='Image']
"""

# dataset, slide, Y, _ = mk_graph(dfs[0:], mode='WSI', to_use=to_use)
# dataset, slide, Y, used_feats=mk_graph(to_use=to_use)
dataset, slide, Y, used_feats = mk_graphs(dataset="heba", load_graphs="heba")
feat_names = used_feats
make_charts = True

short_names = []
for f in used_feats:
    if "Smoothed" in f:
        short_names.append(": ".join(["Smoothed"] + f.split(": ")[2:]))
    elif "Circle: Diameter" in f:
        if "Haralick" in f:
            short_names.append(": ".join(["Haralick"] + [f.split(": ")[3]] + [f[-5:]]))
        else:
            app = f[-5:]
            if app == ".dev.":
                app = "Std.dev."
            short_names.append(": ".join([f.split(": ")[3]] + [app]))
    elif "ROI" in f:
        short_names.append(": ".join(["ROI"] + [f.split(": ")[2]] + [f[-5:]]))
    else:
        short_names.append(f)
    if len(short_names[-1]) > 25:
        split_n = short_names[-1].split(": ")
        short_names[-1] = (
            ": ".join(split_n[0 : math.ceil(len(split_n) / 2)])
            + "\n"
            + ": ".join(split_n[math.ceil(len(split_n) / 2) :])
        )


used_feats = short_names
# model_path = r'D:\Results\TMA_results\test_run23\\model_fold_0_r2.pt'
model_path = r"D:\Results\heba_results\test_run8\\model_fold_0_r0.pt"
model = torch.load(model_path)

if make_charts:
    feat_imp = []

    explainer = GNNExplainer(
        model, epochs=200, num_hops=4, return_type="raw", feat_mask_type="feature"
    )
    node_idx = 10
    for core in dataset:
        # core=dataset[5]
        x = core.x
        edge_index = core.edge_index
        # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
        node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)
        feat_imp.append(node_feat_mask.cpu())

    Y = np.array(Y)
    # Y=np.tile(Y,4)
    feat_imps = np.array(np.vstack(feat_imp))

    p1 = make_chart(feat_imps, used_feats, "All")
    p2 = make_chart(feat_imps[Y == 0, :], used_feats, "False")
    p3 = make_chart(feat_imps[Y == 1, :], used_feats, "True")
    # p4 = make_chart(feat_imps[Y==2,:],used_feats,'Sarcomatoid')
    # show the plots in a row
    l = row(p1, p2, p3)  # ,p4)
    show(l)  # ,p4))
    export_svg(l, filename=Path(model_path).parent / "feat_imp.svg")


# per node feature importances
explainer_output = {}
for core in dataset:
    explainer_output[core.core] = {}
    topN, topN_score, feat_scores = explain_net(core, type="feature")
    # feat_df=pd.DataFrame(node_imp, columns=['imp'])
    # top_feats,top_scores=[],[]
    # for k in range(topN.shape[0]):
    # top_str='\r\n'.join(feat_names[topN[k,:]])
    # top_feats.append(top_str)
    # top_feats.append(np.array(used_feats)[topN[k,:]])
    # top_scores.append(topN_score[k,:])

    explainer_output[core.core]["top_feats"] = [short_names[i] for i in topN]
    explainer_output[core.core]["top_scores"] = topN_score
    explainer_output[core.core]["feat_scores"] = feat_scores
    # explainer_output[core.core]['node_imp']=node_imp
    # feat_df['top_feats']=top_feats
    # feat_df['top_scores']=top_scores
    # sorted_feats=['imp','top_feats']

"""
explainer = GNNExplainer(model, epochs=200, num_hops=5, return_type='raw', feat_mask_type='scalar')
#node importances
#node_imp = {}
for core in dataset:
    #core=dataset[5]
    x = core.x
    edge_index= core.edge_index
    #node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)
    explainer_output[core.core]['node_imp'] = node_feat_mask.cpu()
"""

# save here...
with open(Path(model_path).parent / "explainer_output.pkl", "wb") as f:
    pickle.dump(explainer_output, f)

vis = False
if vis:
    im = plt.imread(f"D:\All_cores\{core.core}.jpg")
    plt.imshow(im)
    ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, threshold=0.5)
    plt.show()
