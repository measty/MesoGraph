from pathlib import Path
import pandas as pd
import numpy as np
from tiatoolbox.annotation.storage import SQLiteStore, Annotation
import pickle
from utils import get_short_names


"""Prepare an annotation store and associated graph .plk file for visualization.
Includes features and GNN output scores in the annotation properties,
and the feature importances per node in the graph.
"""

store_path = Path(r"D:\QuPath_Projects\Meso_TMA\detections\stores")
res_path = Path(r"D:\Results\TMA_results\test_run22\node_preds")
vis_path = Path(r"D:\Results\TMA_results\test_run22\vis")
vis_path.mkdir(exist_ok=True)
graph_path = Path(r"D:\QuPath_Projects\Meso_TMA\detections\graphs_min_r")
vis_path.mkdir(exist_ok=True)
with open(r"D:\Results\TMA_results\test_run22\explainer_output.pkl", "rb") as f:
    explainer_output = pickle.load(f)

core_list = list(store_path.glob("*.db"))
# get to_use feat list from a graph
graphs = list(graph_path.glob("*.pkl"))
with open(graphs[0], "rb") as f:
    g = pickle.load(f)
to_use_base = get_short_names(g.feat_names)
for i in range(512):
    to_use_base.remove(f"res{i}")

for core in core_list:
    print(f"Processing core {core.stem}")
    # Load the scores per node
    node_scores = pd.read_csv(res_path / (f"GNN_scores_{core.stem}.csv"))
    # drop last row as its virtual core node
    node_scores.drop(node_scores.tail(1).index, inplace=True)
    node_scores["score"] = (
        1 + node_scores["score_S"].values - node_scores["score_E"].values
    ) / 2
    # create graph dict
    with open(graph_path / (f'{core.stem.split("_")[0]}.pkl'), "rb") as f:
        g = pickle.load(f)
    graph_dict = {
        "edge_index": g.edge_index.cpu().numpy()
    }  # , 'feat_names': explainer_output['top_feats'], 'feat_importances': explainer_output['top_scores']}
    graph_dict["score"] = node_scores["score"].values
    graph_dict["top_feats"] = explainer_output[core.stem.split("_")[0]]["top_feats"]
    graph_dict["top_scores"] = explainer_output[core.stem.split("_")[0]]["top_scores"]
    # add the top feats if they arent included (ie if a res feat)
    to_use = set(to_use_base + graph_dict["top_feats"])
    # Load the annotation store
    SQ = SQLiteStore(core)
    # props = [ann.properties for ann in SQ.values()]
    # df = pd.DataFrame(props)
    df = pd.DataFrame(g.x.cpu().numpy(), columns=get_short_names(g.feat_names))
    # rename dataframe columns to corresponding short names
    # short_names = get_short_names(df.columns)
    # df.rename(dict(zip(df.columns, short_names)), axis=1, inplace=True)
    df = df[to_use]
    # normalize all properties so 5th and 95th percentile between 0 and 1
    df = df.apply(
        lambda x: (x - x.quantile(0.05)) / (x.quantile(0.95) - x.quantile(0.05))
    )
    # Add the node scores to the annotation properties
    df = pd.concat([df, node_scores[["score_E", "score_S", "score"]]], axis=1)
    # df['node_exp'] = np.nan_to_num(explainer_output[core.stem.split('_')[0]]['node_imp'][:-1])
    df.dropna(inplace=True, axis=0)
    # make new store with the new properties
    SQ2 = SQLiteStore(vis_path / (f"{core.stem}.db"))
    annotations = []
    coords = []
    for i, ann in enumerate(SQ.values()):
        coords.append([ann.geometry.centroid.x, ann.geometry.centroid.y])
        annotations.append(Annotation(ann.geometry, df.loc[i].to_dict()))
    SQ2.append_many(annotations)
    coords = np.array(coords)
    graph_dict["coordinates"] = coords
    with open(vis_path / (f"{core.stem}.pkl"), "wb") as f:
        pickle.dump(graph_dict, f)
    # Save the annotation store
    # SQ2.dump(vis_path/(core.stem+'.db'))
    print(f"Done core {core.stem}")
