from sklearn.model_selection import StratifiedKFold, train_test_split
from mk_graph import mk_graphs, slide_fold
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
import torch
import numpy as np
from meso_models import MesoBranched, MesoSep
from meso_engine import NetWrapper
import pandas as pd
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CyclicLR
from utils import add_missranked

"""Code for setting up, running and saving outputs from experiments 
with MesoGraph models on specificed dataset(s).
"""


class StratifiedSampler(Sampler):
    """Stratified Sampling
    return a stratified batch
    """

    def __init__(self, class_vector, batch_size=10):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        """
        self.batch_size = batch_size
        self.n_splits = int(class_vector.size(0) / self.batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        YY = self.class_vector.numpy()
        idx = np.arange(len(YY))
        return [
            tidx for _, tidx in skf.split(idx, YY)
        ]  # return array of arrays of indices in each batch

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


if __name__ == "__main__":
    # parameters
    learning_rate = 0.00005
    weight_decay = 0.02
    epochs = 300
    scheduler = "cyclic"
    opt = "adam"  # sgd or adam
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    save_res = "core"  #'core', 'node' or False
    # core will save only core-level predictions, node will additionally save node-level predictions.
    split_strat = "slide_fold"  # slide_fold or cross_val
    model_type = "branched"  #'branched' or 'separate'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = Path(r"D:\Results\test_run") #where to save results
    base_path.mkdir(exist_ok=True)
    load_graphs = None # Path(r"D:\QuPath_Projects\Meso_TMA\detections\graphs") # or None to generate graphs
    use_res = True #no effect if loading graphs
    # dataset list: if two given, will train on first and test on second.
    # otherwise, will use split_strat to split dataset into train/test
    dataset_list = ["meso"]
    dim_target = 2
    layers = [20, 10, 10]
    dropout = 0
    do_ls = True
    notes = ""   # notes to add to info file
    info_str = f"""folder={base_path.name},lr={learning_rate},wd={weight_decay},epochs={epochs},scheduler={scheduler},split_strat={split_strat},load_graphs={load_graphs},
        model_type={model_type},dataset_list={dataset_list},dim_target={dim_target},layers={layers},dropout={dropout},do_ls={do_ls},opt={opt},use_res={use_res},notes={notes}"""
    split_dataset = False

    if len(dataset_list) == 1:
        split_dataset = True
        dataset, slide, Y, _ = mk_graphs(
            dataset_list[0], load_graphs=load_graphs, use_res=use_res
        )
    else:
        dataset, slide, Y, _ = mk_graphs(
            dataset_list[0], load_graphs=load_graphs, use_res=use_res
        )
        test_dataset, slide_t, Y_t, _ = mk_graphs(
            dataset_list[1], load_graphs=load_graphs, use_res=use_res
        )
        tr_test_split = [(1, 1)]  # dummy split, as separate test dataset given
    print("made graphs, starting training..")

    va, ta = [], []
    nreps = 1
    for reps in range(nreps):
        Vacc, Tacc = [], []
        dfs = []
        m = 0
        if split_dataset:
            if split_strat == "slide_fold":
                tr_test_split = slide_fold(slide)
            else:
                tr_test_split = skf.split(dataset, Y)
        for tr, te in tr_test_split:
            if split_dataset:
                test_dataset = [dataset[i] for i in te]
            tt_loader = DataLoader(test_dataset, shuffle=True)

            if not split_dataset:
                tr = range(len(dataset))
            train, valid = train_test_split(
                tr, test_size=0.25, shuffle=True, stratify=np.array(Y)[tr]
            )
            sampler = StratifiedSampler(
                class_vector=torch.from_numpy(np.array(Y)[train]).cpu(), batch_size=16
            )

            train_dataset = [dataset[i] for i in train]
            valid_dataset = [dataset[i] for i in valid]

            v_loader = DataLoader(valid_dataset, shuffle=True)
            tr_loader = DataLoader(train_dataset, batch_sampler=sampler)

            if model_type == "branched":
                model = MesoBranched(
                    dim_features=train_dataset[0].x.shape[1],
                    dim_target=dim_target,
                    layers=layers,
                    dropout=dropout,
                    pooling="mean",
                    eps=100.0,
                    train_eps=False,
                    do_ls=do_ls,
                )
            elif model_type == "separate":
                model = MesoSep(
                    dim_features=train_dataset[0].x.shape[1],
                    dim_target=dim_target,
                    layers=layers,
                    dropout=dropout,
                    pooling="mean",
                    eps=100.0,
                    train_eps=False,
                    do_ls=do_ls,
                )
            else:
                raise ValueError("model_type must be branched or separate")
            net = NetWrapper(
                model, loss_function=None, device=device, save_dir=base_path
            )
            model = model.to(device=net.device)
            if opt == "adam":
                optimizer = optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                # scheduler = OneCycleLR(optimizer,max_lr=learning_rate, steps_per_epoch=len(tr_loader), epochs=epochs, pct_start=0.25, div_factor=20, final_div_factor=20)
                scheduler = CyclicLR(
                    optimizer,
                    learning_rate,
                    5 * learning_rate,
                    40 * len(tr_loader),
                    mode="exp_range",
                    gamma=0.8,
                    cycle_momentum=False,
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    momentum=0.7,
                    nesterov=True,
                )
                scheduler = CyclicLR(
                    optimizer,
                    learning_rate,
                    5 * learning_rate,
                    40 * len(tr_loader),
                    mode="exp_range",
                    gamma=0.8,
                    cycle_momentum=True,
                )

            (
                best_model,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                tt_loss,
                tt_acc,
            ) = net.train(
                train_loader=tr_loader,
                max_epochs=epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                clipping=None,
                validation_loader=v_loader,
                test_loader=tt_loader,
                early_stopping=None,
                log_every=50,
            )

            Vacc.append(val_acc)
            Tacc.append(tt_acc)
            print("fold complete", len(Vacc), train_acc, val_acc, tt_acc)
            torch.save(best_model, base_path / f"model_fold_{m}_r{reps}.pt")
            m += 1
            if save_res:
                # get preds
                (Path(net.save_dir) / "node_preds").mkdir(exist_ok=True)
                # gets node preds in G and core preds in df
                G, df = net.predict(test_dataset, best_model)
                df["fold"] = m
                df = add_missranked(df)
                dfs.append(df)
            if save_res == "node":
                # save node preds
                for key in G:
                    net.save_preds(G[key], include_feats=False)

        if save_res:
            # save core preds
            pred_df = pd.concat(dfs, axis=0, ignore_index=True)
            pred_df.to_csv(base_path / f"GNN_class_temp_dual_r{reps}.csv")
        print("avg Valid acc=", np.mean(Vacc), "+/", np.std(Vacc))
        print("avg Test acc=", np.mean(Tacc), "+/", np.std(Tacc))
        va.append(np.mean(Vacc))
        ta.append(np.mean(Tacc))
    # add accuracies to info_str
    info_str += f",valid auc={np.mean(va)}+/-{np.std(va)}"
    info_str += f",test auc={np.mean(ta)}+/-{np.std(ta)}"
    # save info_str to file
    with open(base_path / "info.txt", "w") as f:
        f.write(info_str)
    print(f"val accs were: {va}")
    print(f"test accs were: {ta}")
    print(info_str)
