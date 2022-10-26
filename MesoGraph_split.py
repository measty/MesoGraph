# -*- coding: utf-8 -*-
"""
Multiple Instance Graph Classification
@author: fayyaz
"""

from bokeh.core.enums import Palette
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from mk_graph_old import mk_graph, slide_fold
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CyclicLR
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from copy import deepcopy
from numpy.random import randn  # importing randn
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (
    GINConv,
    EdgeConv,
    DynamicEdgeConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    PANConv,
    PANPooling,
    global_sort_pool,
)
import time
from tqdm import tqdm
from scipy.spatial import distance_matrix, Delaunay
import random
from torch_geometric.data import Data, DataLoader
from PIL import Image
from pathlib import Path
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import layout
from bokeh.models import Slider, Toggle, Dropdown, Div, ColumnDataSource
from bokeh.models.callbacks import CustomJS

# from wasabi import change_pixel
from bokeh.models.mappers import EqHistColorMapper
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlGn11
from bokeh.embed import file_html
from bokeh.resources import CDN
import pandas as pd
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_regression, f_regression
from bokeh.models.tools import TapTool
from torch_geometric.nn.models import GNNExplainer

output_file(
    filename="D:/Meso/TMA_vis_temp.html", title="TMA cores graph NN visualisation"
)
sys.setrecursionlimit(10**4)

USE_CUDA = torch.cuda.is_available()
device = {True: "cuda", False: "cpu"}[USE_CUDA]


def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v


def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))


def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()


#%% Graph Fitting
class GraphFit:
    """
    A Pytorch implementation of "Fittng a graph to vector data"
    @inproceedings{Daitch:2009:FGV:1553374.1553400,
     author = {Daitch, Samuel I. and Kelner, Jonathan A. and Spielman, Daniel A.},
     title = {Fitting a Graph to Vector Data},
     booktitle = {Proceedings of the 26th Annual International Conference on Machine Learning},
     series = {ICML '09},
     year = {2009},
     isbn = {978-1-60558-516-1},
     location = {Montreal, Quebec, Canada},
     pages = {201--208},
     numpages = {8},
     url = {http://doi.acm.org/10.1145/1553374.1553400},
     doi = {10.1145/1553374.1553400},
     acmid = {1553400},
     publisher = {ACM},
     address = {New York, NY, USA},
    }

    Solves: min_w \sum_i {d_i x_i - \sum_j w_{i,j}x_j}
    such that:
        \sum_i max{0,1-d_i}^2 \le \alpha n

    wher d_i = \sum_jW_{i,j}
    """

    def __init__(self, n, d):
        self.n, self.d = n, d
        self.W = cuda(Variable(torch.rand((n, n)).float())).requires_grad_()
        self.history = []

    def fit(self, X, lr=1e-2, epochs=500):
        X = toTensor(X, requires_grad=False)
        self.X = X
        optimizer = optim.Adam([self.W], lr=lr)
        alpha = 1.0
        zero = toTensor([0])
        for epochs in range(epochs):
            L, D = self.getLaplacianDegree()
            loss = torch.norm(L @ X) + alpha * torch.sum(torch.max(zero, 1 - D) ** 2)

            self.history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self

    def getLaplacianDegree(self):
        W = self.getW()
        L = -W
        D = torch.sum(W, dim=0)
        L.diagonal(dim1=-2, dim2=-1).copy_(D)
        return L, D

    def getW(self):
        """
        Gets adjacency matrix for the graph
        """
        Z = (torch.transpose(self.W, 0, 1) + self.W) ** 2
        Z.fill_diagonal_(0)
        return Z


def toGeometric(Gb, y, tt=1e-3):
    """
    Create pytorch geometric object based on GraphFit Object
    """
    return Data(x=Gb.X, edge_index=(Gb.getW() > tt).nonzero().t().contiguous(), y=y)


def toGeometricWW(X, W, y, tt=0):
    return Data(
        x=toTensor(X, requires_grad=False),
        edge_index=(toTensor(W, requires_grad=False) > tt).nonzero().t().contiguous(),
        y=toTensor([y], dtype=torch.long, requires_grad=False),
    )


#%% GIN Model Implementation

from torch.utils.data import Sampler


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
        # #Binary sampling: Returns a pair index of positive and negative index-All samples from majority class are paired with repeated samples from minority class
        # U, C = np.unique(YY, return_counts=True)
        # M = U[np.argmax(C)]
        # Midx = np.nonzero(YY==M)[0]
        # midx = np.nonzero(YY!=M)[0]
        # midx_ = np.random.choice(midx,size=len(Midx))
        # if M>0: #if majority is positive
        #     return np.vstack((Midx,midx_)).T
        # else:
        #     return np.vstack((midx_,Midx)).T

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)


def calc_roc_auc(target, prediction):
    # import pdb;pdb.set_trace()
    # return np.mean(toNumpy(target)==(toNumpy(prediction[:,1]-prediction[:,0])>0))

    return roc_auc_score(toNumpy(target), toNumpy(prediction[:, -1]))
    # output = F.softmax(prediction, dim=1)
    # output = output.detach()[:, 1]
    # fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy())
    # roc_auc = auc(fpr, tpr)
    # import pdb;pdb.set_trace()
    # return roc_auc


def add_missranked(df):
    df.set_index("core")
    miss = []
    for core in df.index:
        lab, s = df.loc[core, "label"], df.loc[core, "GNN"]
        miss.append(
            (
                df[
                    np.logical_and(df["label"].values < lab, df["GNN"].values > s)
                ].shape[0]
                + df[
                    np.logical_and(df["label"].values > lab, df["GNN"].values < s)
                ].shape[0]
            )
            / sum(df["label"].values != lab)
        )
    df["miss"] = miss
    return df


class alpha_scaler:
    def __init__(self, alpha, step_size) -> None:
        self.alpha = alpha
        self.step_size = step_size

    def update_alpha(self):
        if self.alpha + self.step_size < 1:
            self.alpha += self.step_size


class learnable_sig(torch.nn.Module):
    def __init__(self, fsize) -> None:
        super(learnable_sig, self).__init__()
        # self.l1=nn.Linear(fsize,1)
        self.l1 = Sequential(Linear(fsize, fsize), ReLU(), Linear(fsize, 2))
        self.l2 = Sequential(Linear(fsize, fsize), ReLU(), Linear(fsize, 2))
        # self.first_h=PANConv(dim_features, out_emb_dim,5)
        self.alpha = nn.parameter.Parameter(2 * torch.ones(1, 2))
        self.beta = nn.parameter.Parameter(torch.zeros(1, 2))
        self.gamma = nn.parameter.Parameter(torch.zeros(1, 2))

    def forward(self, x, xcore, batch):
        y = []
        for i in torch.unique(batch):
            # last_ind=torch.sum(batch<=i)-1
            # y.append(torch.sigmoid(x[batch==i,:]*self.alpha-self.beta+self.gamma*torch.mean(x[batch==i],dim=0,keepdim=True)))
            # y.append(torch.sigmoid(x[batch==i,:]*self.alpha-self.beta))
            # y.append(torch.sigmoid(x[batch==i,:]*self.alpha-self.l1(xcore.T[:,i])))
            y.append(
                torch.sigmoid(
                    x[batch == i, :] * (2 + self.l2(xcore.T[:, i]))
                    - self.l1(xcore.T[:, i])
                )
            )
        return torch.cat(y)


class MesoBranched(torch.nn.Module):
    def __init__(
        self,
        dim_features,
        dim_target,
        layers=[16],
        pooling="max",
        dropout=0.0,
        eps=0.0,
        train_eps=False,
        do_ls=False,
    ):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.do_ls = do_ls
        if do_ls:
            self.ls = learnable_sig(dim_features)
            # self.ls=learnable_sig(np.sum(layers))   #if on embed feats
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
            "PAN": PANPooling(in_channels=1, ratio=0.5),
            "topN": global_sort_pool,
        }[pooling]
        self.ecnns = []
        self.ecs = []
        self.last = None
        if dim_target > 2:
            self.last = nn.Sequential(ReLU(), Linear(dim_target, 2))
        # train_eps = True#config['train_eps']

        # TOTAL NUMBER OF PARAMETERS #

        # first: dim_features*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target
        # l-th: input_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target

        # -------------------------- #

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )
                # self.first_h=PANConv(dim_features, out_emb_dim,5)
                # self.linears.append(Linear(dim_features, dim_target))
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                # self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                #                       Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                # self.convs.append(GINConv(self.nns[-1], eps=eps, train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

                subnet = Sequential(
                    Linear(2 * input_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )

                self.ecnns.append(subnet)

                self.ecs.append(
                    EdgeConv(self.ecnns[-1], aggr="mean")
                )  # DynamicEdgeConv#EdgeConv
                # self.ecs.append(GINConv(self.ecnns[-1]))
                # self.ecs.append(DynamicEdgeConv(self.ecnns[-1],k=10,aggr='mean',num_workers=8))

        # self.first_h = torch.nn.ModuleList(self.first_h)
        # self.nns = torch.nn.ModuleList(self.nns)
        # self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(
            self.linears
        )  # has got one more for initial input

        self.ecnns = torch.nn.ModuleList(self.ecnns)
        self.ecs = torch.nn.ModuleList(self.ecs)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        if edge_index == None:
            xfeat, edge_index, batch = x.x, x.edge_index, x.batch
            explaining = False
        else:
            xfeat = x
            batch = torch.zeros((xfeat.shape[0],), dtype=torch.long, device="cuda")
            explaining = True

        out = 0
        pooling = self.pooling
        Z = 0

        last_ind = []
        for i in torch.unique(batch):
            last_ind.append(torch.sum(batch <= i) - 1)

        core_x = []
        for layer in range(self.no_layers):
            # print(f'Forward: layer {l}')
            if layer == 0:
                # x, M = self.first_h(x, edge_index)
                x = self.first_h(xfeat)
                if self.do_ls:
                    core_x.append(x[last_ind, :])
                z = F.dropout(
                    self.linears[layer](x), p=self.dropout, training=self.training
                )
                Z += z
                # dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                # dout = F.dropout(torch.mean(pooling(z, batch, 1000),dim=1,keepdim=True), p=self.dropout, training=self.training)
                # dout=pooling(z, M, batch)
                # dout=global_mean_pool(dout[0],dout[3])
                # out += dout
            else:
                # Layer l ("convolution" layer)
                # import pdb;pdb.set_trace()
                # x = self.convs[layer-1](x, edge_index)
                x = self.ecs[layer - 1](x, edge_index)
                if self.do_ls:
                    core_x.append(x[last_ind, :])
                # x = self.ecs[layer-1](x,batch)
                z = F.dropout(
                    self.linears[layer](x), p=self.dropout, training=self.training
                )
                Z += z
                # dout=pooling(z, M, batch)
                # dout = F.dropout(torch.mean(pooling(z, batch, 1000),dim=1,keepdim=True), p=self.dropout, training=self.training)
                # dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)#F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                # out += dout

        if self.last:
            Z = self.last(Z)
        if self.do_ls:
            core_x = torch.cat(core_x, dim=1)
            # ZZ=self.ls(Z,core_x.detach(),batch) #if want sig thresh learnt on embed feats but not backprop to edgeconv weights etc
            # ZZ=self.ls(Z,core_x,batch)
            ZZ = self.ls(Z, xfeat[last_ind, :], batch)
            out = pooling(ZZ, batch)
            if explaining:
                return out
            else:
                return out, ZZ
        else:
            out = pooling(Z, batch)
            if explaining:
                return out
            else:
                return out, Z


class NetWrapper:
    def __init__(self, model, loss_function, device="cpu", classification=True):
        self.model = model
        self.scaler = alpha_scaler(0, 0.005)
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification

    def _pair_train(self, train_loader, optimizer, scheduler, clipping=None):
        """
        Performs pairwise comparisons with ranking loss
        """
        model = self.model.to(self.device)
        model.train()
        loss_all = 0
        acc_all = 0
        assert self.classification
        # lossfun = nn.MarginRankingLoss(margin=1.0,reduction='sum')
        for data in train_loader:

            data = data.to(self.device)

            optimizer.zero_grad()
            output, xx = model(data)
            # import pdb; pdb.set_trace()
            # Can add contrastive loss if reqd
            # import pdb; pdb.set_trace()
            y = data.y
            loss = 0
            c = 0
            # z = Variable(torch.from_numpy(np.array(0))).type(torch.FloatTensor)
            z = toTensor([0])
            for i in range(len(y) - 1):
                for j in range(i + 1, len(y)):
                    if y[i] != y[j]:
                        c += 1
                        dz = output[i, :] - output[j, :]
                        dy = torch.stack([y[j] - y[i], y[i] - y[j]])
                        loss += torch.mean(torch.max(z, 1.0 - dy * dz))  # 1.0 or 0.5?
                        # loss+=lossfun(zi,zj,dy)
            loss = loss / c
            # extra loss component to penalise cells being both ep and sarc, also may
            # act as a regularisation
            # loss_reg=torch.mean(xx)
            # loss_es=torch.mean(torch.prod(xx,dim=1)**2)
            # loss_es=torch.mean(torch.max(toTensor(0.0).to(device),torch.prod(xx+toTensor(-0.1).to(device),dim=1))**2)
            # loss_es=torch.mean(torch.max(toTensor(0.0).to(device),torch.prod(xx+toTensor(-0.1).to(device),dim=1)))**2
            # loss=loss+0.5*self.scaler.alpha*loss_es#+0.1*loss_reg
            # loss=loss+0.5*self.scaler.alpha*loss_reg

            acc = loss
            loss.backward()

            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss.item() * num_graphs
            acc_all += acc.item() * num_graphs

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
            scheduler.step()

        return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)

    def _train(self, train_loader, optimizer, clipping=None):
        """
        Original training method
        """
        model = self.model.to(self.device)

        model.train()

        loss_all = 0
        acc_all = 0
        for data in train_loader:

            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            if self.classification:
                loss, acc = self.loss_fun(data.y, *output)
                loss.backward()

                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    num_graphs = data.adj.size(0)

                loss_all += loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
            else:
                loss = self.loss_fun(data.y, *output)
                loss.backward()
                loss_all += loss.item()

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()

        if self.classification:
            return acc_all / len(train_loader.dataset), loss_all / len(
                train_loader.dataset
            )
        else:
            return None, loss_all / len(train_loader.dataset)

    def classify_graphs(self, loader):
        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                data = data.to(self.device)
                output, xx = model(data)
                if i == 0:
                    Z = output
                    y = data.y
                else:
                    Z = torch.cat((Z, output))
                    y = torch.cat((y, data.y))

            loss = 0
            c = 0
            # z = Variable(torch.from_numpy(np.array(0))).type(torch.FloatTensor)
            z = toTensor([0])
            for i in range(len(y) - 1):
                for j in range(i + 1, len(y)):
                    if y[i] != y[j]:
                        c += 1
                        dz = Z[i, :] - Z[j, :]
                        dy = torch.stack([y[j] - y[i], y[i] - y[j]])
                        loss += torch.mean(torch.max(z, 1.0 - dy * dz))
                        # loss+=lossfun(zi,zj,dy)
            loss = loss.item() / c

            # if not isinstance(Z, tuple):
            #    Z = (Z,)
            # loss, acc = self.loss_fun(Y, *Z)
            # loss = 0
            auc_val = calc_roc_auc(
                torch.minimum(y, torch.ones(1, dtype=torch.int64).to(self.device)),
                torch.unsqueeze(Z[:, -1] - Z[:, 0], 1),
            )  # torch.unsqueeze(Z[:,-1]/(Z[:,0]+0.00001),1))
            # pr = calc_pr(Y, *Z)
            return auc_val, loss  # , auc, pr

    def train(
        self,
        train_loader,
        max_epochs=100,
        optimizer=torch.optim.Adam,
        scheduler=None,
        clipping=None,
        validation_loader=None,
        test_loader=None,
        early_stopping=None,
        log_every=1000,
    ):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_loss, val_acc = -1, -1
        test_loss, test_acc = None, None

        time_per_epoch = []
        self.history = []

        best_val_acc = -1
        return_best = True
        test_acc_at_best_val_acc = -1
        for epoch in tqdm(range(1, max_epochs + 1)):

            # if scheduler is not None:
            # scheduler.step(epoch)
            start = time.time()

            train_acc, train_loss = self._pair_train(
                train_loader, optimizer, scheduler, clipping
            )

            end = time.time() - start
            time_per_epoch.append(end)

            if test_loader is not None:
                test_acc, test_loss = self.classify_graphs(test_loader)

            if validation_loader is not None:
                val_acc, val_loss = self.classify_graphs(validation_loader)

            if epoch % log_every == 0:
                msg = (
                    f"Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} "
                    f"TE loss: {test_loss} TE acc: {test_acc}"
                )
                print("\n" + msg)

            self.history.append(train_loss)
            self.scaler.update_alpha()

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val_acc = test_acc
                best_model = deepcopy(self.model)
        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc
        else:
            best_model = deepcopy(self.model)

        if early_stopper is not None:
            (
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                test_loss,
                test_acc,
                best_epoch,
            ) = early_stopper.get_best_vl_metrics()

        return best_model, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


#%% Dummy toy data generation
def getExamples(n=800, d=2):
    """
    Generates n d-dimensional normally distributed examples of each class
    The mean of the positive class is [1] and for the negative class it is [-1]
    """
    Xp1 = randn(int(n / 2), d) + 3.0  # +1   #generate examples of the positie class
    Xp2 = randn(int(n / 2), d) - 9.0  # +1   #generate examples of the positie class
    Xp = np.vstack((Xp1, Xp2))

    Xn1 = randn(int(n / 2), d) - np.array(
        [3, 3]
    )  # -1   #generate n examples of the negative class
    Xn2 = randn(int(n / 2), d) + 7.0
    Xn = np.vstack((Xn1, Xn2))

    X = np.vstack((Xp, Xn))  # Stack the examples together to a single matrix
    y = np.array([+1] * Xp.shape[0] + [-1] * Xn.shape[0])  # Associate Labels

    Noise = randn(n, d) + [-3, 5]  # generate noise
    Noise = np.vstack((Noise, randn(n, d) + [-3, -11]))
    X = np.vstack((X, Noise))  # add noise
    y = np.append(y, [0] * len(Noise))

    X += 2
    #    y = -y
    ridx = np.random.permutation(range(len(y)))
    X, y = X[ridx, :], y[ridx]
    return X, y


def genBags(y):
    """
    Add examples to bags
        Positive bag: has at least one positive example
        mexpb: maximum number of examples per bag
        mprop: proportion of majority class in a bag
        nprop: proportion of noise class in a bag

    """
    pid, nid, noise = (
        list(np.where(y == 1)[0]),
        list(np.where(y == -1)[0]),
        list(np.where(y == 0)[0]),
    )

    Nbags = 30  # number of bags
    mepb = 30  # max number of example per bag
    mprop = 0.05  # majority proportion
    nprop = 0.00  # noise proportion per bag
    Bsize = np.random.binomial(n=mepb, p=0.5, size=Nbags)
    print("Avg. Examples/Bag:", np.mean(Bsize))
    Bidx = []
    Y = np.array(
        [-1] * int(Nbags / 2) + [1] * int(Nbags / 2)
    )  # np.random.choice([-1, 1], size=(Nbags,), p=[0.5, 0.5])
    for i in range(len(Y)):
        M = int(np.ceil(Bsize[i] * mprop))
        n = int(Bsize[i] * nprop)
        m = Bsize[i] - M - n
        if Y[i] == 1:
            B = pid[:M]
            pid = pid[M:]  # add M examples from the positive class
            #            print("Pos",len(B))
            B += nid[:m]
            nid = nid[m:]  # add m examples from the negative class
        else:
            B = nid[:M]
            nid = nid[M:]  # add M+m examples from negative class
            B += nid[:m]
            nid = nid[m:]

        B += noise[:n]
        noise = noise[n:]  # add n examples of noise

        Bidx.append(np.array(B))

    return Bidx, Y


#%% VISUALIZATION
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx

# import umap
# from skimage import exposure
from sklearn.decomposition import PCA


def sort_feats(xdf, feat_names, stat=mutual_info_regression):
    feat_sel = SelectKBest(stat, k=20)
    feat_sel.fit(xdf[feat_names].values, xdf["score"].values)
    f_scores = feat_sel.scores_
    inds = np.argsort(f_scores)
    sort_feats = np.flip(feat_names[inds])
    return list(sort_feats)


def explain_net(core, node_idx=None, N=10):
    explainer = GNNExplainer(
        model, epochs=200, return_type="raw", feat_mask_type="individual_feature"
    )
    # core=dataset[5]
    x = core.x
    edge_index = core.edge_index
    # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)
    topN = np.fliplr(np.argsort(node_feat_mask.to("cpu"), axis=1))[:, :N]
    topN_score = np.vstack(
        [node_feat_mask.cpu().numpy()[i, topN[i, :]] for i in range(topN.shape[0])]
    )
    node_imp = [
        np.mean(
            edge_mask[np.any((edge_index.T == n).cpu().numpy(), axis=1)].cpu().numpy()
        )
        for n in range(x.shape[0])
    ]

    # im=plt.imread(f'D:\All_cores\{core.core}.jpg')
    # plt.imshow(im)
    # ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, threshold=0.5)
    # plt.show()

    return topN, topN_score, node_imp


def change_pixel(mask, i, j, val):
    if val == 255:
        val = 0.01

    if mask[i, j] == 255:
        mask[i, j] = val

        for a, b in zip([0, 0, 1, -1], [-1, 1, 0, 0]):
            # if mask[i+a,j+b,0]==255:
            # mask[i+a,j+b,:]=val
            if not (
                i + a < 0
                or i + a >= mask.shape[0]
                or j + b < 0
                or j + b >= mask.shape[1]
            ):
                change_pixel(mask, i + a, j + b, val)
    else:
        return


def bokeh_plot(g):

    Xn = toNumpy(g.coords)  # get coordinates
    Wn = np.array([e for e in toNumpy(g.edge_index.t())])
    xdf = pd.DataFrame(toNumpy(g.x), columns=g.feat_names[0])
    c = g.c
    core = g.core[0]
    feat_names = pd.Index(g.feat_names[0])
    print(f"creating vis for core: {core}...")
    tx = f"core {core}, true label is: {g.type_label}, score is: {g.z[1].item()/(g.z[0].item()+0.00001)}"

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("Top Feats", "@top_feats"),
    ]
    p = figure(
        title=tx,
        x_range=(0, 2854),
        y_range=(0, 2854),
        width=900,
        height=900,
        tooltips=TOOLTIPS,
    )
    p.add_tools(TapTool())

    # mask=p.image_url(url=[f'D:\All_masks_inv\{core}.png'],x=0, y=2854, w=2854, h=2854, anchor="top_left", global_alpha=0.35)

    with Image.open(Path(f"D:\All_cores\{core}.jpg")) as imfile:
        blah = np.array(imfile.convert("RGBA"))
        blah_rgb = blah.view(dtype=np.uint32).reshape(blah.shape[:-1])

    blah_rgb = np.flipud(blah_rgb)
    # coreim=p.image_url(url=[f'D:\All_cores\{core}.jpg'],x=0, y=2854, w=2854, h=2854, anchor="top_left")
    coreim = p.image_rgba(image=[blah_rgb], x=0, y=0, dw=2854, dh=2854)

    fmask = Image.open(Path(f"D:\All_masks_tiff\{core}.tiff"))
    im = np.array(fmask, dtype=np.float32)
    im[im == 1] = 256
    im[im == 2] = 0
    im = im - 1
    # im=255-im
    # im_rgba=np.concatenate((np.repeat(im[:,:,None],3,axis=2), 255*np.ones((im.shape[0], im.shape[1],1), dtype=np.uint8)), axis=2)
    # im_rgb=np.repeat(im[:,:,None],3,axis=2)
    # I = Image.fromarray(im_rgb).convert("RGBA")
    pal = [[1 - i / 20, i / 20, 0] for i in range(20)]
    cvals = []
    for i in range(len(Xn)):
        cvals.append(0.5 + (c[i, 0] ** 2 - c[i, 1] ** 2) / 2)
        if c[i, 0] < 0.4 and c[i, 1] < 0.4:
            cval = 101
        else:
            cval = cvals[-1]
        change_pixel(im, int(Xn[i, 1]), int(Xn[i, 0]), cval)

    xdf["score"] = cvals
    xdf["ep_score"] = c[:, 1]
    xdf["s_score"] = c[:, 0]
    xdf["x"] = Xn[:, 0]
    xdf["y"] = 2854 - Xn[:, 1]
    sorted_feats = sort_feats(xdf, feat_names, mutual_info_regression)
    sorted_feats.extend(["ep_score", "s_score"])

    # gnn explainer stuff
    topN, topN_score, node_imp = explain_net(g)
    xdf["imp"] = node_imp
    top_feats, top_scores = [], []
    for k in range(topN.shape[0]):
        # top_str='\r\n'.join(feat_names[topN[k,:]])
        # top_feats.append(top_str)
        top_feats.append(feat_names[topN[k, :]])
        top_scores.append(topN_score[k, :])

    xdf["top_feats"] = top_feats
    xdf["top_scores"] = top_scores
    sorted_feats = ["imp", "top_feats"] + sorted_feats

    im = np.flipud(im)
    fmask.close()
    # cmapper=LinearColorMapper(low=0,high=np.max(im[im<255]), palette='RdYlGn11', low_color=(255,255,255,0))

    highv = np.max(im[im < 100])
    lowv = np.min(im[im >= 0])
    mdiff = np.maximum(highv - 0.5, 0.5 - lowv)
    mdiff = np.maximum(mdiff, 0.4)
    lowv, highv = 0.5 - mdiff, 0.5 + mdiff
    fds = ColumnDataSource(xdf)

    cmapper = LinearColorMapper(
        low=lowv,
        high=highv,
        palette="RdYlGn11",
        high_color=(255, 255, 255, 1),
        low_color=(255, 255, 255, 0),
    )
    drop_ehist_mapper = EqHistColorMapper(
        low=lowv,
        high=highv,
        palette="RdYlGn11",
        high_color=(255, 255, 255, 1),
        low_color=(255, 255, 255, 0),
    )
    drop_lin_mapper = LinearColorMapper(
        low=lowv,
        high=highv,
        palette="RdYlGn11",
        high_color=(255, 255, 255, 1),
        low_color=(255, 255, 255, 0),
    )
    circ_cmapper = linear_cmap(
        field_name="score",
        palette="RdYlGn11",
        low=0,
        high=1,
        high_color=(255, 255, 255, 1),
        low_color=(255, 255, 255, 0),
    )
    # mask=p.image(image=[im],x=0, y=0, dw=2854, dh=2854, global_alpha=0.35, color_mapper=EqHistColorMapper('RdYlGn3'))
    mask = p.image(
        image=[im], x=0, y=0, dw=2854, dh=2854, global_alpha=0.45, color_mapper=cmapper
    )
    # mask=p.image_rgba(image=[np.array(I)],x=0, y=0, dw=2854, dh=2854, global_alpha=0.35)
    edges = p.segment(
        x0=Xn[Wn[:, 0], 0],
        y0=2854 - Xn[Wn[:, 0], 1],
        x1=Xn[Wn[:, 1], 0],
        y1=2854 - Xn[Wn[:, 1], 1],
        line_width=0.6,
    )
    nodes = p.circle(x="x", y="y", color=circ_cmapper, source=fds, radius=3.5)

    menu = list(zip(sorted_feats, sorted_feats))
    drop_i = Dropdown(label="Select Feat (inf)", button_type="warning", menu=menu)
    sorted_feats = sort_feats(xdf, feat_names, f_regression)
    menu = list(zip(sorted_feats, sorted_feats))
    drop_f = Dropdown(label="Select Feat (f1)", button_type="warning", menu=menu)

    s2 = ColumnDataSource(
        data=dict(
            names=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            scores=[0.1, 0.2, 0.3, 0.1, 0.1, 0.24, 0.5, 0.4, 0.3, 0.2],
        )
    )
    p2 = figure(width=700, height=500, x_range=(0, 1), y_range=(0, 11))
    bar = p2.hbar(
        y="names", height=0.5, left=0, right="scores", color="navy", source=s2
    )
    # p2.yaxis.ticker = [1,2,3,4,5,6,7,8,9,10]

    topN_code = """if (cb_data.source.selected.indices.length > 0){
            lines.visible = true;
            var selected_index = cb_data.source.selected.indices[0];
            lines.data_source.data['y'] = lines_y[selected_index]
            lines.data_source.change.emit(); 
          }"""

    fds.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(s1=fds, s2=s2, ax=p2.yaxis[0], b=bar),
            code=r"""
        const inds = cb_obj.indices;
        const d1 = s1.data;
        const d2 = s2.data;
        console.log(d2)
        console.log(inds)
        const feats = d1['top_feats'][inds[0]]
        //console.log(p.y_range)
        //p.y_range.factors=feats
        const od={1: 'a', 2: 'a',3: 'a',4: 'a',5: 'a',6: 'a',7: 'a',8: 'a',9: 'a',10: 'a'}
        const scores = d1['top_scores'][inds[0]]
        //d2['names'] = []
        d2['scores'] = scores
        //console.log(p)
        console.log(feats)
        console.log(scores)
        for (let i = 0; i < feats.length; i++) {
            //d2['names'].push(feats[i])
            //d2['scores'].push(scores[i])
            od[i]=feats[i]
        }
        ax.major_label_overrides = od
        console.log(d2)
        s2.change.emit();
    """,
        ),
    )

    # p.select(TapTool).callback=

    slider = Slider(title="Adjust alpha", start=0, end=1, step=0.05, value=0.5)
    toggle1 = Toggle(label="Show Mask", button_type="success")
    toggle2 = Toggle(label="Show Lines", button_type="success")
    toggle3 = Toggle(label="Show Dots", button_type="success")
    div = Div(text="""dots: score""", width=300, height=60)
    # drop=Dropdown(label='choose core', button_type='warning', menu=[('3-B','3-B'),('3-C','3-C'),('3-D','3-D')])

    callback = CustomJS(
        args=dict(m=mask, s=slider),
        code="""

    // JavaScript code goes here

    const a = 10;
    console.log('checkbox_button_group: active=' + this.active, this.toString())
    // the model that triggered the callback is cb_obj:
    const b = cb_obj.active;
    
    // models passed as args are automagically available
    if (this.active) {
        m.glyph.global_alpha = 0;
    } else {
        m.glyph.global_alpha = s.value;
    };
    console.log('b is: ' + b)

    """,
    )
    callback2 = CustomJS(
        args=dict(e=edges, s=slider),
        code="""
        if (this.active) {
            e.visible = false;
        } else {
            e.glyph.line_alpha = s.value;
            e.visible = true;
        };
    """,
    )
    callback3 = CustomJS(
        args=dict(n=nodes, s=slider),
        code="""
        if (this.active) {
            n.visible = false;
        } else {
            n.glyph.line_alpha = s.value;
            n.visible = true;
        };
    """,
    )
    slidercb = CustomJS(
        args=dict(e=edges, m=mask, n=nodes, s=slider, mt=toggle1, et=toggle2),
        code="""
        if (et.active) {
            e.glyph.line_alpha = 0;
        } else {
            e.glyph.line_alpha = s.value;
        };
        if (mt.active) {
            m.glyph.global_alpha = 0;
        } else {
            m.glyph.global_alpha = s.value;
        };
        n.glyph.line_alpha = s.value;
        n.glyph.fill_alpha = s.value;

    """,
    )

    # dropcb=CustomJS(args=dict(c=coreim), code="""
    #    c.glyph.url=`D:\All_cores\${this.item}.jpg`
    #
    # """)

    dropcb = CustomJS(
        args=dict(
            c=circ_cmapper,
            ehm=drop_ehist_mapper,
            lm=drop_lin_mapper,
            n=nodes,
            ds=fds,
            pal=RdYlGn11,
            p=div,
        ),
        code=r"""
        var low = Math.min.apply(Math,ds.data[this.item]);
        var high = Math.max.apply(Math,ds.data[this.item]);
        this.label=this.item
        p.text='dots:'+this.item
        console.log(this.item)
        c.field_name=this.item
        //var color_mapper = new Bokeh.LinearColorMapper({palette:pal, low:0, high:1});
        c.transform.update_data()
        if (this.item == 'ep_score' || this.item=='s_score') {
            var cm=lm
        }
        else {
            var cm=ehm
        }
        cm.low=low
        cm.high=high
        n.glyph.color = {field: this.item, transform: cm};
        n.glyph.fill_color = {field: this.item, transform: cm};
        n.glyph.line_color = {field: this.item, transform: cm};
        ds.change.emit();
    """,
    )
    # <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/dev/bokeh-api-3.0.0dev1.min.js"></script>

    # slider.js_link("value", nodes.glyph , "fill_alpha")
    slider.js_link("value", nodes.glyph, "line_alpha")
    # slider.js_link("value", edges.glyph , "line_alpha")
    slider.js_on_change("value", slidercb)

    toggle1.js_on_click(callback)
    toggle2.js_on_click(callback2)
    toggle3.js_on_click(callback3)
    drop_i.js_on_event("menu_item_click", dropcb)
    drop_f.js_on_event("menu_item_click", dropcb)

    # create layout
    gr = layout(
        [
            [p, [slider, toggle1, toggle2, toggle3, drop_i, drop_f, div, p2]],
        ]
    )

    # show result
    output_file(
        filename=f"D:/Meso/Bokeh_core_temp/{core}_{g.type_label[0]}.html",
        title="TMA cores graph NN visualisation",
    )
    save(gr)
    # show(gr)
    # html = file_html(gr, CDN, "my plot")
    # with open('D:/blah.html', 'w') as f:
    # f.write(html)
    print("save?")


def plotGraph(Xn, Wn, c=(0, 0, 0), tt=None, node_size=50, edge_alpha=1.0):
    """
    Shows graph based on node positions Xn, Adjacency Wn and color c
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(Xn.shape[0])))
    G.add_edges_from(Wn)
    pos = dict(zip(range(Xn.shape[0]), Xn))
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=np.abs(c), node_size=node_size)


def showGraph(G):
    """
    Show single pytorch graph object
    G.x: features
    G.v: features to be used for visualization
    G.c: used for colors (based on G.v or G.x)
    G.coords: position coordinates
    G.egde_index
    G.y
    """
    try:
        coords = toNumpy(G.coords)  # get coordinates
    except:
        coords = G.x[:, :2]  # if no coordinates use color specs
    plotGraph(
        coords,
        [tuple(e) for e in toNumpy(G.edge_index.t())],
        c=G.c,
        node_size=20,
        edge_alpha=0.25,
    )


def showGraphDataset(G):
    """
    Visualize a graph dataset through dimensionality reduction
    """
    G = deepcopy(G)
    try:
        X = np.vstack([toNumpy(G[g].v) for g in G.keys()])
    except:
        X = np.vstack([toNumpy(G[g].x) for g in G.keys()])
    L = np.cumsum([0] + [G[g].x.shape[0] for g in G.keys()])
    Y = [G[g].y for g in G.keys()]
    pos = np.sum([toNumpy(G[g].y) == 1 for g in G.keys()])
    neg = len(G) - pos
    if False:
        X = (X[:, 1] - X[:, 0]).reshape(-1, 1)

        # import pdb;pdb.set_trace()
        X = StandardScaler().fit_transform(X)
        if X.shape[1] > 3:
            tx = PCA()  # umap.UMAP(n_components=3,n_neighbors=6,min_dist = 0.0)#
            Xp = tx.fit_transform(X)[:, [0, 1, 2]]
        else:
            Xp = np.zeros((X.shape[0], 3))
            Xp[:, : X.shape[1]] = X
        Xp = MinMaxScaler().fit_transform(Xp)  # [:,[0,1,2]]

        # import pdb;pdb.set_trace()
        for i in range(X.shape[1]):
            Xp[:, i] = exposure.equalize_hist(Xp[:, i]) ** 2
        Xp[:, 1] = 1 - Xp[:, 0]

        # fig, axs = plt.subplots(2, 3)#max(pos,neg))
        # plt.subplots_adjust(wspace=0, hspace=0)
    else:
        Xp = np.zeros((X.shape[0], 3))
        Xp[:, [1, 0]] = X

    counts = [0, 0]
    for i, g in enumerate(G.keys()):
        G[g].c = Xp[L[i] : L[i + 1]]
        # y = int(G[g].y)
        # if counts[y]>=3:
        #    continue
        # ax = axs[y,counts[y]]
        # counts[y]+=1
        # plt.sca(ax)
        # showGraph(G[g])
        # plt.title(toNumpy(G[g].z)[0])
    # plt.show()

    for i, g in enumerate(G.keys()):
        # core=g
        # with Image.open(Path(f'D:\All_cores\{}.jpg')) as img:
        # plt.imshow(img)
        # showGraph(g)
        # coords = toNumpy(G[g].coords)     # get coordinates
        # bokeh_plot(coords,np.array([e for e in toNumpy(G[g].edge_index.t())]),G[g].c,core)
        bokeh_plot(G[g])
        # print('skip')


def Predict(data, model, device):
    """
    Get a pytorch dataset for node representation based on model output
    The node feaures of the input data are replaced with model based node repn

    """
    G = {}
    loader = DataLoader(data)
    model = model.to(device)
    model.eval()
    Z, core, lab, all_pred = [], [], [], []
    with torch.no_grad():
        for i, d in enumerate(loader):
            d = d.to(device)
            output, xx = model(d)
            Z.append(toNumpy(output[0]))
            G[d.core[0]] = Data(
                x=d.x,
                v=xx,
                edge_index=d.edge_index,
                y=d.y,
                coords=d.coords,
                z=output[0],
                core=d.core,
                type_label=d.type_label,
                feat_names=d.feat_names,
            )
            lab.append(d.y.item())
            core.append(d.core[0])
            all_pred.append((output[0, 1] / output[0, 0] + 0.00001).item())

    df = pd.DataFrame({"core": core, "label": lab, "GNN": all_pred})

    return G, df


#%%
def showImageDataset(G):
    X = np.vstack([toNumpy(g.v) for g in G])
    b, s = np.mean(X), np.std(X)
    pos = np.sum([toNumpy(g.y) == 1 for g in G])
    neg = len(G) - pos
    fig, axs = plt.subplots(4, min(10, max(pos, neg)))
    plt.subplots_adjust(wspace=0, hspace=None)

    counts = [0, 0]
    for i, g in enumerate(G):

        y = int(g.y)
        if counts[y] >= 10:
            continue
        ZZ, Z0 = showImage(g, b, s)
        ax = axs[2 * y, counts[y]]
        plt.sca(ax)
        plt.imshow(Z0)
        plt.title(toNumpy(g.z)[0])
        ax = axs[2 * y + 1, counts[y]]
        plt.sca(ax)
        plt.imshow(ZZ)
        counts[y] += 1


def showImage(g, b=0, s=1):
    v = toNumpy(g.v)
    # vn = StandardScaler().fit(v).transform(v)
    vn = (v - b) / s
    vn = 1 / (1 + np.exp(-2 * vn))
    # vn = 0.1+vn**2
    vn = vn > 0.7
    ZZ = np.zeros((24 * 32, 28 * 32, 3))
    Z0 = np.zeros((24 * 32, 28 * 32, 3))
    for k, z in enumerate(g.x):
        i, j = np.asarray(toNumpy(g.coords[k]), dtype=np.int)
        x = toNumpy(g.x[k].reshape(32, 32, 3))
        Z0[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32, :] = x
        ZZ[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32, :] = x * (vn[k])
    return ZZ, Z0


#%%


def save_preds(G):
    save_df = pd.DataFrame(np.array(G.coords.cpu()), columns={"x", "y"})
    save_df[G.feat_names[0]] = np.array(G.x.cpu())
    save_df[["score_E", "score_S"]] = np.array(G.v.cpu())
    save_df.to_csv(
        Path(
            f"D:\Results\TMA_results\\node_preds_split\GNN_scores_{G.core[0]}_{G.type_label[0]}.csv"
        )
    )


from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


def connectClusters(C, dthresh=3000):
    # W =  NearestNeighbors(n_neighbors=9).fit(C).kneighbors_graph(C).todense()
    W = NearestNeighbors(radius=3.5).fit(C).radius_neighbors_graph(C).todense()
    np.fill_diagonal(W, 0)
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


#%% Data Loading and Graph Creation
if __name__ == "__main__":
    # import pickle
    """PIK = "D:/breast_mil_spatial_3.5x.pkl"
    try:
        with open(PIK, "rb") as f:
            B,YY,Y,CC,dataset = pickle.load(f)
    except FileNotFoundError:
        print('Data file not found. Creating')

        from sklearn.preprocessing import normalize
        from misvmio import bag_set,parse_c45

        # dataset='musk2'
        # Bset = bag_set(parse_c45(dataset, rootdir='musk'))
        # B = [normalize(np.array(b.to_float())[:, 2:-1]) for b in Bset]
        # YY = np.array([b.label for b in Bset], dtype=float)

        from breast_mil import read_data
        B,YY,CC = read_data()

        print("Making Graphs")
        G = []
        Gbags = []
        Y = []
        #(0,inf,mean->0995/0.05,0.94/0.08),
        #(500,1e-2,mean,eps=0,eps_train=True->099/0.03,0.89/0.1)
        #(500,1e-2,mean,eps=1.0,eps_train=True->0.97/0.05,0.91/0.08)
        #(500,1e-2,mean,eps=5.0,eps_train=False->0.99/0.03,0.91/0.07)
        #(500,1e-2,mean,eps=10.0,eps_train=False->0.99/0.02,0.94/0.07)

        # gf_epochs = 500
        # gg_tt = 1e-2#np.inf
        # for b in tqdm(range(len(B))):
        #   Xb = B[b]
        #   n,d = Xb.shape
        #   Gb = GraphFit(n,d).fit(Xb, lr=1e-2, epochs=gf_epochs)
        #   if Gb.X.shape[0]>1:
        #       Gbags.append(Gb)
        #       G.append(toGeometric(Gb,tt=gg_tt,y=toTensor([1.0*(YY[b]>0)],dtype=torch.long,requires_grad = False)))
        #       Y.append(YY[b])


        for b in tqdm(range(len(B))):
            Xb,Cb,Yb = B[b],CC[b],YY[b]
            Wb = connectClusters(Cb)
            g = toGeometricWW(Xb,Wb,Yb)
            g.coords = toTensor(Cb)
            G.append(g)
            Y.append(YY[b])

        dataset = G
        datax = [B,YY,Y,CC,dataset]
        with open(PIK, "wb") as fp:
            pickle.dump(datax, fp)"""
    # for i in range(len(dataset)):
    #     dataset[i].coords = toTensor(CC[i])

    #%% MAIN TRAINING and VALIDATION

    # loss_class = MulticlassClassificationLoss
    learning_rate = 0.00005
    weight_decay = 0.02
    epochs = 400
    scheduler = None
    from sklearn.model_selection import StratifiedKFold, train_test_split

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # Vacc,Tacc=[],[]
    visualize = False  #'plots' #'pred'

    # added
    # dataset,Y, slide, _=mk_graph()
    dataset, Y, slide, _ = mk_graph("mesobank")
    test_dataset, Y_v, slide_v, _ = mk_graph("meso")
    print("made graphs, starting training..")

    # for trvi, test in skf.split(dataset, Y):

    va, ta = [], []
    for reps in range(3):
        Vacc, Tacc = [], []
        dfs = []
        m = 0
        for blah in range(1):
            # for trvi, test in skf.split(dataset, Y): #trvi, test in slide_fold(slide):
            # test_dataset=[dataset[i] for i in test]
            tt_loader = DataLoader(test_dataset, shuffle=True)

            trvi = range(len(dataset))
            train, valid = train_test_split(
                trvi, test_size=0.25, shuffle=True, stratify=np.array(Y)[trvi]
            )
            # train,valid = trvi, test
            sampler = StratifiedSampler(
                class_vector=torch.from_numpy(np.array(Y)[train]).cpu(), batch_size=16
            )

            train_dataset = [dataset[i] for i in train]
            # tr_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            valid_dataset = [dataset[i] for i in valid]

            v_loader = DataLoader(valid_dataset, shuffle=True)
            # sampler = StratifiedSampler(class_vector=torch.from_numpy(np.array(Y)[train_dataset]),batch_size = 16)
            tr_loader = DataLoader(train_dataset, batch_sampler=sampler)

            model = GIN(
                dim_features=train_dataset[0].x.shape[1],
                dim_target=2,
                layers=[10, 10, 10, 10, 10],
                dropout=0,
                pooling="mean",
                eps=100.0,
                train_eps=False,
                do_ls=True,
            )
            net = NetWrapper(model, loss_function=None, device=device)
            model = model.to(device=net.device)
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.7, nesterov=True)
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

            # if visualize: showGraphDataset(getVisData(test_dataset,net.model,net.device));#1/0

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
            torch.save(
                best_model, Path(f"D:\Results\TMA_results\models\\n11_fold_{m}.pt")
            )
            m += 1
            if visualize:
                G, df = Predict(test_dataset, net.model, net.device)
                df = add_missranked(df)
                if visualize == "plots" and reps == 0:
                    showGraphDataset(G)
                dfs.append(df)
                print("vis")
                for key in G:
                    save_preds(G[key])

        if visualize:
            pred_df = pd.concat(dfs, axis=0, ignore_index=True)
            pred_df.to_csv(
                Path(f"D:\Results\TMA_results\GNN_class_temp_dual_r{reps}.csv")
            )
        print("avg Valid acc=", np.mean(Vacc), "+/", np.std(Vacc))
        print("avg Test acc=", np.mean(Tacc), "+/", np.std(Tacc))
        va.append(np.mean(Vacc))
        ta.append(np.mean(Tacc))
    print(f"val accs were: {va}")
    print(f"test accs were: {ta}")
