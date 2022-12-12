import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CyclicLR
import torch
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
from torch_geometric.data import Data, DataLoader

"""MesoGraph model architecture definitions"""


class learnable_sig(torch.nn.Module):
    def __init__(self, fsize, type="branched") -> None:
        super(learnable_sig, self).__init__()
        if type == "branched":
            self.l1 = Sequential(Linear(fsize, fsize), ReLU(), Linear(fsize, 2))
            self.l2 = Sequential(Linear(fsize, fsize), ReLU(), Linear(fsize, 2))
            self.alpha = nn.parameter.Parameter(2 * torch.ones(1, 2))
            self.beta = nn.parameter.Parameter(torch.zeros(1, 2))
            self.gamma = nn.parameter.Parameter(torch.zeros(1, 2))
        else:
            self.l1 = Sequential(Linear(fsize, fsize), ReLU(), Linear(fsize, 1))
            self.l2 = Sequential(Linear(fsize, fsize), ReLU(), Linear(fsize, 1))
            self.alpha = nn.parameter.Parameter(torch.ones(1))
            self.beta = nn.parameter.Parameter(torch.zeros(1))
            self.gamma = nn.parameter.Parameter(torch.zeros(1))

    def forward(self, x, xcore, batch):
        y = []
        for i in torch.unique(batch):
            # last_ind=torch.sum(batch<=i)-1
            # y.append(torch.sigmoid(x[batch==i,:]*self.alpha-self.beta+self.gamma*torch.mean(x[batch==i],dim=0,keepdim=True))) #1
            # y.append(torch.sigmoid(x[batch==i,:]*self.alpha-self.beta))   #2
            # y.append(torch.sigmoid(x[batch==i,:]-self.beta))
            y.append(torch.sigmoid(x[batch == i, :] - self.l1(xcore.T[:, i])))
            # y.append(torch.sigmoid(x[batch==i,:]*self.alpha-0.1*self.l1(xcore.T[:,i])))
            # y.append(torch.sigmoid(x[batch==i,:]*self.l2(xcore.T[:,i])-self.l1(xcore.T[:,i])))
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
        super(MesoBranched, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.do_ls = do_ls
        if do_ls:
            self.ls = learnable_sig(dim_features, type="branched")
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
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]

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
                ) 

        self.linears = torch.nn.ModuleList(
            self.linears
        )  # has got one more for initial input

        self.ecnns = torch.nn.ModuleList(self.ecnns)
        self.ecs = torch.nn.ModuleList(self.ecs)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        
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


class MesoSep(torch.nn.Module):
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
        feats=[],
    ):
        super(MesoSep, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.do_ls = do_ls

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
        self.dim_features = dim_features
        self.subE = self.make_subnet()
        self.subS = self.make_subnet()
        self.feats = feats

        # train_eps = True#config['train_eps']

        # TOTAL NUMBER OF PARAMETERS #

        # first: dim_features*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target
        # l-th: input_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target

        # -------------------------- #

    def make_subnet(self):
        ecnns = []
        ecs = []
        linears = []
        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                first_h = Sequential(
                    Linear(self.dim_features, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )
                linears.append(Linear(out_emb_dim, 1))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]

                linears.append(Linear(out_emb_dim, 1))

                subnet = Sequential(
                    Linear(2 * input_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )

                ecnns.append(subnet)

                ecs.append(EdgeConv(ecnns[-1], aggr="mean"))  
        linears = torch.nn.ModuleList(linears)  # has got one more for initial input

        ecnns = torch.nn.ModuleList(ecnns)
        ecs = torch.nn.ModuleList(ecs)
        if self.do_ls:
            ls = learnable_sig(self.dim_features, type="sep")
            return nn.ModuleDict(
                {
                    "first": first_h,
                    "linears": linears,
                    "ecnns": ecnns,
                    "ecs": ecs,
                    "ls": ls,
                }
            )
        return nn.ModuleDict(
            {"first": first_h, "linears": linears, "ecnns": ecnns, "ecs": ecs}
        )

    def forward_sub(self, sub, data, edge_index=None, edge_weight=None):
        if edge_index == None:
            xfeat, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            xfeat = data
            batch = torch.zeros((xfeat.shape[0],), dtype=torch.long, device="cuda")

        out = 0
        pooling = self.pooling
        Z = 0

        last_ind = []
        for i in torch.unique(batch):
            last_ind.append(torch.sum(batch <= i) - 1)

        core_x = []
        for layer in range(self.no_layers):
            
            if layer == 0:
                # x, M = self.first_h(x, edge_index)
                x = sub["first"](xfeat)
                if self.do_ls:
                    core_x.append(x[last_ind, :])
                z = sub["linears"][layer](x)
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
                x = sub["ecs"][layer - 1](x, edge_index)
                if self.do_ls:
                    core_x.append(x[last_ind, :])
                # x = self.ecs[layer-1](x,batch)
                z = sub["linears"][layer](x)
                Z += z
                # dout=pooling(z, M, batch)
                # dout = F.dropout(torch.mean(pooling(z, batch, 1000),dim=1,keepdim=True), p=self.dropout, training=self.training)
                # dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)#F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                # out += dout

        if self.do_ls:
            core_x = torch.cat(core_x, dim=1)
            # ZZ=self.ls(Z,core_x.detach(),batch)    #learn based on embed
            ZZ = sub["ls"](
                Z, xfeat[last_ind, :], batch
            )  # learn sig thresh based on base feats
            out = pooling(ZZ, batch)
            return out, ZZ
        else:
            out = pooling(Z, batch)
            return out, Z

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        core_outE, cell_outE = self.forward_sub(self.subE, x, edge_index, edge_weight)
        core_outS, cell_outS = self.forward_sub(self.subS, x, edge_index, edge_weight)

        core_out = torch.cat([core_outE, core_outS], dim=1)
        cell_out = torch.cat([cell_outE, cell_outS], dim=1)
        if edge_index == None:
            return core_out, cell_out
        else:
            return cell_out
