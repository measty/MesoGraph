# -*- coding: utf-8 -*-
"""
Multiple Instance Graph Classification
@author: fayyaz
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from copy import deepcopy
from numpy.random import randn #importing randn
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
import time
from tqdm import tqdm
from scipy.spatial import distance_matrix, Delaunay
import random
from torch_geometric.data import Data, DataLoader

USE_CUDA = torch.cuda.is_available()
device = {True:'cuda',False:'cpu'}[USE_CUDA] 
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = True):
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
    """    
    def __init__(self,n,d):          
        self.n,self.d = n,d
        self.W = cuda(Variable(torch.rand((n,n)).float())).requires_grad_()
        self.history = []
    def fit(self,X,lr=1e-2,epochs=500):
        X = toTensor(X,requires_grad=False)
        self.X = X              
        optimizer = optim.Adam([self.W], lr=lr)        
        alpha = 1.0      
        zero = toTensor([0])        
        for epochs in range(epochs):
            L,D = self.getLaplacianDegree()                 
            loss = torch.norm(L@X)+alpha*torch.sum(torch.max(zero,1-D)**2)
            
            self.history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
        
        return self
            
    def getLaplacianDegree(self):
        W = self.getW()            
        L = -W
        D = torch.sum(W,dim=0)
        L.diagonal(dim1=-2, dim2=-1).copy_(D)  
        return L,D
            
    def getW(self):
        """
        Gets adjacency matrix for the graph
        """
        Z = (torch.transpose(self.W, 0, 1)+self.W)**2
        Z.fill_diagonal_(0)
        return Z
    

def toGeometric(Gb,y,tt=1e-3):
    """
    Create pytorch geometric object based on GraphFit Object
    """
    return Data(x=Gb.X, edge_index=(Gb.getW()>tt).nonzero().t().contiguous(),y=y)
#%% GIN Model Implementation
    
from torch.utils.data import Sampler
class StratifiedSampler(Sampler):
    """Stratified Sampling
         return a stratified batch
    """
    def __init__(self, class_vector, batch_size = 10):
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
        skf = StratifiedKFold(n_splits= self.n_splits,shuffle=True)
        YY = self.class_vector.numpy()
        idx = np.arange(len(YY))
        return [tidx for _,tidx in skf.split(idx,YY)] #return array of arrays of indices in each batch
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

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        """
        outputs = outputs[0]
        loss = self.loss(outputs, targets)
        accuracy = self._calculate_accuracy(outputs, targets)
        return loss, accuracy

    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs, targets):
        correct = self._get_correct(outputs)
        return 100. * (correct == targets).sum().float() / targets.size(0)

class BinaryClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def _get_correct(self, outputs):
        return outputs > 0.5
    
class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.CrossEntropyLoss(reduction=reduction)
        else:
            self.loss = nn.CrossEntropyLoss()

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)

from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
def calc_roc_auc(target, prediction):
    #import pdb;pdb.set_trace()
    #return np.mean(toNumpy(target)==(toNumpy(prediction[:,1]-prediction[:,0])>0))
    
    return roc_auc_score(toNumpy(target),toNumpy(prediction[:,1]-prediction[:,0]))
    # output = F.softmax(prediction, dim=1)
    # output = output.detach()[:, 1]
    # fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy())
    # roc_auc = auc(fpr, tpr)
    # import pdb;pdb.set_trace()
    # return roc_auc

class GIN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[16],pooling='max',dropout = 0.0,eps=0.0,train_eps=False):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=[layers[0]]+layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        #train_eps = True#config['train_eps']

        # TOTAL NUMBER OF PARAMETERS #

        # first: dim_features*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target
        # l-th: input_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target

        # -------------------------- #

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                #self.linears.append(Linear(dim_features, dim_target))
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], eps=eps, train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        #self.first_h = torch.nn.ModuleList(self.first_h)
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        out = 0
        pooling = self.pooling

        for layer in range(self.no_layers):
            # print(f'Forward: layer {l}')
            if layer == 0:
                x = self.first_h(x)
                dout = F.dropout(pooling(self.linears[layer](x), batch), p=self.dropout, training=self.training)
                out += dout
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                dout = F.dropout(pooling(self.linears[layer](x), batch), p=self.dropout, training=self.training)#F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout
        
        return out,x
    
   
class NetWrapper:
    def __init__(self, model, loss_function, device='cpu', classification=True):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification
    def _pair_train(self,train_loader,optimizer,clipping = None):
        """
        Performs pairwise comparisons with ranking loss
        """
        model = self.model.to(self.device)
        model.train()
        loss_all = 0
        acc_all = 0
        assert self.classification
        #lossfun = nn.MarginRankingLoss(margin=1.0,reduction='sum')
        for data in train_loader:
            
            data = data.to(self.device)
            
            optimizer.zero_grad()
            output,xx = model(data)
            #import pdb; pdb.set_trace()
            # Can add contrastive loss if reqd
            #import pdb; pdb.set_trace()
            y = data.y
            loss =0
            c = 0
            #z = Variable(torch.from_numpy(np.array(0))).type(torch.FloatTensor)
            z = toTensor([0])  
            for i in range(len(y)-1):
                for j in range(i+1,len(y)):
                    if y[i]!=y[j]:
                        c+=1
                        dz = output[i,1]-output[j,1]
                        dy = y[i]-y[j]                        
                        loss+=torch.max(z, 1-dy*dz)
                        #loss+=lossfun(zi,zj,dy)
            loss=loss/c

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
            return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)
        else:
            return None, loss_all / len(train_loader.dataset)

    def classify_graphs(self, loader):
        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                data = data.to(self.device)
                output,xx = model(data)
                if i == 0:
                    Z = output
                    Y = data.y
                else:
                    Z = torch.cat((Z, output))
                    Y = torch.cat((Y, data.y))
            if not isinstance(Z, tuple):
                Z = (Z,)
            loss, acc = self.loss_fun(Y, *Z)
            auc_val = calc_roc_auc(Y, *Z)
            #pr = calc_pr(Y, *Z)
            return auc_val, loss#, auc, pr
        
    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=None, log_every=1000):
        
        early_stopper = early_stopping() if early_stopping is not None else None

        val_loss, val_acc = -1, -1
        test_loss, test_acc = None, None

        time_per_epoch = []
        self.history = []
        
        best_val_acc = -1
        return_best = True
        test_acc_at_best_val_acc = -1
        for epoch in tqdm(range(1, max_epochs+1)):

            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()
            
            train_acc, train_loss = self._pair_train(train_loader, optimizer, clipping)
            
            end = time.time() - start
            time_per_epoch.append(end)
            
            if test_loader is not None:
                test_acc, test_loss = self.classify_graphs(test_loader)

            if validation_loader is not None:
                val_acc, val_loss = self.classify_graphs(validation_loader)
            
            if epoch % log_every == 0 :
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} ' \
                    f'TE loss: {test_loss} TE acc: {test_acc}'
                print('\n'+msg)   
                
            self.history.append(train_loss)
            
            if val_acc>=best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val_acc = test_acc
                
        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc

        if early_stopper is not None:
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, best_epoch = early_stopper.get_best_vl_metrics()
            
        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

#%% Dummy toy data generation        
def getExamples(n=800,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]    
    """
    Xp1 = randn(int(n/2),d)+3.0#+1   #generate examples of the positie class    
    Xp2 = randn(int(n/2),d)-9.0#+1   #generate examples of the positie class
    Xp = np.vstack((Xp1,Xp2))
    
    Xn1 = randn(int(n/2),d)-np.array([3,3])#-1   #generate n examples of the negative class
    Xn2 = randn(int(n/2),d)+7.0
    Xn = np.vstack((Xn1,Xn2))
    
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    y = np.array([+1]*Xp.shape[0]+[-1]*Xn.shape[0]) #Associate Labels
    
    Noise = randn(n,d)+[-3,5] #generate noise
    Noise = np.vstack((Noise, randn(n,d)+[-3,-11]))
    X = np.vstack((X,Noise)) #add noise
    y = np.append(y,[0]*len(Noise))
    
    X+=2
#    y = -y
    ridx = np.random.permutation(range(len(y)))
    X,y = X[ridx,:],y[ridx]
    return X,y

def genBags(y):
    """
    Add examples to bags
        Positive bag: has at least one positive example 
        mexpb: maximum number of examples per bag
        mprop: proportion of majority class in a bag
        nprop: proportion of noise class in a bag
        
    """
    pid,nid,noise = list(np.where(y==1)[0]),list(np.where(y==-1)[0]),list(np.where(y==0)[0])   
    
    Nbags = 30 #number of bags
    mepb = 30 # max number of example per bag
    mprop = 0.05 #majority proportion
    nprop = 0.00 #noise proportion per bag
    Bsize = np.random.binomial(n=mepb, p=0.5, size=Nbags)
    print("Avg. Examples/Bag:",np.mean(Bsize))
    Bidx = []
    Y = np.array([-1]*int(Nbags/2)+[1]*int(Nbags/2))#np.random.choice([-1, 1], size=(Nbags,), p=[0.5, 0.5])
    for i in range(len(Y)):
        M = int(np.ceil(Bsize[i]*mprop))
        n = int(Bsize[i]*nprop)
        m = Bsize[i]-M-n 
        if Y[i]==1:
            B = pid[:M]; pid = pid[M:] #add M examples from the positive class
#            print("Pos",len(B))
            B+= nid[:m]; nid = nid[m:] #add m examples from the negative class            
        else:
            B = nid[:M]; nid = nid[M:] #add M+m examples from negative class
            B+= nid[:m]; nid = nid[m:]

        B+= noise[:n]; noise = noise[n:] #add n examples of noise
        
        Bidx.append(np.array(B))
        
    return Bidx,Y


#%% VISUALIZATION 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
#import umap
from skimage import exposure
from sklearn.decomposition  import PCA

def plotGraph(Xn, Wn, c=(0,0,0),tt=None,node_size=50,edge_alpha=1.0): 
    """
    Shows graph based on node positions Xn, Adjacency Wn and color c
    """
    G=nx.Graph()
    G.add_nodes_from(list(range(Xn.shape[0])))
    G.add_edges_from(Wn)   
    pos = dict(zip(range(Xn.shape[0]),Xn))    
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width = 0.5);    
    nx.draw_networkx_nodes(G, pos, node_color=np.abs(c),node_size=node_size)

def showGraph(G):
    """
    Show single pytorch graph object 
    """
    try:
        coords = toNumpy(G.coords)     # get coordinates
    except:
        coords = G.c[:,:2]  #if no coordinates use color specs
    plotGraph(coords,[tuple(e) for e in toNumpy(G.edge_index.t())],c=G.c,node_size=5,edge_alpha=0.25)
    
def showGraphDataset(G):
    """
    Visualize a graph dataset through dimensionality reduction
    """
    G = deepcopy(G)
    X = np.vstack([g.x for g in G])
    L = np.cumsum([0]+[g.x.shape[0] for g in G])
    Y = [g.y for g in G]
    X = StandardScaler().fit_transform(X)
    tx = PCA()#umap.UMAP(n_components=3,n_neighbors=6,min_dist = 0.0)#
    Xp = tx.fit_transform(X)[:,[0,1,2]]
    #Xp = MinMaxScaler().fit_transform(Xp)[:,[0,1,2]]      
    for i in range(Xp.shape[1]):
        Xp[:,i] = 1-exposure.equalize_hist(Xp[:,i])**2
    
    
    fig, axs = plt.subplots(2, 10)
    plt.subplots_adjust(wspace=0, hspace=0)

    counts = [0,0]
    for i,g in enumerate(G):    
        g.c = Xp[L[i]:L[i+1]]
        y = int(g.y)
        if counts[y]>=10:
            continue
        ax = axs[y,counts[y]]
        counts[y]+=1
        plt.sca(ax)
        showGraph(g)

    
def getVisData(data,model,device):
    """
    Get a pytorch dataset for node representation based on model output
    The node feaures of the input data are replaced with model based node repn

    """
    G = []
    loader = DataLoader(data)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(loader):
            d = d.to(device)
            output,xx = model(d)
            G.append(Data(x=xx, edge_index=d.edge_index,y=d.y))
    return G
    

#%% Data Loading and Graph Creation
if __name__=='__main__':
    from sklearn.preprocessing import normalize
    from misvmio import bag_set,parse_c45
    
    dataset='musk2'
    Bset = bag_set(parse_c45(dataset, rootdir='musk'))
    B = [normalize(np.array(b.to_float())[:, 2:-1]) for b in Bset]
    YY = np.array([b.label for b in Bset], dtype=float)
    
    print("Making Graphs")
    G = []
    Gbags = []    
    Y = []
    #(0,inf,mean->0995/0.05,0.94/0.08),
    #(500,1e-2,mean,eps=0,eps_train=True->099/0.03,0.89/0.1)
    #(500,1e-2,mean,eps=1.0,eps_train=True->0.97/0.05,0.91/0.08)
    #(500,1e-2,mean,eps=5.0,eps_train=False->0.99/0.03,0.91/0.07)
    #(500,1e-2,mean,eps=10.0,eps_train=False->0.99/0.02,0.94/0.07)
    gf_epochs = 500
    gg_tt = 1e-2#np.inf
    for b in tqdm(range(len(B))):
      Xb = B[b]
      n,d = Xb.shape
      Gb = GraphFit(n,d).fit(Xb, lr=1e-2, epochs=gf_epochs)
      if Gb.X.shape[0]>1:
          Gbags.append(Gb)
          G.append(toGeometric(Gb,tt=gg_tt,y=toTensor([1.0*(YY[b]>0)],dtype=torch.long,requires_grad = False)))
          Y.append(YY[b])
    
    dataset = G
    #%% MAIN TRAINING and VALIDATION
       
    loss_class = MulticlassClassificationLoss
    learning_rate = 0.001
    weight_decay =0.001
    epochs = 300
    scheduler = None
    from sklearn.model_selection import StratifiedKFold, train_test_split
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    Vacc,Tacc=[],[]
    for trvi, test in skf.split(dataset, Y):
        test_dataset=[dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=True)
        
        train, valid = train_test_split(trvi,test_size=0.1,shuffle=True,stratify=np.array(Y)[trvi])
        #train,valid = trvi, test
        sampler = StratifiedSampler(class_vector=torch.from_numpy(np.array(Y)[train]),batch_size = 16)
        
          
        train_dataset=[dataset[i] for i in train]    
        #tr_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        tr_loader = DataLoader(train_dataset, batch_sampler = sampler)
        valid_dataset=[dataset[i] for i in valid]    
        v_loader = DataLoader(valid_dataset, shuffle=True)
    
        model = GIN(dim_features=166, dim_target=2, layers=[6,],dropout = 0.25,pooling='mean',eps=100.0,train_eps=False)
        net = NetWrapper(model, loss_function=loss_class(), device=device)
        model = model.to(device = net.device)
        optimizer = optim.Adam(model.parameters(),
                                lr=learning_rate, weight_decay=weight_decay)
        
        #showGraphDataset(getVisData(valid_dataset,net.model,net.device))
            
        train_loss, train_acc, val_loss, val_acc, tt_loss, tt_acc = net.train(train_loader=tr_loader,
                                                                   max_epochs=epochs,
                                                                   optimizer=optimizer, scheduler=scheduler,
                                                                   clipping=None,
                                                                   validation_loader=v_loader,
                                                                   test_loader=tt_loader,
                                                                   early_stopping=None,
                                                                   )
        
        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        print ("fold complete", len(Vacc),train_acc,val_acc,tt_acc)
        
        #showGraphDataset(getVisData(valid_dataset,net.model,net.device))
        #1/0
    print ("avg Valid acc=", np.mean(Vacc),"+/", np.std(Vacc))
    print ("avg Test acc=", np.mean(Tacc),"+/", np.std(Tacc))
    
    
        
            