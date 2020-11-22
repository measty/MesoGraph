#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:17:25 2020

@author: u1876024
"""

from utils import *
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



from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
def calc_roc_auc(target, prediction):
    #import pdb;pdb.set_trace()
    #return np.mean(toNumpy(target)==(toNumpy(prediction[:,1]-prediction[:,0])>0))
    
    return roc_auc_score(toNumpy(target),toNumpy(prediction[:,-1]))
    # output = F.softmax(prediction, dim=1)
    # output = output.detach()[:, 1]
    # fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy())
    # roc_auc = auc(fpr, tpr)
    # import pdb;pdb.set_trace()
    # return roc_auc

#%% Graph Neural Network 
class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[6,6],pooling='max',dropout = 0.0,conv='GINConv',gembed=False,**kwargs):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        self.gembed = gembed #if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        #train_eps = True#config['train_eps']

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())                
                self.linears.append(Linear(out_emb_dim, dim_target))
                
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))                
                if conv=='GINConv':
                    self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                           Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                    self.convs.append(GINConv(self.nns[-1], **kwargs))  # Eq. 4.2 eps=100, train_eps=False
                elif conv=='EdgeConv':
                    subnet = Sequential(Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                          Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())                    
                    self.nns.append(subnet)                    
                    self.convs.append(EdgeConv(self.nns[-1],**kwargs))#DynamicEdgeConv#EdgeConv                aggr='mean'

                else:
                    raise NotImplementedError  
                    
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
        Z = 0
        for layer in range(self.no_layers):            
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z+=z
                dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z+=z
                    dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout

        return out,Z,x
    
#%% Wrapper for neetwork training   
    
def decision_function(model,loader,device='cpu',outOnly=True,returnNumpy=False): 
    if type(loader) is not DataLoader: #if data is given
        loader = DataLoader(loader)
    if type(device)==type(''):
        device = torch.device(device)
    ZXn = []    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            output,zn,xn = model(data)
            if returnNumpy:
                zn,xn = toNumpy(zn),toNumpy(xn)
            if not outOnly:
                ZXn.append((zn,xn))
            if i == 0:
                Z = output
                Y = data.y
            else:
                Z = torch.cat((Z, output))
                Y = torch.cat((Y, data.y))
    if returnNumpy:
        Z,Y = toNumpy(Z),toNumpy(Y)
    return Z,Y,ZXn
#%%   
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
            output,_,_ = model(data)
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
                        dz = output[i,-1]-output[j,-1]
                        dy = y[i]-y[j]                        
                        loss+=torch.max(z, 1.0-dy*dz)
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
        Z,Y,_ = decision_function(self.model,loader,device=self.device)
        if not isinstance(Z, tuple):
            Z = (Z,)
        #loss, acc = self.loss_fun(Y, *Z)
        loss = 0
        auc_val = calc_roc_auc(Y, *Z)
        #pr = calc_pr(Y, *Z)
        return auc_val, loss#, auc, pr
        
    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=100, log_every=0):
        
        

        val_loss, val_acc = -1, -1
        test_loss, test_acc = None, None

        time_per_epoch = []
        self.history = []
        
        best_val_acc = -1
        return_best = True
        test_acc_at_best_val_acc = -1
        patience = early_stopping
        best_epoch = 0
        iterator = tqdm(range(1, max_epochs+1))
        
        for epoch in iterator:
            updated = False

            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()
            
            train_acc, train_loss = self._pair_train(train_loader, optimizer, clipping)
            
            end = time.time() - start
            time_per_epoch.append(end)            
            if validation_loader is not None:
                val_acc, val_loss = self.classify_graphs(validation_loader)
                
            if val_acc>=best_val_acc:
                best_val_acc = val_acc
                if test_loader is not None:
                    test_acc, test_loss = self.classify_graphs(test_loader)
                test_acc_at_best_val_acc = test_acc
                best_model = deepcopy(self.model)
                best_epoch = epoch
                updated = True
            showresults = False
            if log_every==0:
                showresults = updated
            elif (epoch-1) % log_every == 0:   
                showresults = True
                if test_loader is not None and not updated:
                    test_acc, test_loss = self.classify_graphs(test_loader)
            if showresults:                
                # msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} ' \
                #     f'TE loss: {test_loss} TE acc: {test_acc}'
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL acc: {val_acc} ' \
                    f'TE acc: {test_acc}, Best: VL acc: {best_val_acc} TE acc: {test_acc_at_best_val_acc}'
                tqdm.write('\n'+msg)                   
                self.history.append(train_loss)
                
                
            if False: #or 

                from vis import showGraphDataset,getVisData                
                fig = showGraphDataset(getVisData(validation_loader,best_model,self.device,showNodeScore=False))
                plt.savefig(f'./figout/{epoch}.jpg')
                plt.close()
                    
            

            if epoch-best_epoch>patience: 
                iterator.close()
                break
            
                
        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc            
        else:
            best_model = deepcopy(self.model)
            
        return best_model,train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
