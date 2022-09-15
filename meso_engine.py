from zmq import device
from utils import toNumpy, toTensor
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import torch
import time
from tqdm import tqdm
from copy import deepcopy
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd

"""Core training and evaluation/inference code for MesoNet models"""

def calc_roc_auc(target, prediction):
   
    return roc_auc_score(toNumpy(target),toNumpy(prediction[:,-1]))
   

class alpha_scaler():
    def __init__(self, alpha, step_size) -> None:
        self.alpha=alpha
        self.step_size=step_size
    
    def update_alpha(self):
        if self.alpha+self.step_size<1:
            self.alpha+=self.step_size


class NetWrapper:
    def __init__(self, model, loss_function, device='gpu', classification=True, save_dir=None,):
        self.model = model
        self.scaler=alpha_scaler(0,0.005)
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification
        self.save_dir = save_dir
    
    @staticmethod
    def _clip_diff(dy, dz):
        return torch.clamp(dy, min=-1.0, max=1.0)

    def _pair_train(self,train_loader,optimizer,scheduler,clipping = None):
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
                        dz = output[i,:]-output[j,:]
                        dy = torch.stack([self._clip_diff(y[j]-y[i], dz[0]),self._clip_diff(y[i]-y[j], dz[1])])                       
                        loss+=torch.mean(torch.max(z, 0.8-dy*dz))  #1.0 or 0.5?
                        #loss+=lossfun(zi,zj,dy)
            loss=loss/c
            #extra loss component to penalise cells being both ep and sarc, also may
            #act as a regularisation
            #loss_reg=torch.mean(xx)
            #loss_es=torch.mean(torch.prod(xx,dim=1)**2)
            #loss_es=torch.mean(torch.max(toTensor(0.0).to(device),torch.prod(xx+toTensor(-0.1).to(device),dim=1))**2)
            #loss_es=torch.mean(torch.max(toTensor(0.0).to(device),torch.prod(xx+toTensor(-0.1).to(device),dim=1)))**2
            #loss=loss+0.5*self.scaler.alpha*loss_es#+0.1*loss_reg
            #loss=loss+0.5*self.scaler.alpha*loss_reg

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
        Original training method. Not used at the moment but kept for reference.
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
                    y = data.y
                else:
                    Z = torch.cat((Z, output))
                    y = torch.cat((y, data.y))

            loss =0
            c = 0
            #z = Variable(torch.from_numpy(np.array(0))).type(torch.FloatTensor)
            z = toTensor([0])  
            for i in range(len(y)-1):
                for j in range(i+1,len(y)):
                    if y[i]!=y[j]:
                        c+=1
                        dz = Z[i,:]-Z[j,:]
                        dy = torch.stack([y[j]-y[i],y[i]-y[j]])                       
                        loss+=torch.mean(torch.max(z, 1.0-dy*dz))
                        #loss+=lossfun(zi,zj,dy)
            loss=loss.item()/c

            #if not isinstance(Z, tuple):
            #    Z = (Z,)
            #loss, acc = self.loss_fun(Y, *Z)
            #loss = 0
            auc_val = calc_roc_auc(torch.minimum(y,torch.ones(1, dtype=torch.int64).to(self.device)), torch.unsqueeze(Z[:,-1] - Z[:,0], 1))#torch.unsqueeze(Z[:,-1]/(Z[:,0]+0.00001),1))
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

            #if scheduler is not None:
                #scheduler.step(epoch)
            start = time.time()
            
            train_acc, train_loss = self._pair_train(train_loader, optimizer, scheduler, clipping)
            
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
            self.scaler.update_alpha()
            
            if val_acc>=best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val_acc = test_acc
                best_model = deepcopy(self.model)
        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc            
        else:
            best_model = deepcopy(self.model)

        if early_stopper is not None:
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, best_epoch = early_stopper.get_best_vl_metrics()
            
        return best_model,train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

    def predict(self,data,model):
        """
        Predict scores on data. Return node scores in G, 
        and core-level scores in df.

        """
        device = self.device
        G = {}
        loader = DataLoader(data)
        model = model.to(device)
        model.eval()
        Z, core, lab, all_pred = [],[],[],[]
        with torch.no_grad():
            for i, d in enumerate(loader):
                d = d.to(device)
                output,xx = model(d)
                Z.append(toNumpy(output[0]))
                G[d.core[0]]=Data(x=d.x,v=xx, edge_index=d.edge_index,y=d.y,coords=d.coords,z=output[0],core=d.core, type_label=d.type_label,feat_names=d.feat_names)
                lab.append(d.y.item())
                core.append(d.core[0])
                all_pred.append((output[0,1]/output[0,0]+0.00001).item())

        df=pd.DataFrame({'core': core, 'y': lab, 'y_pred': all_pred})

        return G, df

    def save_preds(self,g, include_feats=False):
        #save all node outputs to a df for each core, optionally with node feats
        save_df=pd.DataFrame(np.array(g.coords.cpu()), columns={'x','y'})
        if include_feats:
            #save_df[g.feat_names[0]]=np.array(g.x.cpu())
            save_df = pd.concat([save_df,pd.DataFrame(np.array(g.x.cpu()),index=save_df.index,columns=g.feat_names[0])],axis=1)
        save_df[['score_E', 'score_S']]=np.array(g.v.cpu())
        save_df.to_csv(Path(self.save_dir)/'node_preds'/f'GNN_scores_{g.core[0]}_{g.type_label[0]}.csv')    
