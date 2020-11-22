#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:47:12 2020

@author: u1876024
"""


# -*- coding: utf-8 -*-
"""
Multiple Instance Graph Classification
@author: fayyaz
"""

from grafit import *
from GNN import *
from vis import *
#%% Data Loading and Graph Creation
def create_bags_mat(path='./data/tiger_100x100_matlab.mat'):#elephant,fox,tiger
    mat=scipy.io.loadmat(path)
    ids=mat['bag_ids'][0]
    f=scipy.sparse.csr_matrix.todense(mat['features'])
    l=np.array(scipy.sparse.csr_matrix.todense(mat['labels']))[0]
    bags=[]
    labels=[]
    for i in set(ids):
        bags.append(np.array(f[ids==i]))
        labels.append(l[ids==i][0])
    bags=np.array(bags)
    labels=np.array(labels)
    return bags, labels
if __name__=='__main__':

    from sklearn.preprocessing import normalize
    import scipy
    bags, labels=create_bags_mat()

    B = [normalize(b) for b in bags]
    YY = labels
    

    
    print("Making Graphs")
    G = []
    Gbags = []    
    Y = []
    
    gf_epochs = 500#0#
    gg_tt = 1e-2#np.inf#
    for b in tqdm(range(len(B))):
      Xb = B[b]
      n,d = Xb.shape
      Gb = GraphFit(n,d).fit(Xb, lr=1e-2, epochs=gf_epochs)
      if Gb.X.shape[0]>0:
          Gbags.append(Gb)
          G.append(toGeometric(Gb,tt=gg_tt,y=toTensor([1.0*(YY[b]>0)],dtype=torch.long,requires_grad = False)))
          Y.append(YY[b])
    
    

    dataset = G


            
    #%% MAIN TRAINING and VALIDATION
       
    #loss_class = MulticlassClassificationLoss
    learning_rate = 0.001
    weight_decay =0.001
    epochs = 300
    scheduler = None
    from sklearn.model_selection import StratifiedKFold, train_test_split
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    Vacc,Tacc=[],[]
    
    Fdata = []
    for trvi, test in skf.split(dataset, Y):
        test_dataset=[dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=False)
        
        train, valid = train_test_split(trvi,test_size=0.1,shuffle=True,stratify=np.array(Y)[trvi])
        #train,valid = trvi, test
        sampler = StratifiedSampler(class_vector=torch.from_numpy(np.array(Y)[train]),batch_size = 12)
        
          
        train_dataset=[dataset[i] for i in train]    
        #tr_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        tr_loader = DataLoader(train_dataset, batch_sampler = sampler)
        valid_dataset=[dataset[i] for i in valid]    
        v_loader = DataLoader(valid_dataset, shuffle=False)
    
        model = GNN(dim_features=dataset[0].x.shape[1], dim_target=1, layers=[6,6],dropout = 0.25,pooling='mean',
                    conv='GINConv',train_eps=True,eps=100)#
        net = NetWrapper(model, loss_function=None, device=device)
        model = model.to(device = net.device)
        optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        
        #if visualize: showGraphDataset(getVisData(test_dataset,net.model,net.device));#1/0
            
        best_model,train_loss, train_acc, val_loss, val_acc, tt_loss, tt_acc = net.train(train_loader=tr_loader,
                                                                   max_epochs=epochs,
                                                                   optimizer=optimizer, scheduler=scheduler,
                                                                   clipping=None,
                                                                   validation_loader=v_loader,
                                                                   test_loader=tt_loader,
                                                                   early_stopping=np.inf,
                                                                   )
        Fdata.append((best_model,test_dataset,valid_dataset))
        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        print ("fold complete", len(Vacc),train_acc,val_acc,tt_acc)
        

    print ("avg Valid AUC=", np.mean(Vacc),"+/", np.std(Vacc))
    print ("avg Test AUC=", np.mean(Tacc),"+/", np.std(Tacc))
    
    
    #%% acc calcl
    print ("avg Valid AUC=", np.mean(Vacc),"+/", np.std(Vacc))
    print ("avg Test AUC=", np.mean(Tacc),"+/", np.std(Tacc))
    aa = []
    for idx in [8]:#range(len(Fdata)):
        model,test_dataset,valid_dataset = Fdata[idx]
        Gv = getVisData(test_dataset,model,net.device); 
        yy= np.array([toNumpy(g.y) for g in Gv]).ravel(); 
        zz = np.array([toNumpy(g.z) for g in Gv]).ravel();
        _,_,tt = roc_curve(yy,zz)
        aa.append(np.max([np.mean(yy==(zz>=t)) for t in tt]))
    print(np.mean(aa),np.std(aa));print(aa)
    visualize = 1
    if visualize:
            showGraphDataset(getVisData(test_dataset,model,net.device))
    
        
            