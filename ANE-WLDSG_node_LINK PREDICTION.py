# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:34:13 2023

@author: amr AL-furas
"""

import time
import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import coo_matrix
import math
from tqdm import tqdm
from keras.optimizers import SGD
import pickle
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from karateclub.utils.walker import RandomWalker
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import  confusion_matrix
#'=============================================='
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = x 
    return e_x / e_x.sum(axis=0)

def amr_re(x):
    for i in range(x.size):
        y=np.zeros((x.size,1))
        if x[i]>1:
           y[i]=1
        else:
            y[i]=x[i]
    return y        
#==========================================================================

from pathlib import Path
p = Path('dataset/node_level/wikipedia/edges.csv')
data=pd.read_csv(p)
Gnx = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
x=Gnx.number_of_nodes()
y=Gnx.number_of_edges()
#===========================================================================
p = Path('dataset/node_level/wikipedia/features.csv')
data_f=pd.read_csv(p)
row = np.array(data_f["node_id"])
col = np.array(data_f["feature_id"])
values = np.array(data_f["value"])
node_count = max(row) + 1
feature_count = max(col) + 1
shape = (node_count, feature_count)
X1 = coo_matrix((values, (row, col)), shape=shape)
X2=coo_matrix.todense(X1)
#==========================================================================
p = Path('dataset/node_level/wikipedia/target.csv')
data_t=pd.read_csv(p)
target = np.array(data_t["target"])

#=============================================================================

walker = RandomWalker(20, 5)
walker.do_walks(Gnx)
walkers=walker.walks
nod=[]
trget=[]
for i in tqdm(range( len(walkers))):
 j=1
 while (j<len(walkers[i])-1):
     tt=[]
     nod.append(int(walkers[i][j]))
#     tt.append(int(walkers[i][j-2]))
     tt.append(int(walkers[i][j-1]))
     tt.append(int(walkers[i][j]))
     tt.append(int(walkers[i][j+1]))
 #    tt.append(int(walkers[i][j+2]))
     trget.append(tt)
     j=j+1
    
with open('WLk_wikipedia', 'wb') as f:
     pickle.dump(trget, f) 
#X=preprocessing.normalize(X2)


#=============================================================================
#=============================================================================
print('split _graph')
fb_df_temp = data.copy()
fb_df_partial,fb_df_ghost = train_test_split(fb_df_temp,test_size=.20) 
G_data = nx.from_pandas_edgelist(fb_df_partial, "id_1", "id_2", create_using=nx.Graph())
all_unconnected_pairs = []
for v in  (Gnx.nodes()):
    j=0
    for u in (Gnx.nodes()):
        if j<3 :   
           if nx.shortest_path_length(Gnx, v, u) >3 and v!=u:
               all_unconnected_pairs.append([int(v),int(u)])
               j=j+1
node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]
data1 = pd.DataFrame({'id_1':node_1_unlinked, 
                     'id_2':node_2_unlinked})

# add target variable 'link'
data1['link'] = -1
fb_df_ghost['link'] = 1
#=====================================================================================
X=X2
for k in range(4): 
 print('iteration----------------',k)   
 for v in Gnx.nodes():
     T1=X[int(v)]
     for u in Gnx.neighbors(v):
         Y1 =(X[int(v)] +(1/math.sqrt(Gnx.degree(v)*Gnx.degree(u)))* X2[int(u)])
         Y=np.where(Y1 >1.0, 1.0, Y1)
         X[int(v)]=Y
       
 X2=X 
 
#====================================================================================== 
with open('WL_wikipedia.npy', 'wb') as f:
        np.save(f, np.array(X2)) 
        
        
target_ok1=np.zeros((len(nod),len(X2[1].T)),dtype=float)
target_ok2=np.zeros((len(nod),len(X2[1].T)),dtype=float)
target_ok3=np.zeros((len(nod),len(X2[1].T)),dtype=float)
target_ok4=np.zeros((len(nod),len(X2[1].T)),dtype=float)

target_ok5=np.zeros((len(nod),len(X2[1].T)),dtype=float)
x_inp=np.zeros((len(nod),len(X2[1].T)),dtype=float)
for i in tqdm(range(len(nod)-1 )):
   sss= np.concatenate((X2[int(trget[i][0])].T,X2[int(trget[i][1])].T,X2[int(trget[i][2])].T,X2[int(trget[i][3])].T,X2[int(trget[i][4])].T))
   target_ok1[i]=X2[int(trget[i][0])]
   target_ok2[i]=X2[int(trget[i][1])]
   target_ok3[i]=X2[int(trget[i][2])]
   target_ok4[i]=X2[int(trget[i][3])]
   target_ok5[i]=X2[int(trget[i][4])]
   x_inp[i]=X2[int(trget[i][2])]

#for v in Gnx.nodes():
 #    T2=softmax(X[int(v)].T)
  #   X[int(v)]=T2.T
#X2=preprocessing.quantile_transform(X2)
with open('WL_wikipedia.npy', 'wb') as f:
        np.save(f, np.array(X2))
 #============================================================================
for t in [128]:
  inpt_p=keras.Input(shape=(len(X2.T),))       
  encoded=layers.Dense(2000,activation='relu')(inpt_p)
  encoded1=layers.Dense(1000,activation='relu')(encoded)
  encoded2=layers.Dense(t,activation='relu')(encoded1)
#==================
  decod=layers.Dense(1000,activation='relu')(encoded2)
  decod=layers.Dense(2000,activation='relu')(decod)
  decod=layers.Dense(len(X2.T),activation='relu')(decod)
#==================
  decod1=layers.Dense(1000,activation='relu')(encoded2)
  decod1=layers.Dense(2000,activation='relu')(decod1)
  decod1=layers.Dense(len(X2.T),activation='relu')(decod1)
#==================
  decod2=layers.Dense(1000,activation='relu')(encoded2)
  decod2=layers.Dense(2000,activation='relu')(decod2)
  decod2=layers.Dense(len(X2.T),activation='relu')(decod2)
#==================
  decod3=layers.Dense(1000,activation='relu')(encoded2)
  decod3=layers.Dense(2000,activation='relu')(decod3)
  decod3=layers.Dense(len(X2.T),activation='relu')(decod3)
#==================
  decod4=layers.Dense(1000,activation='relu')(encoded2)
  decod4=layers.Dense(2000,activation='relu')(decod4)
  decod4=layers.Dense(len(X2.T),activation='relu')(decod4)
#==================
  decod_ok=keras.layers.Concatenate(axis=1)([decod, decod1,decod2,decod3,decod4])
#==================
  autoe=keras.Model(inpt_p,decod_ok)
  opt = SGD(lr=0.3, momentum=0.9, decay=0.01)
  autoe.compile(optimizer='adam',loss='mean_squared_error')
#==========================================================
  for k in range(50):
   btch=int(len(nod)/10000)
   for j in tqdm(range(1)):
     rng=  int(len(nod) % 10000)
     if (j<btch):
         rng=10000     
     target_ok=np.zeros((rng,5*len(X2[1].T)),dtype=float)
     x_inp=np.zeros((rng,len(X2[1].T)),dtype=float)
     for i in (range(rng )):
      sss= np.concatenate((X2[int(trget[j*10000+i][0])].T,X2[int(trget[j*10000+i][1])].T,X2[int(trget[j*10000+i][2])].T,X2[int(trget[j*10000+i][3])].T,X2[int(trget[j*10000+i][4])].T))
      target_ok[i]=sss.T
      x_inp[i]=X2[int(trget[j*10000+i][2])]
     autoe.fit(x_inp,target_ok,epochs=5, batch_size=100, shuffle=True)

#====================================classfacation
  encoder = keras.Model(inpt_p, encoded2)
  X_tr = encoder.predict(X2)
#===========================================
  from sklearn.metrics import  roc_auc_score
  for nn in [15]:
   p_tr=[]
   xxa=fb_df_partial.values.tolist()
   for i in  (range(len(fb_df_partial))):
    p_tr.append(np.hstack((X_tr[xxa[i][0]], X_tr[xxa[i][1]])))
    p_tr.append(np.hstack((X_tr[xxa[i][1]], X_tr[xxa[i][0]])))
   p_ts=[]
   y_t=[]
   xxa=fb_df_ghost.values.tolist()
   for i in  (range(len(fb_df_ghost))):
     p_ts.append(np.hstack((X_tr[xxa[i][0]], X_tr[xxa[i][1]])))
     y_t.append( xxa[i][2])
   xxa=data1.values.tolist()  
   poln=  len(fb_df_ghost)  
   if len(fb_df_ghost)>len(data1) :
      poln=len(data1)
   for i in  ((range(poln))):
     p_ts.append(np.hstack((X_tr[xxa[i][0]], X_tr[xxa[i][1]])))
     y_t.append( xxa[i][2])    
    #p_tr.append(np.hstack((X_tr[xxa[i][1]], X_tr[xxa[i][0]])))
   print('Merg',nn)
   tt=time.time()
   lof = LocalOutlierFactor(novelty=True,n_neighbors=nn, contamination=0.30)
   lof.fit(p_tr) 
   tt_f_M=  tt=time.time()-tt
   tt=time.time()
   y_pred=lof.predict(p_ts)
   #y_scor=lof.decision_function(p_ts) 
   s_tt_M=  tt=time.time()-tt
   print(confusion_matrix(y_t, y_pred))
   print(roc_auc_score(y_t, y_pred))
 # print(classification_report(y_t, y_pred))

#==============================================
   print("Avg")
   p_tr=[]
   xxa=fb_df_partial.values.tolist()
   for i in  (range(len(fb_df_partial))):
     p_tr.append(((X_tr[xxa[i][0]]+ X_tr[xxa[i][1]])/2))
     p_tr.append(((X_tr[xxa[i][1]]+ X_tr[xxa[i][0]])/2))
   p_ts=[]
   y_t=[]
   xxa=fb_df_ghost.values.tolist()
   for i in  (range(len(fb_df_ghost))):
     p_ts.append(((X_tr[xxa[i][0]]+ X_tr[xxa[i][1]])/2))
     y_t.append( xxa[i][2])
  
   xxa=data1.values.tolist()    
   for i in  (range(poln)):
     p_ts.append(((X_tr[xxa[i][0]]+ X_tr[xxa[i][1]])/2))
     y_t.append( xxa[i][2])    
    #p_tr.append(np.hstack((X_tr[xxa[i][1]], X_tr[xxa[i][0]])))

   tt=time.time()
   lof = LocalOutlierFactor(novelty=True,n_neighbors=nn, contamination=0.30)
   lof.fit(p_tr) 
   tt_f_a=  tt=time.time()-tt
   tt=time.time()
   y_pred=lof.predict(p_ts)
  #y_scor=lof.decision_function(p_ts) 
   s_tt_a=  tt=time.time()-tt
   print(confusion_matrix(y_t, y_pred))
   print(roc_auc_score(y_t, y_pred))
 # print(classification_report(y_t, y_pred))
#==============================================
   print("Had")
   p_tr=[]
   xxa=fb_df_partial.values.tolist()
   for i in  (range(len(fb_df_partial))):
     p_tr.append(((X_tr[xxa[i][0]]* X_tr[xxa[i][1]])))
    #p_tr.append(((X_tr[xxa[i][1]]* X_tr[xxa[i][0]])))
   p_ts=[]
   y_t=[]
   xxa=fb_df_ghost.values.tolist()
   for i in  (range(len(fb_df_ghost))):
     p_ts.append(((X_tr[xxa[i][0]]* X_tr[xxa[i][1]])))
     y_t.append( xxa[i][2])
   xxa=data1.values.tolist()    
   for i in  (range(poln)):
     p_ts.append(((X_tr[xxa[i][0]]* X_tr[xxa[i][1]])))
     y_t.append( xxa[i][2])    
    #p_tr.append(np.hstack((X_tr[xxa[i][1]], X_tr[xxa[i][0]])))
   tt=time.time()
   lof = LocalOutlierFactor(novelty=True,n_neighbors=nn, contamination=0.30)
   lof.fit(p_tr) 
   tt_f_h=  tt=time.time()-tt
   tt=time.time()
   y_pred=lof.predict(p_ts)
  #y_scor=lof.decision_function(p_ts) 
   s_tt_h=  tt=time.time()-tt
   print(confusion_matrix(y_t, y_pred))
   print(roc_auc_score(y_t, y_pred))
 # print(classification_report(y_t, y_pred))
#==============================================
   print("l1")
   p_tr=[]
   xxa=fb_df_partial.values.tolist()
   for i in  (range(len(fb_df_partial))):
    p_tr.append((np.abs(X_tr[xxa[i][0]]- X_tr[xxa[i][1]])))
    #p_tr.append(((X_tr[xxa[i][1]]* X_tr[xxa[i][0]])))
   p_ts=[]
   y_t=[]
   xxa=fb_df_ghost.values.tolist()
   for i in  (range(len(fb_df_ghost))):
    p_ts.append((np.abs(X_tr[xxa[i][0]]- X_tr[xxa[i][1]])))
    y_t.append( xxa[i][2])
   xxa=data1.values.tolist()    
   for i in  ((range(poln))):
    p_ts.append((np.abs(X_tr[xxa[i][0]]- X_tr[xxa[i][1]])))
    y_t.append( xxa[i][2])    
    #p_tr.append(np.hstack((X_tr[xxa[i][1]], X_tr[xxa[i][0]])))
   tt=time.time()
   lof = LocalOutlierFactor(novelty=True,n_neighbors=nn, contamination=0.30)
   lof.fit(p_tr) 
   tt_f_l1=  tt=time.time()-tt
   tt=time.time()
   y_pred=lof.predict(p_ts)
  #y_scor=lof.decision_function(p_ts) 
   s_tt_l1=  tt=time.time()-tt
   print(confusion_matrix(y_t, y_pred))
   print(roc_auc_score(y_t, y_pred))
 # print(classification_report(y_t, y_pred))
#==============================================
   print("l2")
   p_tr=[]
   xxa=fb_df_partial.values.tolist()
   for i in  (range(len(fb_df_partial))):
     p_tr.append(np.square(np.abs(X_tr[xxa[i][0]]- X_tr[xxa[i][1]])))
    #p_tr.append(((X_tr[xxa[i][1]]* X_tr[xxa[i][0]])))
   p_ts=[]
   y_t=[]
   xxa=fb_df_ghost.values.tolist()
   for i in  (range(len(fb_df_ghost))):
    p_ts.append(np.square(np.abs(X_tr[xxa[i][0]]- X_tr[xxa[i][1]])))
    y_t.append( xxa[i][2])
   xxa=data1.values.tolist()    
   for i in  ((range(poln))):
     p_ts.append(np.square(np.abs(X_tr[xxa[i][0]]- X_tr[xxa[i][1]])))
     y_t.append( xxa[i][2])    
    #p_tr.append(np.hstack((X_tr[xxa[i][1]], X_tr[xxa[i][0]])))
   tt=time.time()
   lof = LocalOutlierFactor(novelty=True,n_neighbors=nn, contamination=0.30)
   lof.fit(p_tr) 
   tt_f_l2=  tt=time.time()-tt
   tt=time.time()
   y_pred=lof.predict(p_ts)
  #y_scor=lof.decision_function(p_ts) 
   s_tt_l2=  tt=time.time()-tt
   print(confusion_matrix(y_t, y_pred))
   print("========================================")
   print(roc_auc_score(y_t, y_pred))
 # print(classification_report(y_t, y_pred))
        #=============================================================================


