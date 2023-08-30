# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:34:13 2023

@author: amr AL-furas
"""
from sklearn import svm
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
from sklearn.metrics import  f1_score
from karateclub.utils.walker import RandomWalker
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
  for i in[20]:
    print('test with pers ' ,k ,'\n')
    X_train, X_test, y_train, y_test = train_test_split(X2, target, test_size=i/100) 
    encoder = keras.Model(inpt_p, encoded2)
    encoded_tr = encoder.predict(X_train)
    encoded_ts = encoder.predict(X_test)
    svclassifier = svm.SVC(kernel='rbf',gamma=.001,C=3)
    svclassifier.fit(encoded_tr, y_train)
    y_pred = svclassifier.predict(encoded_ts)
   # print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    print(t,'-micro=',f1_score(y_test,y_pred,average='micro'))
    print(t,'-macro=',f1_score(y_test,y_pred,average='macro'))
#=============================================================================


