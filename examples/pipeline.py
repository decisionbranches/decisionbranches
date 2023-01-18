import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from decisionbranches.utils.helpers import generate_fidxs
from decisionbranches.models.boxSearch.boxClassifier import BoxClassifier
from py_kdtree.treeset import KDTreeSet

seed=42
np.random.seed(seed)


#Parameter
nfeat = 10
nind = 100
dbranch_cfg = {"top_down":False,"max_evals":"all","stop_infinite":True}

label = "4."

X, y = fetch_openml('satimage', version=1, return_X_y=True, as_frame=False)

y_bin = np.zeros(len(y),dtype=int)
y_bin[y==label] = 1

X_train,X_test,y_train,y_test = train_test_split(X,y_bin,train_size=0.05,random_state=seed)
print("Number of rare training objects: ",np.sum(y_train))
print("Number of points to query: ",len(X_test))


#Generate feature subsets
subsets = generate_fidxs(n_feat=nfeat,n_ind=nind,feats=np.arange(X.shape[1]),seed=seed)

##### Create indexes #######
treeset = KDTreeSet(subsets,path="./indexes/",leaf_size=60,verbose=False)
treeset.fit(X_test)

###### DecisionBranch ######### 
dbranch = BoxClassifier(tot_feat=X.shape[1],n_feat=nfeat,n_ind=nind,cfg=dbranch_cfg,postTree=False)

dbranch.fit(X_train,y_train)

mins,maxs,fidxs = dbranch.get_boxes()

#### Query boxes #########
inds,counts,time,loaded_leaves = treeset.multi_query_ranked_cy(mins,maxs,fidxs)

print("Number of found points: ",len(inds))
print("Loading time: ",time)
print("Number of loaded leaves: ",loaded_leaves)
