'''
This file contains the parameter configurations for all tested methods in the paper during the model benchmark.
They can simply be fed into the run_experiment function.
'''



from decisionbranches.utils.train_pipeline import run_experiment

####################### DBranch [B,10] ######################
model_type = "dbranch"

param_grid = {"top_down":[False],"max_nbox":[1000],"min_pts":["auto"],
            "del_nonrare":[True,False],"splitter":["half","random","max","min"],
            "max_features":["auto",0.5,0.75,"all"]}


model_params = {"n_jobs":1,"n_feat":10,"n_ind":10,"postTree": False,"verbose":False,"debug":False}



##################### DBranch [B,T_s,10] ###########################
model_type = "dbranch"

param_grid = {"top_down":[True],"max_nbox":[1000],"min_pts":["auto"],"del_nonrare":[True,False],
            "splitter":["half","random","max","min"],"max_features":["auto",0.5,0.75,"all"]}

model_params = {"n_jobs":1,"n_feat":10,"n_ind":10,"postTree": False,"verbose":False,"debug":False}

##################### DBranch [B,T_a,10] ###########################
model_type = "dbranch"

param_grid = {"top_down":[False],"max_nbox":[1000],"min_pts":["auto"],"stop_infinite":[True,False],
            "del_nonrare":[True],"splitter":["half","random","max","min"],"max_features":["auto",0.5,0.75,"all"]}

model_params = {"n_jobs":1,"n_feat":10,"n_ind":10,"postTree": True,"verbose":False,"debug":False}


#################### DBEns[B,25t,10] ###########################
model_type = "ensemble"

param_grid = {"top_down":[False],"max_nbox":[1000],"min_pts":["auto"],"n_estimators":[25],"bootstrap":[True,False],
                  "del_nonrare":[True,False],"splitter":["half","random"],"max_features":["auto",0.5,0.75,"all"]}
    
model_params = {"n_jobs":25,"n_feat":10,"n_ind":10,"postTree": False,"stop_infinite":False,"verbose":False}


#################### DBEns[B,T_s,25t,10] ###########################

model_type = "ensemble"

param_grid = {"top_down":[True],"max_nbox":[1000],"min_pts":["auto"],"n_estimators":[25],"bootstrap":[True,False],
                  "del_nonrare":[True,False],"splitter":["half","random"],"max_features":["auto",0.5,0.75,"all"]}
    
model_params = {"n_jobs":25,"n_feat":10,"n_ind":10,"postTree": False,"stop_infinite":False,"verbose":False}


#################### DBEns[B,T_a,25t,10] ###########################
model_type = "ensemble"

param_grid = {"top_down":[False],"max_nbox":[1000],"min_pts":["auto"],"n_estimators":[25],"bootstrap":[True,False],
                "del_nonrare":[True,False],"splitter":["half","random"],"max_features":["auto",0.5,0.75,"all"]}
    
model_params = {"n_jobs":25,"n_feat":10,"n_ind":10,"postTree": True,"stop_infinite":False,"verbose":False}

#################### DTree ###########################

model_type = "dtree"

param_grid = {'criterion': ["gini", "entropy"], 'splitter': ["random", "best"],
            "min_samples_leaf":list(range(1,10,2)),"max_features":["auto",0.5,0.75,"all"]}

model_params = {}

#################### DTree[4] ###########################

model_type = "lim_dtree"

param_grid = {'criterion': ["gini", "entropy"], 'splitter': ["random", "best"],
                "min_samples_leaf":list(range(1,10,2)),"max_features":["auto",0.5,0.75,"all"]}

model_params = {"max_global_features":[4]}

#################### DTree[10] ###########################

model_type = "lim_dtree"

param_grid = {'criterion': ["gini", "entropy"], 'splitter': ["random", "best"],
                "min_samples_leaf":list(range(1,10,2)),"max_features":["auto",0.5,0.75,"all"]}

model_params = {"max_global_features":[10]}

#################### RF ###########################

model_type = "rf"

param_grid = {"n_estimators":list(range(25,101,25)),"bootstrap":[True,False],
            'criterion': ["gini", "entropy"],"max_features":["auto",0.5,0.75,"all"]}

model_params = {"n_jobs":25}

#################### RForest[4] ###########################

model_type = "lim_rf"

param_grid = {"n_estimators":list(range(25,101,25)),"bootstrap":[True,False],
            'criterion': ["gini", "entropy"],"max_features":["auto",0.5,0.75,"all"]}

model_params = {"max_global_features":[4],"n_jobs":25}


#################### RForest[10] ###########################

model_type = "lim_rf"

param_grid = {"n_estimators":list(range(25,101,25)),"bootstrap":[True,False],'criterion': ["gini", "entropy"],"max_features":["auto",0.5,0.75,"all"]}

model_params = {"max_global_features":[10],"n_jobs":25}


#################### ExTrees ###########################

model_type = "extra"

param_grid = {"n_estimators":list(range(25,101,25)),"max_features":["auto",0.5,0.75,"all"]}

model_params = {"n_jobs":25}


##################### Experimental Parameters ####################################################

experiment_params = {"n_rare":30,"val_set":True,"n_rare_val":0.5,"scaling":True,"n_seeds":3,"val_criterion":"f1"}

############################# Example ###########################################

# Examplary call of the run_experiment for all given datasets
res = run_experiment(datasets = "all", model=model_type,model_params=model_params,
                     experiment_params=experiment_params,
                     param_grid=param_grid,seed=10,progress_bar=False)

