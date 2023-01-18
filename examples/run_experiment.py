'''
This example shows how to run experiments of the decision branch models (or sklearn tree models) 
on specific datasets. We use an own wrapper function 'run_experiment' that takes care of the 
whole experimental pipeline.
'''

from decisionbranches.utils.train_pipeline import run_experiment

model_type = "dbranch"

'''
param_grid defines hyperparameter to be optimized 
before each run in a grid search (in brackets the choices are listed)
'''
param_grid = {"max_features":["auto",0.5,0.75,"all"]}

'''
model_params contain the model parameter that can be compared during the experiments.
Only one parameter can be given as list to compare the influence of different values on
the final outcome.
'''
model_params = {"n_jobs":1,"n_ind":[0.25,0.5,1],"n_feat":2,"verbose":False}

'''
experiment_params are parameters concerning the experimental setting. 
'''
experiment_params = {"n_rare":30,"val_set":True,"n_rare_val":0.5,"scaling":True,"n_seeds":3,"val_criterion":"f1"}

res = run_experiment(datasets={"iris":{"labels":"all"}}, model=model_type,model_params=model_params,
                     experiment_params=experiment_params,
                     param_grid=param_grid,seed=10,progress_bar=False)