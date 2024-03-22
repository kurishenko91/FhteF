import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import gurobipy as gu

def cqQ(y,y_hat,I): 
    q = 2*y@y_hat.transpose()/I
    Q = y_hat@y_hat.transpose()/I
    c = y@y/I
    
    return c,q,Q
"""
The FhteF model without fairness term
input:
T number of trees
Y training outcome
y_hat predicted outcom in each tree
reg regularization parameter 
output:
w weights of the FhteF
"""
def MSE_of_OLS(T,Y,y_hat,reg=0,TimeLimit = None,w_start=None, max_weight=None):
    m = gu.Model("Weighted Tree")
    w = m.addMVar(T, name = 'w', vtype = gu.GRB.CONTINUOUS, lb = 0, ub = 1)   
    c,q,Q = cqQ(Y,y_hat,len(Y)) 
    if reg != 0:
        reg_M = np.zeros((T,T))
        np.fill_diagonal(reg_M, reg)
        qexp = c-w@q+w@Q@w+w@reg_M@w
    else:
        qexp = c-w@q+w@Q@w
    if w_start is not None:
        for i,t in enumerate(w_start):
            w[t].start = w_start[i]
    m.setObjective(qexp, gu.GRB.MINIMIZE)              
    m.addConstr(sum(w[t] for t in range(T)) == 1)
    if max_weight is not None:
        m.addConstrs(w[t] >= max_weight for t in range(T))
    m.Params.OutputFlag = 0
    m.Params.Seed = 12
    if TimeLimit is not None:
        m.Params.TimeLimit = TimeLimit
#    m.Params.NonConvex=2
    try:
        m.optimize()
        tolerance = False
    except:
        add_M = np.zeros((T,T))
        np.fill_diagonal(add_M, 0.000001)
        additional = w@add_M@w
        m.setObjective(qexp+additional, gu.GRB.MINIMIZE) 
        m.optimize()
        tolerance = True
        
    
    try:
        var = [v.x for v in m.getVars()]
        w = var[:T]
        obj = m.objVal
        stat = m.status
        runtime = m.runtime
        return {'w':w,'stat':stat, 'runtime':runtime, 'obj':obj, 'tolerance':tolerance}
    except:
        stat = m.status
        return {'stat':stat,'tolerance':tolerance}

"""
The FhteF model without fairness term
input:
T number of trees
Y training outcome
y_hat predicted outcom in each tree
beta_w predicted treatment effect in each leave
leaf_label labels of the obtained leaves
leaves a leaf for each individual for each tree
I_hat sensititve individuals
alpha weight of the fairness term
reg regularization parameter 
output:
w weights of the FhteF
"""
def MSE_of_OLS_fair(T,I,Y,y_hat,beta_w,leaf_label,leaves,I_hat,alpha,reg=0,TimeLimit = None, w_start=None, max_weight=None):
    K=beta_w.shape[2]
    m = gu.Model("Weighted Tree")
    w = m.addMVar(T, name = 'w', vtype = gu.GRB.CONTINUOUS, lb = 0, ub = 1)  
    mod = m.addMVar(1,name = 'mod', vtype = gu.GRB.CONTINUOUS, lb = 0)
    c,q,Q = cqQ(Y,y_hat,len(Y)) 
    if reg != 0:
        reg_M = np.zeros((T,T))
        np.fill_diagonal(reg_M, reg)
        qexp = c-w@q+w@Q@w+w@reg_M@w
    else:
        qexp = c-w@q+w@Q@w
    if w_start is not None:
        for i,t in enumerate(w_start):
            w[t].start = w_start[i]

    qexp = qexp + alpha*mod
    m.setObjective(qexp, gu.GRB.MINIMIZE)              
    m.addConstr(sum(w[t] for t in range(T)) == 1)
    for k in range(K):
        m.addConstr(sum(w[t]*(sum(beta_w[t,leaf_label.index(leaves[t,i]),k] for i in I_hat)/len(I_hat) - sum(beta_w[t,leaf_label.index(leaves[t,i]),k] for i in set(range(I))-set(I_hat))/(I-len(I_hat))) for t in range(T)) <= mod) 
        m.addConstr(sum(w[t]*(sum(beta_w[t,leaf_label.index(leaves[t,i]),k] for i in I_hat)/len(I_hat) - sum(beta_w[t,leaf_label.index(leaves[t,i]),k] for i in set(range(I))-set(I_hat))/(I-len(I_hat))) for t in range(T)) >= -mod) 
    if max_weight is not None:
        m.addConstrs(w[t] >= max_weight for t in range(T))
    m.Params.OutputFlag = 0
    m.Params.Seed = 12
    if TimeLimit is not None:
        m.Params.TimeLimit = TimeLimit
    try:
        m.optimize()
        tolerance = False
    except:
        add_M = np.zeros((T,T))
        np.fill_diagonal(add_M, 0.000001)
        additional = w@add_M@w
        m.setObjective(qexp+additional, gu.GRB.MINIMIZE) 
        m.optimize()
        tolerance = True      
   
    try:
        var = [v.x for v in m.getVars()]
        w = var[:T]
        mod = var[T]
        obj = m.objVal
        stat = m.status
        runtime = m.runtime
        return {'w':w, 'mod': mod,'stat':stat, 'runtime':runtime, 'obj':obj, 'tolerance':tolerance}
    except:
        stat = m.status
        return {'stat':stat,'tolerance':tolerance}

"""
Predicted treatment effect 
beta_w predicted treatment effect in each leave
leaf_label labels of the obtained leaves
leaves a leaf for each individual for each tree
"""   
def prediction_OLS(beta_w,leaf_label,leaves_test, weight = None):
    T = leaves_test.shape[0]
    n_test = leaves_test.shape[1]
    tau_test_t = np.zeros((T,n_test))
    for t in range(T):
        for i in range(n_test):
            tau_test_t[t,i]=beta_w[t,leaf_label.index(leaves_test[t,i])]
    if weight is None:
        weight=np.ones(T)/T
    return np.dot(weight,tau_test_t)

