import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn import tree

from FhteF_models import *

def experiment(flag,max_depth=2,T=100,rand_stat=12):
    np.random.seed(rand_stat)
    if flag == 1:
        name = 'simulated_Athey_1'
        n = 50000
        n_test=50000
        p=2
        K=2#number of treatments
        nu = lambda x: 0.5*x[:,0]+x[:,1]
        kf = lambda x: 0.5*x[:,0]  
        X=np.random.normal(0,1,(n,p))
        X_test=np.random.normal(0,1,(n_test,p)) 
        W=np.random.binomial(1, 0.5,n)
        W_test=np.random.binomial(1, 0.5,n_test)
        Y=nu(X)+0.5*(2*W-1)*kf(X) + np.random.normal(0,0.01,n)
        Y_test=nu(X_test)+0.5*(2*W_test-1)*kf(X_test) + np.random.normal(0,0.01,n_test)
        W=pd.get_dummies(W, prefix='W').values
        W_test=pd.get_dummies(W_test, prefix='W').values
        tau_truth = kf(X_test)

    if flag == 2:
        name = 'simulated_Athey_2'
        n = 50000
        n_test=50000
        p=10
        K=2      
        nu = lambda x: 0.5*(x[:,0]+x[:,1])+x[:,2]+x[:,3]+x[:,4]+x[:,5]
        kf = lambda x: np.where(x[:,0]>= 0,x[:,0],0)+np.where(x[:,1]>= 0,x[:,1],0)
        X=np.random.normal(0,1,(n,p))
        X_test=np.random.normal(0,1,(n_test,p)) 
        W=np.random.binomial(1, 0.5,n)
        W_test=np.random.binomial(1, 0.5,n_test)
        Y=nu(X)+0.5*(2*W-1)*kf(X) + np.random.normal(0,0.01,n)
        Y_test=nu(X_test)+0.5*(2*W_test-1)*kf(X_test) + np.random.normal(0,0.01,n_test)
        W=pd.get_dummies(W, prefix='W').values
        W_test=pd.get_dummies(W_test, prefix='W').values
        tau_truth = kf(X_test)

    if flag == 3:
        name = 'simulated_Athey_3'
        n = 50000
        n_test=50000
        p=20
        K=2          
        nu = lambda x: 0.5*(x[:,0]+x[:,1]+x[:,2]+x[:,3])+x[:,4]+x[:,5]+x[:,6]+x[:,7]
        kf = lambda x: np.where(x[:,0]>= 0,x[:,0],0)+np.where(x[:,1]>= 0,x[:,1],0)+np.where(x[:,2]>= 0,x[:,2],0)+np.where(x[:,3]>= 0,x[:,3],0)
        X=np.random.normal(0,1,(n,p))   
        X_test=np.random.normal(0,1,(n_test,p)) 
        W=np.random.binomial(1, 0.5,n)
        W_test=np.random.binomial(1, 0.5,n_test)
        Y=nu(X)+0.5*(2*W-1)*kf(X) + np.random.normal(0,0.01,n)
        Y_test=nu(X_test)+0.5*(2*W_test-1)*kf(X_test) + np.random.normal(0,0.01,n_test)
        W=pd.get_dummies(W).values
        W_test=pd.get_dummies(W_test, prefix='W').values
        tau_truth = kf(X_test)
    
    min_samples_leaf=50
    RF = RandomForestRegressor(random_state=rand_stat,n_estimators=T, criterion='mse', min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_leaf, max_depth=max_depth,max_features=2)
    RF.fit(X, Y)
    t_list = RF.estimators_
    T = len(t_list)
    
    max_l = 0 #min number of leaves
    min_l = 100 #max number of leaves
    for t in t_list: 
        leaf_id = t.apply(X)
        if leaf_id.min()<min_l:
            min_l = leaf_id.min()
        if leaf_id.max()>max_l:
            max_l = leaf_id.max()
    tau = np.zeros((K,max_l+1,T))#dif of means
    tau_reg = np.zeros((K-1,max_l+1,T))#lin reg in the leaf
    
    y_hat_OLS =  np.zeros((T,n)) 
    y_hat = np.zeros((K,T,n))
    y_hat_test = np.zeros((K,T,n_test))

    tau_test = np.zeros((K,n_test))
    tau_reg_test = np.zeros((K-1,n_test))
    
    leaves = np.zeros((T,n),dtype=int)
    leaves_test = np.zeros((T,n_test),dtype=int)
    
    ub = 0
    lb = 1000
    good_trees=[]
    print('First step: build an ensemble')
    for j,t in enumerate(t_list):   
        leaf_id = t.apply(X)
        leaves[j,:]=leaf_id
        good=1
        for l in range(leaf_id.min(),leaf_id.max()+1):
            idx = np.where(leaf_id==l)[0] 
            if (good>0) & (len(idx)>0) & np.prod([sum(W[idx,i])>=10 for i in range(K)]).astype('bool'):                  
                reg = LinearRegression().fit(np.hstack((W[idx,K-1:K],X[idx])), Y[idx])
                tau_reg[:,l,j]=reg.coef_[0]
                y_hat_OLS[j,idx] =reg.predict(np.hstack((W[idx,K-1:K],X[idx])))
                for k in range(K):
                    tau[k,l,j]=sum(Y[idx]*W[idx,k])/sum(W[idx,k])
                    y_hat[k,j,idx] = reg.predict(np.hstack((np.ones((len(idx),1))*k,X[idx])))
                if ub <= np.max(reg.coef_):
                    ub = np.max(reg.coef_)
                if lb >= np.min(reg.coef_):
                    lb = np.min(reg.coef_)
            else:
                if len(idx)>0:
                    good=0
        if good:
            good_trees.append(j)         
            leaf_id_test = t.apply(X_test)
            leaves_test[j,:]=leaf_id_test
            for i in range(n_test):
                for k in range(K):
                    y_hat_test[k,j,i] = tau[k,leaf_id_test[i],j]
                    if k > 0:
                        tau_test[k-1,i] = tau_test[k-1,i]+tau[k,leaf_id_test[i],j]-tau[0,leaf_id_test[i],j]
                        tau_reg_test[k-1,i] = tau_reg_test[k-1,i]+tau_reg[k-1,leaf_id_test[i],j]
    
    T = len(good_trees) #exclude trees which does not meet requirements
    leaves = leaves[good_trees,:]
    leaves_test = leaves_test[good_trees,:]
    leaf_label = list(np.unique(leaves))
    tau_test=tau_test/T
    tau_reg_test=tau_reg_test/T
    print('Second step: reweight the ensemble')
    r_wOLS = MSE_of_OLS(T,Y,y_hat_OLS[good_trees,:])
    mytau_test_wOLS= prediction_OLS(tau_reg[0,leaf_label,:].transpose(),leaf_label,leaves_test,weight=np.where(np.array(r_wOLS.get('w'))<=0.000001,0,r_wOLS.get('w')))
    res={'test':np.linalg.norm(tau_truth-mytau_test_wOLS,2)/n_test}

    return res
