import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn import tree

from FhteF_models import *
   
def experiment_fair(flag,max_depth=2,T=100,alpha_list=[1],rand_stat=12):
    np.random.seed(rand_stat)
    if flag == 1:
        name = 'simulated_fair1'

        n = 50000
        n_test=50000
        p=2
        K=2#number of treatments
        nu = lambda x: 0.5*x[:,0]+x[:,1]
        kf = lambda x,z: 0.5*x[:,0]+z  
        X=np.random.normal(0,1,(n,p))
        X_test=np.random.normal(0,1,(n_test,p)) 
        
        W=np.random.binomial(1, 0.5,n)
        W_test=np.random.binomial(1, 0.5,n_test)

        Z=np.random.binomial(1, np.where(X[:,0]+X[:,1]>=0,0.8,0.2),n)
        Z_test=np.random.binomial(1, np.where(X_test[:,0]+X_test[:,1]>=0,0.8,0.2),n_test)     

        Y=nu(X)+0.5*(2*W-1)*kf(X,Z) + np.random.normal(0,0.01,n)
        Y_test=nu(X_test)+0.5*(2*W_test-1)*kf(X_test,Z_test) + np.random.normal(0,0.01,n_test)
        W=pd.get_dummies(W, prefix='W').values
        W_test=pd.get_dummies(W_test, prefix='W').values
        tau_truth = kf(X_test,Z_test)
        X = np.hstack([X,Z.reshape((n,1))])
        X_test = np.hstack([X_test,Z_test.reshape((n_test,1))])
        I_hat = np.where(Z)[0]
        I_hat_test = np.where(Z_test)[0]
        I_hat_not_test = list(set(range(n_test))-set(I_hat_test))
        p=3
        sens_feature_name=2

    if flag == 2:
        name = 'simulated_fair2'
        n = 50000
        n_test=50000
        p=10
        K=2
                    
        nu = lambda x: 0.5*(x[:,0]+x[:,1])+x[:,2]+x[:,3]+x[:,4]+x[:,5]
        kf = lambda x,z: np.where(x[:,0]>= 0,x[:,0],0)+np.where(x[:,1]>= 0,x[:,1],0)+z
        X=np.random.normal(0,1,(n,p))
        X_test=np.random.normal(0,1,(n_test,p)) 
        
        W=np.random.binomial(1, 0.5,n)
        W_test=np.random.binomial(1, 0.5,n_test)
        
        Z=np.random.binomial(1, np.where(X[:,0]+X[:,1]+X[:,2]+X[:,3]>=0,0.8,0.2),n)
        Z_test=np.random.binomial(1, np.where(X_test[:,0]+X_test[:,1]+X_test[:,2]+X_test[:,3]>=0,0.8,0.2),n_test)     

        Y=nu(X)+0.5*(2*W-1)*kf(X,Z) + np.random.normal(0,0.01,n)
        Y_test=nu(X_test)+0.5*(2*W_test-1)*kf(X_test,Z) + np.random.normal(0,0.01,n_test)
        W=pd.get_dummies(W, prefix='W').values
        W_test=pd.get_dummies(W_test, prefix='W').values

        tau_truth = kf(X_test,Z_test)
        X = np.hstack([X,Z.reshape((n,1))])
        X_test = np.hstack([X_test,Z_test.reshape((n_test,1))])
        I_hat = np.where(Z)[0]
        I_hat_test = np.where(Z_test)[0]
        I_hat_not_test = list(set(range(n_test))-set(I_hat_test))

        p=11
        sens_feature_name=10


    if flag==3:
        name = 'simulated_fair3'

        n = 50000
        n_test=50000
        p=20
        K=2
            
        nu = lambda x: 0.5*(x[:,0]+x[:,1]+x[:,2]+x[:,3])+x[:,4]+x[:,5]+x[:,6]+x[:,7]
        kf = lambda x,z: np.where(x[:,0]>= 0,x[:,0],0)+np.where(x[:,1]>= 0,x[:,1],0)+np.where(x[:,2]>= 0,x[:,2],0)+np.where(x[:,3]>= 0,x[:,3],0)+z
        
        X=np.random.normal(0,1,(n,p))   
        X_test=np.random.normal(0,1,(n_test,p))       
        W=np.random.binomial(1, 0.5,n)
        W_test=np.random.binomial(1, 0.5,n_test)
        Z=np.random.binomial(1, np.where(X[:,0]+X[:,1]+X[:,2]+X[:,3]+X[:,4]+X[:,5]+X[:,6]>=0,0.8,0.2),n)
        Z_test=np.random.binomial(1, np.where(X_test[:,0]+X_test[:,1]+X_test[:,2]+X_test[:,3]+X_test[:,4]+X_test[:,5]+X_test[:,6]>=0,0.8,0.2),n_test)
        Y=nu(X)+0.5*(2*W-1)*kf(X,Z) + np.random.normal(0,0.01,n)
        Y_test=nu(X_test)+0.5*(2*W_test-1)*kf(X_test,Z) + np.random.normal(0,0.01,n_test)
        W=pd.get_dummies(W).values
        W_test=pd.get_dummies(W_test, prefix='W').values
        tau_truth = kf(X_test,Z_test)
        X = np.hstack([X,Z.reshape((n,1))])
        X_test = np.hstack([X_test,Z_test.reshape((n_test,1))])
        I_hat = np.where(Z)[0]
        I_hat_test = np.where(Z_test)[0]
        I_hat_not_test = list(set(range(n_test))-set(I_hat_test))
        p=21
        sens_feature_name=20
        
    min_samples_leaf=50
    RF = RandomForestRegressor(random_state=rand_stat,n_estimators=T, criterion='mse', min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_leaf, max_depth=max_depth,max_features='sqrt')
    RF.fit(X, Y)
    t_list = RF.estimators_
    T = len(t_list)
    
    max_l = 0
    min_l = 100
    for t in t_list: 
        leaf_id = t.apply(X)
        if leaf_id.min()<min_l:
            min_l = leaf_id.min()
        if leaf_id.max()>max_l:
            max_l = leaf_id.max()
    tau_reg = np.zeros((K-1,max_l+1,T))#lin reg in the leaf  
    y_hat_OLS =  np.zeros((T,n))
    
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
        if True:#sens_feature_name in set(t.tree_.feature):
            for l in range(leaf_id.min(),leaf_id.max()+1):
                idx = np.where(leaf_id==l)[0] 
                if (good>0) & (len(idx)>0) & np.prod([sum(W[idx,i])>=10 for i in range(K)]).astype('bool'):
                    reg = LinearRegression().fit(np.hstack((W[idx,1:],X[idx])), Y[idx])
                    tau_reg[:,l,j]=reg.coef_[:K-1]
                    y_hat_OLS[j,idx] =reg.predict(np.hstack((W[idx,1:],X[idx])))
                    if ub <= np.max(reg.coef_):
                        ub = np.max(reg.coef_)
                    if lb >= np.min(reg.coef_):
                        lb = np.min(reg.coef_)
                else:
                    if len(idx)>0:
                        good=0
        else:
            good=0
        if good:
            good_trees.append(j)
            
            leaf_id_test = t.apply(X_test)
            leaves_test[j,:]=leaf_id_test
    T = len(good_trees)
    leaves = leaves[good_trees,:]
    leaves_test = leaves_test[good_trees,:]
    leaf_label = list(np.unique(leaves))
    res=[]
    print('Second step: reweight the ensemble')
    for alpha in alpha_list:
        r_wOLS_cs = MSE_of_OLS_fair(T,n,Y,y_hat_OLS[good_trees,:],tau_reg[:,leaf_label,:].transpose(),leaf_label,leaves,I_hat,reg=0.01,alpha=alpha)
        mytau_test_wOLS_cs= prediction_OLS(tau_reg[0,leaf_label,:].transpose(),leaf_label,leaves_test,weight=np.where(np.array(r_wOLS_cs.get('w'))<=0.000001,0,r_wOLS_cs.get('w')))     
        res.append({'test':[alpha,np.round(np.linalg.norm(tau_truth-mytau_test_wOLS_cs,2)/n_test,4),np.round(np.abs(np.mean(mytau_test_wOLS_cs[I_hat_test])-np.mean(mytau_test_wOLS_cs[I_hat_not_test])),4),np.round(np.abs(np.mean(tau_truth[I_hat_test])-np.mean(tau_truth[I_hat_not_test])),4)]})
        mytau_wOLS_cs= prediction_OLS(tau_reg[0,leaf_label,:].transpose(),leaf_label,leaves,weight=np.where(np.array(r_wOLS_cs.get('w'))<=0.000001,0,r_wOLS_cs.get('w')))
        res.append({'train':[alpha,np.round(np.linalg.norm(kf(X,Z)-mytau_wOLS_cs,2)/n,4),np.round(np.abs(np.mean(mytau_wOLS_cs[I_hat])-np.mean(mytau_wOLS_cs[list(set(range(n))-set(I_hat))])),4),np.round(np.abs(np.mean(kf(X,Z)[I_hat])-np.mean(kf(X,Z)[list(set(range(n))-set(I_hat))])),4)]})

    return res       
