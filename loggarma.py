import numpy as np
import scipy as sp
import statsmodel as sm
c = 2

def loggarma(X,Y,p,q,max_iter,t_0):
    Y_1 = np.max(Y,[c]*Y.shape[0])
    beta = [0]*(X.shape[1] - 1)
    beta.insert(0,np.log(Y.sum()))
    beta = np.array(beta)
    phi = np.random.rand(p)
    theta = np.random.rand(q)
    alpha = 1
    eta = np.zeros([Y.shape[0],1])
    deta_beta = np.zeros([Y.shape[0],beta.shape[0]])
    deta_phi = np.zeros([Y.shape[0],p])
    deta_theta = np.zeros([Y.shape[0],q])
    deta_alpha = np.zeros([Y.shape[0],1])
    deta_alpha[t_0]= 1./alpha

    for i in range(max_iter):
        for j in range(deta_beta):
            if j > 1:
                deta_
    
    
