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
        dold_beta = deta_beta
        dold_phi = deta_phi
        dold_theta = deta_theta
        dold_alpha = deta_alpha

        ##Check for convergence

##Update gradients
        deta_beta = X[:,0:beta.shape[0]]
        for j in range(X.shape[0]):
            if range
            deta_beta[j:,j] =deta_beta[j:,j] - np.dot(beta,deta[])
    
        
        mu = np.exp(eta)
        R = np.dot(deta_beta,beta) + np.dot(deta_phi,phi)+ np.dot(deta_theta,theta) + np.dot(deta_alpha,alpha)  + h*(Y - mu)*mu
        X_R = np.concatenate((deta_beta,deta_phi,deta_theta,deta_alpha),axis = 1)
        wls = sm.api.WLS(R,X_R,weights = mu)
        res_wls = mod_wls.fit()
        beta = res_wls.params[:beta.shape[0]]
        phi = res_wls.params[beta.shape[0]:p+beta.shape[0]]
        theta = res_wls.params[beta.shape[0]+p:beta.shape[0]+p+q]
        alpha = int(res_wls.params[end])
        
