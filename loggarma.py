import numpy as np
import scipy as sp
import statsmodels.api as sm
c=2
h=0.5

def loggarma(X,Y,p,q,max_iter,t_0):
    z = np.array([c]*Y.shape[0])
    Y_1 = np.maximum(Y,z)
    beta = [0]*(X.shape[1] - 1)
    beta.insert(0,np.log(Y.sum()/Y.shape[0]))
    beta = np.array(beta)
    phi = np.random.rand(p).reshape(-1,1)
    theta = np.random.rand(q).reshape(-1,1)
    alpha = 1
    eta = np.random.rand(Y.shape[0],1)
    deta_beta = np.random.rand(Y.shape[0],beta.shape[0])
    deta_phi = np.random.rand(Y.shape[0],p)
    deta_theta = np.random.rand(Y.shape[0],q)
    deta_alpha = np.random.rand(Y.shape[0],1)
    deta_alpha[t_0]= 1./alpha
    max_num = max(max(p,q),beta.shape[0])

    for i in range(max_iter):
        #Save old values 
        dold_beta = deta_beta
        dold_phi = deta_phi
        dold_theta = deta_theta
        dold_alpha = deta_alpha
        eta_old = eta
      
        #calculate new values of eta
        print(np.max(X), np.min(X))
        print('check')
        eta[:max_num] = np.log(Y[:max_num].reshape(-1,1))
        eta[max_num:] = np.dot(X[max_num:,:],beta).reshape(-1,1)
        print(np.max(X), np.min(X))
        print('check')
        
        eta[t_0] += np.log(alpha)
        
        for j in range(X.shape[0] - max_num):
            X_block_p = X[j:j + phi.shape[0],:].transpose()
            Y_block_q = Y[j:j + theta.shape[0]]
            Y_block_p = Y[j:j + phi.shape[0]]
            eta_block_q = eta_old[j:j+theta.shape[0]]
                        
            phi_block = np.log(np.flip(Y_block_p,axis=0).reshape(-1,1)) - np.dot(np.flip(X_block_p,axis=0).T,beta).reshape(-1,1)
            theta_block = np.log(np.flip(Y_block_q,axis=0).reshape(-1,1)) - eta_block_q
            eta[j + max_num] += (np.inner(phi_block.T,phi.T) + np.inner(theta_block.T,theta.T)).reshape(-1,)
            
        ##Check for convergence
        print(np.max(X), np.min(X))
        print('check')
##Update gradients
        deta_beta = X[:,0:beta.shape[0]]
        for j in range(X.shape[0] - phi.shape[0]):
            X_block = X[j:j + phi.shape[0],:]

            Y_block = Y[j:j+ phi.shape[0]]
            deta_beta[j+phi.shape[0],:] -= (np.inner(np.flip(phi,axis=0).T, X_block.T).T).reshape(-1,)
            deta_phi[j+phi.shape[0],:] = np.log(np.flip(Y_block,axis=0)) - np.dot(np.flip(X_block,axis=0),beta)
        print(np.max(X), np.min(X))
        print('final_check')
        for j in range(X.shape[0] - theta.shape[0]):
            # q-sized blocks of older gradients11
            dbeta_block = dold_beta[j:j+theta.shape[0],:].transpose()
            dphi_block = dold_phi[j:j+theta.shape[0],:].transpose()
            dtheta_block = dold_theta[j:j+theta.shape[0],:].transpose()

            # update after multiplying with current values of theta.
            Y_block_q = Y[j:j + theta.shape[0]]
            eta_block_q = eta[j:j + theta.shape[0]]
            deta_theta[j+theta.shape[0],:] = np.log(Y_block_q) - np.flip(eta_block_q.T,axis=0)
            deta_beta[j+theta.shape[0],:] -= (np.inner(np.flip(theta,axis=0).transpose(),dbeta_block).transpose()).reshape(-1)
            deta_phi[j+theta.shape[0],:] -= (np.inner(np.flip(theta,axis=0).transpose(),dphi_block).transpose()).reshape(-1)
            deta_theta[j+theta.shape[0],:] -= (np.inner(np.flip(theta,axis=0).transpose(),dtheta_block).transpose()).reshape(-1)
        
        deta_alpha = np.zeros([Y.shape[0],1])
        deta_alpha[t_0]= 1./alpha
            
   
#OLS minimization
        print(np.max(X), np.min(X))
        print('check')
        mu = np.exp(eta)
        mu = np.clip(mu, 0.1,10e30)
        
        print(max(mu), min(mu))
        
        R = np.dot(deta_beta,beta).reshape(-1,1) + np.dot(deta_phi,phi)+ np.dot(deta_theta,theta) + np.dot(deta_alpha,alpha)  + h*(Y.reshape(-1,1) - mu)*mu
                
        X_R = np.concatenate((deta_beta,deta_phi,deta_theta,deta_alpha),axis = 1)
        
        R = np.clip(R, -10e30,10e30)
        X_R = np.clip(X_R, -10e30,10e30)
        wls = sm.WLS(R,X_R,weights = mu, missing='drop')
        res_wls = wls.fit('qr')
        
        if (i+1)%10 == 0:
            print('iteration: ', i)
            print(res_wls.params)
            
        if np.sum(np.isnan(res_wls.params)) == 0:
            print(res_wls.params)
            
            beta = res_wls.params[:beta.shape[0]]
            phi = res_wls.params[beta.shape[0]:p+beta.shape[0]]
            theta = res_wls.params[beta.shape[0]+p:beta.shape[0]+p+q]
            alpha = float(res_wls.params[-1])
        
    return beta,phi,theta, alpha