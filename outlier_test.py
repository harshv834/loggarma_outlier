
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pylab
import math


# In[54]:

"""
def pearson_residual(Y,mu):
    return pd.DataFrame((Y-mu)/(mu**0.5),columns=['p'])
def greatest_subset(P):
    return P.sort_values(['p'],ascending=[False])[0:math.ceil(0.05*len(P))]
def log_lkhood(P,alpha):
    # P has y and mu as columns
    P['log-likelihood']=P['p']*math.log(alpha)+P['mu']*(1-alpha)
    return P.sort_values(['log-likelihood'],ascending=[False])
"""

def cal_mu(Beta, Phi, Theta, X, Y, mu):
    """
    X -> Observed X matrix NxD
    Y -> Observed Y
    mu -> Initial values of mu
    """
    P = len(Phi); Q = len(Theta)
    N = X.shape[0]
    mu = list(mu)
    t1 = []; t2 = []
    for i in range(max(P, Q)):
        t1.append(np.log(Y[i]) - np.dot(X[i], Beta))
        t2.append(np.log(Y[i]) - mu[i])
    for i in range(max(P, Q), N):
        mu_ = np.dot(X[i], Beta) + np.dot(Phi, t1[:(-P-1):-1]) + np.dot(Theta, t2[:(-Q-1):-1])
        mu.append(np.exp(mu_))
        if i<(N-1):
            t1.append(np.log(Y[i]) - np.dot(X[i+1], Beta))
            t2.append(np.log(Y[i]) - mu_)
    mu = np.array(mu)
    return mu

def cal_mu_Y(Beta, Phi, Theta, X, Y, mu):
    """
    X -> Observed X matrix NxD
    Y -> Only initial values: max(P, Q)
    mu -> Initial values of mu
    """
    P = len(Phi); Q = len(Theta)
    N = X.shape[0]
    mu = list(mu); Y = list(Y)
    t1 = []; t2 = []
    for i in range(max(P, Q)):
        t1.append(np.log(Y[i]) - np.dot(X[i], Beta))
        t2.append(np.log(Y[i]) - mu[i])
    for i in range(max(P, Q), N):
        mu_ = np.dot(X[i], Beta) + np.dot(Phi, t1[:(-P-1):-1]) + np.dot(Theta, t2[:(-Q-1):-1])
        mu.append(np.exp(mu_))
        x = np.random.poisson(np.exp(mu_))
        while x==0:
            x = np.random.poisson(np.exp(mu_))
        Y.append(x)
        if i<(N-1):
            t1.append(np.log(Y[i]) - np.dot(X[i+1], Beta))
            t2.append(np.log(Y[i]) - mu_)
    Y = np.array(Y)
    mu = np.array(mu)
    return Y, mu


####GIVEN Y
#1.FIT NULL MODEL on Y(Nx1 vector)( taking alpha=1 and learn remaining params) #rana,harsh
#2.CALCUATE mu(Nx1 vector) FROM OBTAINED PARAMETERS IN 1.#ravi
P = len(Phi); Q = len(Theta)
mu1 = cal_mu(Beta, Phi, Theta, X, Y, mu[:max(P, Q)]) # Y is the observed value, mu is from original data
def pearson_residual(Y, mu):
    return (Y-mu)/(mu**0.5)
k = int(N*0.05)
A = np.absolute(pearson_residual(Y, mu)).argsort()[:-k:-1]

#P=greatest_subset(pearson_residual(Y,mu))

#3. FIT ATERNATE MODEL ON P #rana,harsh
#4. CALCUATE mu_1 FROM OBTAINED PARAMETERS IN 3. #ravi
mu_1 = cal_mu(Beta, Phi, Theta, X, Y, mu[:max(P, Q)]) # Beta,Theta and Phi are obtained in prev step

P['mu']=mu_1
P=log_lkhood(P,alpha_1) # alpha_1 obtained in 3.
t0=P.index[0]
Tt0=P['log-likelihood'][0]*(-2)

def Test_stat(Y, mu, alpha):
    """
    Y -> Value of Y at t_o
    mu -> Value of mu at t_o
    alpha -> Estimated Alpha
    """
    return -2*(Y*np.log(alpha+(1e-16)) + mu*(1-alpha))
Test = np.array([Test_stat(Y[i], mu[i], alpha) for i in A])
print(Test)

M = 10
Test_boot = []
T_o = Test.argsort()[-1]
for i in range(M):
    mu = list(np.random.uniform(0,2, max(P, Q)))
    Y = []
    for i in range(max(P, Q)):
        x = np.random.poisson(np.exp(nu[i]))
        while x==0:
            x = np.random.poisson(np.exp(nu[i]))
        Y.append(x)
    Y_boot, Mu_boot = cal_mu_Y(Beta, Phi, Theta, X, Y, mu)
    Test_boot.append(Test_stat(Y_boot[T_o], Mu_boot[T_o], 1))
print(Test_boot)

#5. GENERATE N DATASETS USING PARAMS IN  1.    #BOOTSTRAPPING #Ravi
#6. for each dataset D in [1,N] do above and calcuate Tt0   #rohit
#7. Based on obtained D Tt0 values in 6., comparing Tt0 for original data, we decide wether t0 is outier or not #rohit

