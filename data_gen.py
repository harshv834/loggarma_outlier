import numpy as np
from sklearn.datasets import make_spd_matrix

N = 1000
D = 10
mu = np.random.uniform(-1, 1, D)
Sigma = make_spd_matrix(D)

from scipy.stats import multivariate_normal
X = multivariate_normal.rvs(mu, Sigma, N)

from random import randint
P = 3; Q = 4
t_o = randint(max(P, Q), N)

Beta = np.random.uniform(-1, 1, D)
Phi = np.random.uniform(-1, 1, P)
Theta = np.random.uniform(-1, 1, Q)

alpha = 2
t1 = []; t2 = []
nu = list(np.random.uniform(0,2, max(P, Q)))
Y = []
for i in range(max(P, Q)):
    x = np.random.poisson(np.exp(nu[i]))
    while x==0:
        x = np.random.poisson(np.exp(nu[i]))
    Y.append(x)
    t1.append(np.log(Y[i]) - np.dot(X[i], Beta))
    t2.append(np.log(Y[i]) - nu[i])
 
for i in range(max(P, Q), N):
    nu_ = np.dot(X[i], Beta) + np.dot(Phi, t1[:(-P-1):-1]) + np.dot(Theta, t2[:(-Q-1):-1])
    #print(np.exp(nu_))
    nu.append(nu_)
    try:
        if i!=(t_o - 1):
            x = np.random.poisson(np.exp(nu_))
            while x==0:
                x = np.random.poisson(np.exp(nu_))
            Y.append(x)
        else:
            x = np.random.poisson(alpha*np.exp(nu_))
            while x==0:
                x = np.random.poisson(alpha*np.exp(nu[i]))
            Y.append(x)

        if i<(N-1):
            t1.append(np.log(x) - np.dot(X[i+1], Beta))
            t2.append(np.log(x) - nu_)
    except:
        break

