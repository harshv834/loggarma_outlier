import numpy as np
import scipy as sp
import statsmodels.api as sm
from loggarma import loggarma


data = np.load('Data.npz')

X3,Y2 = data['arr_1'], data['arr_0']

beta1,phi1,theta1, alpha1 = loggarma(X3,Y2,p=2,q=4,max_iter=100,t_0=50)
print(beta1)