
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pylab
import math


# In[54]:


def pearson_residual(Y,mu):
    return pd.DataFrame((Y-mu)/(mu**0.5),columns=['p'])
def greatest_subset(P):
    return P.sort_values(['p'],ascending=[False])[0:math.ceil(0.05*len(P))]
def log_lkhood(P,alpha):
    # P has y and mu as columns
    P['log-likelihood']=P['p']*math.log(alpha)+P['mu']*(1-alpha)
    return P.sort_values(['log-likelihood'],ascending=[False])

####GIVEN Y
#1.FIT NULL MODEL on Y(dx1 vector)( taking alpha=1 and learn remaining params) #rana,harsh

#2.CALCUATE mu(dx1 vector) FROM OBTAINED PARAMETERS IN 1.#ravi

P=greatest_subset(pearson_residual(Y,mu))

#3. FIT ATERNATE MODEL ON P #rana,harsh
#4. CALCUATE mu_1 FROM OBTAINED PARAMETERS IN 3. #ravi

P['mu']=mu_1
P=log_lkhood(P,alpha_1) # alpha_1 obtained in 3.
t0=P.index[0]
Tt0=P['log-likelihood'][0]*(-2)

#5. GENERATE N DATASETS USING PARAMS IN  1.    #BOOTSTRAPPING #Ravi
#6. for each dataset D in [1,N] do above and calcuate Tt0   #rohit
#7. Based on obtained D Tt0 values in 6., comparing Tt0 for original data, we decide wether t0 is outier or not #rohit

