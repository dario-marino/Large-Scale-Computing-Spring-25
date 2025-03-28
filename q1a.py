# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:19:39 2025

@author: dario
"""

from numba import jit
import numpy as np
import scipy.stats as sts
import timeit


@jit(nopython=True)
def loop(): 
    #we need to put the final structure inside the jit
    z_mat = np.zeros((T, S), dtype=np.float64)
    for s_ind in range(S): 
        z_tm1 = z_0 
        for t_ind in range(T): 
            e_t = eps_mat[t_ind, s_ind] 
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t 
            z_mat[t_ind, s_ind] = z_t 
            z_tm1 = z_t


# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters, draw all idiosyncratic random shocks,
# and create empty containers
S = 1000  # Set the number of lives to simulate
T = int(4160)  # Set the number of periods for each simulation
np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))
loop()



print(timeit.timeit(loop, number=1))




import numpy as np
import scipy.stats as sts
import timeit

def run_simulation():
    # Set model parameters
    rho = 0.5
    mu = 3.0
    sigma = 1.0
    z_0 = mu

    S = 1000  # Set the number of lives to simulate
    T = int(4160)  # Set the number of periods for each simulation
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    z_mat = np.zeros((T, S))

    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

print(timeit.timeit(run_simulation, number=1))
        
        
        
    