
# coding: utf-8

# In[1]:

import openturns as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Function for conditional gaussian vector

# #### Calcul of conditional mean and variance

# In[2]:

def condMVN(mean, cov, dependent_ind, given_ind, X_given):
    
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.array(cov)
    
    B = cov.take(dependent_ind, axis = 1)
    B = B[dependent_ind]
    
    C = cov.take(dependent_ind, axis = 1)
    C = C[given_ind]
    
    D = cov.take(given_ind, axis = 1)
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C),np.linalg.inv(D))
    
    condMean = mean[dependent_ind] + np.dot(CDinv,(X_given - mean[given_ind]))
    condVar = B - np.dot(CDinv,C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean,condVar


# #### Generate conditional law

# In[3]:

def r_condMVN(n, mean, cov, dependent_ind, given_ind, X_given):
    
    """ Function to simulate conditional gaussian distribution of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cond_mean,cond_var = condMVN(mean, cov, dependent_ind, given_ind, X_given)
    distribution = ot.Normal(cond_mean,cond_var)
    return distribution.getSample(n)


# #### Shapley effects of gaussian linear model with three inputs

# In[4]:

def Sh_effects_gaussian_linear_model(coeff_model, cov_matrix, corr):
    
    """ Function to calculate the Shapley effects on a gaussian linear model with three inputs
    
    We assume X1 is independent from both X2 and X3, and that X2 and X3 may be correlated
    """

    Var_inputs = np.diagonal(cov).copy()
    Sd_inputs = np.sqrt(Var_inputs)

    Var_model = (coeff_model**2*Var_inputs).sum() + 2*corr*coeff_model[1]*coeff_model[2]*Sd_inputs[1]*Sd_inputs[2]
    
    Sh = np.zeros(3)

    Sh[0] = (coeff_model[0]**2*Var_inputs[0])/Var_model

    Effect_2 = coeff_model[1]**2*Var_inputs[1]
    Effect_3 = coeff_model[2]**2*Var_inputs[2]
    Effect_23 = corr*coeff_model[1]*coeff_model[2]*Sd_inputs[1]*Sd_inputs[2]

    Sh[1] = (Effect_2 + Effect_23 + corr**2*(Effect_3 - Effect_2)/2)/Var_model
    Sh[2] = (Effect_3 + Effect_23 + corr**2*(Effect_2 - Effect_3)/2)/Var_model
    
    return Sh


# ## Shapley function

# #### Conceive the design matrix and calculate the output

# In[5]:

def compute_output(method, m, model, Xall, Xcond, d, Nv, No, Ni = 3):
    
    """Function to design the design matrix and calculate  the output"""
    
    if (method == 'exact'):
        perms_tool = ot.KPermutations(d, d)
        perms = perms_tool.generate()

        m = perms.getSize() # number of permutation
    else:
        perms = np.zeros((m,d), dtype = np.int64)
        for i in range(m):
            perms[i] = np.random.permutation(d) # random permutation
    
    #------------------------------
    # Creation of the design matrix
    #------------------------------
    
    X = np.zeros((Nv+m*(d-1)*No*Ni, d)) 
    X[:Nv,:] = Xall(Nv)

    for p in range(m):
    
        pi = perms[p]
        pi_sorted = np.argsort(pi)
    
        for j in range(1,d):
        
            Sj = pi[:j] # set of the 0st-(j-1)th elements in pi      
            Sjc = pi[j:] # set of the jth-dth elements in pi
        
            xjcM = Xcond(No,Sjc,None,None)# sampled values of the inputs in Sjc

            for l in range(No):
                
                xjc = xjcM[l,:]
            
                # sample values of inputs in Sj conditional on xjc
                xj = Xcond(Ni, Sj, Sjc, xjc)
                xx = np.concatenate((xj, np.ones((Ni,1))*xjc), axis = 1)
                ind_inner = Nv + p*(d-1)*No*Ni + (j-1)*No*Ni + l*Ni
                X[ind_inner:(ind_inner + Ni),:] = xx[:,pi_sorted]
    
    #-----------------------
    # Calcul of the response
    #-----------------------
    
    y = model(X)
    
    return perms,y


# #### Caculate the Shapley effetcs and make bootstrap

# In[20]:

def ShapleyPerm(method,bootstrap, perms, y, d, Nv, No, Ni = 3):
    
    """ Function to calculate the Shapley effects and confidence interval of the values """
    
    if (method == 'exact'):
        m = perms.getSize()
    else:
        m = np.int(perms.shape[0])
    
    #-----------------------------------------------------------------
    # Initialize Shapley, main and total Sobol effects for all players
    #-----------------------------------------------------------------
    
    Sh = np.zeros((bootstrap,d))
    Vsob = np.zeros((bootstrap,d))
    Tsob = np.zeros((bootstrap,d))
    
    nV = np.zeros((bootstrap,d)) # number of samples used to estimate V1,...,Vd
    nT = np.zeros((bootstrap,d)) # number of samples used to estimate T1,...,Td
    
    #---------------------
    # Sample for bootstrap
    #---------------------
    
    Y_for_boot = y[:Nv]
    y_for_boot = y[Nv:]
    
    #-----------------------------------------------
    # Estimate Shapley, main and total Sobol effects
    #-----------------------------------------------
    
    cVar = np.zeros(No)

    for b in range(bootstrap):
        
        if (b == 0):
            Y = Y_for_boot.copy()
            VarY = np.var(Y, ddof = 1)

            y = y_for_boot.copy()
        else:
            Y = Y_for_boot[np.random.randint(low = 0, high = Nv, size = Nv)]
            VarY = np.var(Y, ddof = 1)
            
            discrete_index_array = np.arange(0, m*(d-1)*No*Ni).reshape(m*(d-1)*No,Ni)
            random_boot_1 = np.random.randint(low = 0, high = No, size=(m*(d-1),No))
            discrete_No =  np.repeat(np.arange(0,m*(d-1)*No,No)[:,np.newaxis], No, 1)
            boot_1_index_to_select = (random_boot_1 + discrete_No).ravel()
            boot_1 = discrete_index_array[boot_1_index_to_select].ravel()
            y = y_for_boot[boot_1]
            
            random_boot_2 = np.random.randint(low = 0, high = Ni, size=(m*(d-1)*No,Ni))
            discrete_Ni = np.repeat(np.arange(0,m*(d-1)*No*Ni,Ni)[:,np.newaxis], Ni, 1)
            boot_2 = (random_boot_2 + discrete_Ni).ravel()
            y = y[boot_2]
             
        for p in range(m):
    
            pi = perms[p]
            prevC = 0
    
            for j in range(d):
                if (j == (d-1)):
                    Chat = VarY
                    delta = Chat - prevC
                    Vsob[b,pi[j]] = Vsob[b,pi[j]] + prevC # first order effect
                    nV[b,pi[j]] = nV[b,pi[j]] + 1
                else:
                    for l in range(No):
                        Y = y[:Ni]
                        y = y[Ni:]
                        cVar[l] = np.var(Y, ddof = 1)
                    Chat = np.mean(cVar)
                    delta = Chat - prevC
      
                Sh[b,pi[j]] = Sh[b,pi[j]] + delta
        
                prevC = Chat
        
                if (j == 0):
                    Tsob[b,pi[j]] = Tsob[b,pi[j]] + Chat # Total effect
                    nT[b,pi[j]] = nT[b,pi[j]] + 1
        
        Sh[b,:] = Sh[b,:] / m / VarY
        
        print('b  = '+str(b)+'\n')
        
        print('nV = '+str(nV)+'\n')
        
        if (method == 'exact'):
            Vsob[b,:] = Vsob[b,:] / (m/d) / VarY # averaging by number of permutations with j=d-1
            Vsob[b,:] = 1 - Vsob[b,:] 
            Tsob[b,:] = Tsob[b,:] / (m/d) / VarY # averaging by number of permutations with j=1 
        else:
            Vsob[b,:] = Vsob[b,:] / nV[b,:] / VarY # averaging by number of permutations with j=d-1
            Vsob[b,:] = 1 - Vsob[b,:] 
            Tsob[b,:] = Tsob[b,:] / nT[b,:] / VarY # averaging by number of permutations with j=1 
    
    rownames = ['X' + str(i) for i in np.arange(d)+1]
    percentiles = [0.025,0.975]
    
    # Shapley effects
    colnames = ['Sh','IC_min','IC_max']

    Sh = pd.DataFrame(Sh)
    Sh_describe = Sh.iloc[1:,:].describe(percentiles = [0.025,0.975])
    
    Sh_ref = (Sh.iloc[0].values)[:,np.newaxis]
    CI_min = 2*Sh_ref - (Sh_describe.iloc[6].values)[:,np.newaxis]
    CI_max = 2*Sh_ref - (Sh_describe.iloc[4].values)[:,np.newaxis]

    Sh_out = np.concatenate((Sh_ref,CI_min,CI_max), axis = 1)
    Sh_out = pd.DataFrame(Sh_out,index = rownames, columns = colnames)
    
    # First order Sobol
    colnames = ['First Sobol','IC_min','IC_max']
    
    Vsob = pd.DataFrame(Vsob)
    Vsob_describe = Vsob.iloc[1:,:].describe(percentiles = [0.025,0.975])
    
    Vsob_ref = (Vsob.iloc[0].values)[:,np.newaxis]
    CI_min = 2*Vsob_ref - (Vsob_describe.iloc[6].values)[:,np.newaxis]
    CI_max = 2*Vsob_ref - (Vsob_describe.iloc[4].values)[:,np.newaxis]
    
    Vsob_out = np.concatenate((Vsob_ref,CI_min,CI_max), axis  = 1)
    Vsob_out = pd.DataFrame(Vsob_out, index = rownames, columns = colnames)
    
    # Total order Sobol
    colnames = ['Total Sobol','IC_min','IC_max']
    
    Tsob = pd.DataFrame(Tsob)
    Tsob_describe = Tsob.iloc[1:,:].describe(percentiles = [0.025,0.975])
    
    Tsob_ref = (Tsob.iloc[0].values)[:,np.newaxis]
    CI_min = 2*Tsob_ref - (Tsob_describe.iloc[6].values)[:,np.newaxis]
    CI_max = 2*Tsob_ref - (Tsob_describe.iloc[4].values)[:,np.newaxis]

    Tsob_out = np.concatenate((Tsob_ref,CI_min,CI_max), axis = 1)
    Tsob_out = pd.DataFrame(Tsob_out, index = rownames, columns = colnames)

    return Sh_out, Vsob_out, Tsob_out


# ## Bootsrap test

# #### Parameters of the model

# In[7]:

d = 3
coeff_model = np.array([1,1,1])

moyenne = np.zeros(3)
corr = 0.9
cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
cov = ot.CovarianceMatrix(cov)

def gaussian_model(X):
    return np.sum(X,1)


# #### Function to estimate Shapley effects

# In[8]:

def Xall(n):
    distribution = ot.Normal(moyenne,cov)
    return distribution.getSample(n)

def Xcond(n, Sj, Sjc, xjc):
    if Sjc is None:
        cov_int = np.array(cov)
        cov_int = cov_int.take(Sj, axis = 1)
        cov_int = cov_int[Sj]        
        cov_int = ot.CovarianceMatrix(cov_int)
        distribution = ot.Normal(moyenne[Sj],cov_int)
        return distribution.getSample(n)
    else:
        return r_condMVN(n,mean = moyenne, cov = cov, dependent_ind = Sj, given_ind = Sjc, X_given = xjc)


# #### Estimate Shapley effects with Exact method of permutations

# In[9]:

method = 'exact'
m = None
Nv = 10**4
No = 10**3
Ni = 10**2

perms, y = compute_output(method, m, gaussian_model, Xall, Xcond, d, Nv, No, Ni)

bootstrap = 10**3
Sh, Vsob, Tsob = ShapleyPerm(method, bootstrap, perms, y, d, Nv, No, Ni)

writer = pd.ExcelWriter('bootstrap_exact.xlsx', engine='xlsxwriter')
Sh.to_excel(writer, sheet_name = 'Sh_exact')
Vsob.to_excel(writer, sheet_name = 'Vsob_exact')
Tsob.to_excel(writer, sheet_name = 'Tsob_exact')
writer.save()


# #### Estimate Shapley effects with random permutations

# In[21]:

method = 'random'
m = 6000
Nv = 10**4
No = 1
Ni = 3

perms, y = compute_output(method, m, gaussian_model, Xall, Xcond, d, Nv, No, Ni)

bootstrap = 10**3
Sh, Vsob, Tsob = ShapleyPerm(method, bootstrap, perms, y, d, Nv, No, Ni)

writer = pd.ExcelWriter('bootstrap_random.xlsx', engine='xlsxwriter')
Sh.to_excel(writer,sheet_name='Sh_random_1')
Vsob.to_excel(writer,sheet_name='Vsob_random_1')
Tsob.to_excel(writer,sheet_name='Tsob_random_1')


# In[11]:

method = 'random'
m = 6000
Nv = 10**4
No = 10**2
Ni = 10**2

perms, y = compute_output(method, m, gaussian_model, Xall, Xcond, d, Nv, No, Ni)

bootstrap = 10**3
Sh, Vsob, Tsob = ShapleyPerm(method, bootstrap, perms, y, d, Nv, No, Ni)

Sh.to_excel(writer,sheet_name='Sh_random_2')
Vsob.to_excel(writer,sheet_name='Vsob_random_2')
Tsob.to_excel(writer,sheet_name='Tsob_random_2')
writer.save()

