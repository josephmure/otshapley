
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

def design_output(method, m, model, Xall, Xcond, d, Nv, No, Ni = 3):
    
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

# In[6]:

def ShapleyPerm(method,bootstrap, perms, y, d, Nv, No, Ni=3):
    
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
    
    #----------------
    # Estimate Var[Y]
    #----------------
    
    Y = y[:Nv]
    EY = np.mean(Y)
    VarY = np.var(Y, ddof = 1)

    y_for_boot = y[Nv:]
    
    #-----------------------------------------------
    # Estimate Shapley, main and total Sobol effects
    #-----------------------------------------------
    
    cVar = np.zeros(No)

    for b in range(bootstrap):
        
        if (b == 0):
            y = y_for_boot.copy()
        else:
            boot_idx = np.random.randint(low = 0, high = Ni, size=(m*(d-1)*No,Ni))
            temp = np.repeat(np.arange(0,m*(d-1)*No*Ni,Ni)[:,np.newaxis], Ni, 1)
            boot_idx = boot_idx + temp
            boot_idx = boot_idx.reshape(m*(d-1)*No*Ni,1)
            y = y_for_boot[boot_idx]
             
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
    
    Sh = Sh / m / VarY
    
    if (method == 'exact'):
        Vsob = Vsob / (m/d) / VarY # averaging by number of permutations with j=d-1
        Vsob = 1 - Vsob 
        Tsob = Tsob / (m/d) / VarY # averaging by number of permutations with j=1 
    else:
        Vsob = Vsob / nV / VarY # averaging by number of permutations with j=d-1
        Vsob = 1 - Vsob 
        Tsob = Tsob / nT / VarY # averaging by number of permutations with j=1 
    
    col = ['X' + str(i) for i in np.arange(d)+1]
    percentiles = [0.025,0.975]
    
    Sh_effects = pd.DataFrame(Sh, columns = col)
    Sh_describe = Sh_effects.describe(percentiles=[0.025,0.975])

    Vsob_effects = pd.DataFrame(Vsob, columns = col)
    Vsob_describe = Vsob_effects.describe(percentiles=[0.025,0.975])
    
    Tsob_effects = pd.DataFrame(Tsob, columns = col)
    Tsob_describe = Tsob_effects.describe(percentiles=[0.025,0.975])
    

    return Sh_describe, Vsob_describe, Tsob_describe


# ## Evaluation Shapley effects on linear gaussian model

# #### Parameters of the model

# In[ ]:

d = 3
coeff_model = np.array([1,1,1])

moyenne = np.zeros(3)
corr = 0.9
cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
cov = ot.CovarianceMatrix(cov)

def gaussian_model(X):
    return np.sum(X,1)


# #### Function to estimate Shapley effects

# In[ ]:

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


# #### Calculate true values of the Shapley effects

# In[ ]:

True_Sh = Sh_effects_gaussian_linear_model(coeff_model, cov, corr)


# #### Estimate Shapley effects with Exact method of permutations

# In[ ]:

method = 'exact'
bootstrap = 5
m = None
Nv = 10**4
No = 10**3
Ni = 10

perms, y = design_output(method, m, gaussian_model, Xall, Xcond, d, Nv, No, Ni)

Sh, Vsob, Tsob = ShapleyPerm(method, bootstrap, perms, y, d, Nv, No, Ni)

print('Exact method \n\n')

print('Shapley effects \n' + str(Sh) + '\n\n')
print('First order Sobol \n' + str(Vsob) + '\n\n')
print('Total Sobol \n' + str(Tsob) + '\n\n')

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('index.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
Sh.to_excel(writer, sheet_name='Shapley')
Vsob.to_excel(writer, sheet_name='First_sobol')
Tsob.to_excel(writer, sheet_name='Total_sobol')

# Close the Pandas Excel writer and output the Excel file.
writer.save()




