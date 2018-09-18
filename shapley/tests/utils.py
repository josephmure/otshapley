import numpy as np

def get_id(pos, dim):
    k = 0
    for i in range(1, dim):
        for j in range(i):
            if [i, j] == pos or [j, i] == pos:
                return k
            k += 1
            
def true_gaussian_full_ind_sobol(theta, dim=3):
    Vy = 3. + 2.*(sum(theta))
    S_full = np.zeros((dim, ))
    S_ind = np.zeros((dim, ))
    for i in range(dim):
        j, k = np.setdiff1d(range(dim), i)
        theta_ij = theta[get_id([i, j])]
        theta_ik = theta[get_id([i, k])]
        theta_jk = theta[get_id([j, k])]
        Vi = (1 + theta_ij + theta_ik)**2
        Viu = 1 + (2 * np.prod(theta) - theta_ij**2 - theta_ik**2) / (1. - theta_jk**2)
        S_full[i] = Vi / Vy
        S_ind[i] = Viu / Vy
    return S_full, S_ind