import numpy as np


def EssentialMatrixFromFundamentalMatrix(F, K1, K2):
    # Essential matrix: E = K2^T * F * K1
    E = K2.T @ F @ K1
    
    # Enforce essential matrix constraints
    U, S, Vt = np.linalg.svd(E, full_matrices=True)
    sigma = 1
    S_corrected = np.array([sigma, sigma, 0])

    E = U @ np.diag(S_corrected) @ Vt
    
    return E


    


