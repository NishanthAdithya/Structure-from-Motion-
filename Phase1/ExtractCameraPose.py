import numpy as np


def ExtractCameraPose(E):
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation matrices (det = +1)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    t1 = U[:, 2].reshape(3, 1)
    t2 = -U[:, 2].reshape(3, 1)
    
    # Four possible camera poses
    poses = [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2)
    ]
    
    return poses