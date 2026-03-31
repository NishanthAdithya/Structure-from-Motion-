import numpy as np

def LinearPnP(points_3d, points_2d, K):
    n_points = len(points_3d)
    
    if n_points < 6:
        return None, None
    
    # Convert 2D points to normalized coordinates
    points_2d_homo = np.hstack([points_2d, np.ones((n_points, 1))])
    points_2d_norm = (np.linalg.inv(K) @ points_2d_homo.T).T[:, :2]
    
    A = []
    for i in range(n_points):
        X, Y, Z = points_3d[i]
        u, v = points_2d_norm[i]
        
        # Each point gives us 2 constraints #ORDER IS REVERSED
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    try:
        U, S, Vt = np.linalg.svd(A)
        P_vec = Vt[-1]  # Last row of V^T
        
        P = P_vec.reshape(3, 4)
        
        # Since we used normalized coordinates, P = [R|t]
        R_candidate = P[:, :3]
        t_candidate = P[:, 3].reshape(3, 1)
        
        # Ensure R is a proper rotation matrix
        U_r, S_r, Vt_r = np.linalg.svd(R_candidate)
        R = U_r @ Vt_r
        
        # Ensure det(R) = 1 (not -1)
        if np.linalg.det(R) < 0:
            R = -R
            t_candidate = -t_candidate
        
        return R, t_candidate
        
    except Exception:
        return None, None