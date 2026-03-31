import numpy as np

def LinearTriangulation(P1, P2, pts1, pts2):
    """
    triangulate 3D points using Direct Linear Transform
    """
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("Point arrays must have same number of points")
    
    n_points = pts1.shape[0]
    X = np.zeros((n_points, 3))
    
    for i in range(n_points):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A = np.array([
            x1 * P1[2, :] - P1[0, :],      # x1*P1[2,:] - P1[0,:]
            y1 * P1[2, :] - P1[1, :],      # y1*P1[2,:] - P1[1,:]
            x2 * P2[2, :] - P2[0, :],      # x2*P2[2,:] - P2[0,:]
            y2 * P2[2, :] - P2[1, :]       # y2*P2[2,:] - P2[1,:]
        ])

        U, S, Vt = np.linalg.svd(A)
        X_homo = Vt[-1]  # Last row of V^T (smallest singular value)
        
        # Convert to Cartesian coordinates
        if abs(X_homo[3]) > 1e-10:
            X[i] = X_homo[:3] / X_homo[3]
        else:
            # Point at infinity - set to a large distance
            X[i] = X_homo[:3] * 1000
    
    return X