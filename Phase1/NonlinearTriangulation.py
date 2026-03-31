import numpy as np
from scipy.optimize import least_squares
from LinearTriangulation import LinearTriangulation


def NonlinearTriangulation(P1, P2, pts1, pts2, X_init=None):
    """
    Args:
        P1: Camera projection matrix for image 1 (3x4)
        P2: Camera projection matrix for image 2 (3x4)
        pts1: Feature points in image 1 (Nx2)
        pts2: Feature points in image 2 (Nx2)
        X_init: Initial 3D points (Nx3). If None, use linear triangulation
        
    Returns:
        X_optimized: Optimized 3D points (Nx3)
        residuals: Final reprojection residuals
    """
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("Point arrays must have same number of points")
    
    n_points = pts1.shape[0]
    
    # Get initial 3D points if not provided
    if X_init is None:
        X_init = LinearTriangulation(P1, P2, pts1, pts2)
    
    X_optimized = np.zeros_like(X_init)
    residuals = np.zeros(n_points)

    
    
    # Optimize each point individually
    for i in range(n_points):
        result = optimize_single_point(P1, P2, pts1[i], pts2[i], X_init[i])
        # print("Came HERE")
        X_optimized[i] = result['x']
        residuals[i] = result['cost']
    
    return X_optimized, residuals


def optimize_single_point(P1, P2, pt1, pt2, X_init):
    
    def residual_function(X):
        """Compute reprojection residuals for a 3D point"""
        X_homo = np.append(X, 1)
        
        proj1_homo = P1 @ X_homo
        proj2_homo = P2 @ X_homo
        
        # Convert to Cartesian coordinates
        if abs(proj1_homo[2]) > 1e-10:
            proj1 = proj1_homo[:2] / proj1_homo[2]
        else:
            proj1 = np.array([1e6, 1e6])  # Large error for points at infinity
            
        if abs(proj2_homo[2]) > 1e-10:
            proj2 = proj2_homo[:2] / proj2_homo[2]
        else:
            proj2 = np.array([1e6, 1e6])
        
        residual1 = pt1 - proj1
        residual2 = pt2 - proj2
        
        return np.concatenate([residual1, residual2])
    
    # Run optimization
    result = least_squares(residual_function, X_init, method='lm')
    return result