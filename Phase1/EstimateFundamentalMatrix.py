import numpy as np


def EstimateFundamentalMatrix(pts1, pts2):
    """
    Estimate fundamental matrix using 8-point algorithm with normalization
    """
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("Point arrays must have same number of points")
    
    if pts1.shape[0] < 8:
        raise ValueError("Need at least 8 point correspondences")
    
    n_points = pts1.shape[0]
    
    # Normalize points for numerical stability
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # Build constraint matrix A using normalized coordinates
    A = []
    for i in range(n_points):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        # x1, y1 = pts1[i]
        # x2, y2 = pts2[i]

        # Each row represents the constraint x2^T * F * x1 = 0
        # A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    
    # Solution is the last row of V (or last column of V^T)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F, full_matrices=True)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ np.diag(S) @ Vt
    
    # Denormalize: F = T2^T * F_normalized * T1
    F = T2.T @ F @ T1
    
    return F


def normalize_points(pts):
    # Compute centroid
    centroid = np.mean(pts, axis=0)
    
    # Compute mean distance from centroid
    distances = np.linalg.norm(pts - centroid, axis=1)
    mean_dist = np.mean(distances)
    
    # Scaling factor to make mean distance sqrt(2)
    if mean_dist > 0:
        scale = np.sqrt(2) / mean_dist
    else:
        scale = 1
    
    # Construct normalization transformation matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    # Apply transformation to points
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Convert to homogeneous
    pts_norm_homo = (T @ pts_homo.T).T
    pts_norm = pts_norm_homo[:, :2] / pts_norm_homo[:, 2:3]  # Convert back to Cartesian
    
    return pts_norm, T
