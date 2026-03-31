import numpy as np
from LinearTriangulation import LinearTriangulation


def DisambiguateCameraPose(poses, pts1, pts2, K):
    """
    Disambiguate camera pose using cheirality condition
    
    Args:
        poses: List of 4 possible camera poses as tuples
        pts1: Feature points in image 1 (Nx2)
        pts2: Feature points in image 2 (Nx2)
        K: Camera intrinsic matrix (3x3)
        
    Returns:
        best_pose_idx: index of the best pose
        best_R: best rotation matrix (3x3)
        best_t: best translation vector (3x1)
        X: 3D points triangulated with the best pose (Nx3)
    """
    if len(poses) != 4:
        raise ValueError("Expected exactly 4 camera poses")
    
    # First camera at origin
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = camera_pose_to_projection_matrix(R1, t1, K)
    
    best_pose_idx = 0
    best_count = 0
    best_R = None
    best_t = None
    best_X = None
    
    for i, (R2, t2) in enumerate(poses):
        # Create projection matrix for second camera
        P2 = camera_pose_to_projection_matrix(R2, t2, K)
        
        # Triangulate points
        X = LinearTriangulation(P1, P2, pts1, pts2)
        
        # Count points in front of both cameras (cheirality check)
        count_positive = check_cheirality(X, R1, t1, R2, t2)
        
        if count_positive > best_count:
            best_count = count_positive
            best_pose_idx = i
            best_R = R2.copy()
            best_t = t2.copy()
            best_X = X.copy()
    
    return best_pose_idx, best_R, best_t, best_X


def check_cheirality(X, R1, t1, R2, t2, depth_threshold=0.1):
    """
    Returns:
        count: Number of points in front of both cameras
    """
    count = 0
    n_points = X.shape[0]
    
    for i in range(n_points):
        point = X[i].reshape(3, 1)
        
        # Skip non-finite points
        if not np.all(np.isfinite(point)):
            continue
        
        # Transform point to camera coordinate systems
        # we use world coordinates, point_cam = R^T @ (point - t)
        point_cam1 = R1.T @ (point - t1.reshape(3, 1))
        point_cam2 = R2.T @ (point - t2.reshape(3, 1))
        
        depth1 = point_cam1[2, 0]
        depth2 = point_cam2[2, 0]
        
        if depth1 > depth_threshold and depth2 > depth_threshold:
            count += 1
    
    return count

def check_cheirality_single_point(X, R, t, depth_threshold=0.1):
    X_homo = np.append(X, 1).reshape(4, 1)
    Rt = np.hstack([R, t.reshape(3, 1)])
    X_cam = Rt @ X_homo
    return X_cam[2, 0] > depth_threshold

# helper funcs

def camera_pose_to_projection_matrix(R, t, K):
    
    Rt = np.hstack([R, t.reshape(3, 1)])
    
    P = K @ Rt
    
    return P


def decompose_projection_matrix(P):
    M = P[:, :3]
    p4 = P[:, 3]
    
    # QR decomposition to separate K and R
    K, R = np.linalg.qr(np.flipud(np.fliplr(M)))
    K = np.flipud(np.fliplr(K))
    R = np.flipud(np.fliplr(R))
    
    signs = np.diag(np.sign(np.diag(K)))
    K = K @ signs
    R = signs @ R
    
    # Compute translation
    t = np.linalg.inv(K) @ p4
    
    return K, R, t.reshape(3, 1)