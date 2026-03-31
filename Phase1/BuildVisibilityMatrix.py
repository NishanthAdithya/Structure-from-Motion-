import numpy as np


def BuildVisibilityMatrix(camera_indices, point_indices):
    n_observations = len(camera_indices)
    if len(point_indices) != n_observations:
        raise ValueError("Camera and point indices must have same length")
    
    n_cameras = max(camera_indices) + 1
    n_points = max(point_indices) + 1
    
    # Initialize visibility matrix
    visibility_matrix = np.zeros((n_cameras, n_points), dtype=bool)
    observation_map = {}
    
    # Fill visibility matrix
    for obs_idx in range(n_observations):
        cam_idx = camera_indices[obs_idx]
        point_idx = point_indices[obs_idx]
        
        visibility_matrix[cam_idx, point_idx] = True
        observation_map[(cam_idx, point_idx)] = obs_idx
    
    return visibility_matrix, observation_map