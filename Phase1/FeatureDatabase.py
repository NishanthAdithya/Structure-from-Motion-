import numpy as np
import os

class FeatureDatabase:
    """
    A database to track feature correspondences across multiple images.
    Maps feature_id -> {image_id: (x, y)} for correspondence tracking.
    """
    
    def __init__(self):
        self.features = {}  # feature_id -> {image_id: (x, y)}
        self.image_features = {}  # image_id -> [feature_ids]
        self.n_features = 0
        
    def build_from_matching_files(self, data_dir='Phase1/P2Data'):
        """
        Build feature database from matching files.
        
        """
        print("Building feature database from matching files...")
        
        # # Process each matching file
        # for base_img in [1, 2, 3, 4]:
        #     matching_file = f"{data_dir}/matching{base_img}.txt"
        #     if os.path.exists(matching_file):
        #         self._parse_matching_file(matching_file, base_img)

        base_img = 1
        files_found = 0
        while True:
            matching_file = f"{data_dir}/matching{base_img}.txt"
            if not os.path.exists(matching_file):
                break
            self._parse_matching_file(matching_file, base_img)
            files_found += 1
            base_img += 1

        if files_found == 0:
            print(f"  WARNING: No matching files found in {data_dir}")
            return
        
        self._build_image_feature_lists()
        print(f"Feature database built: {len(self.features)} features across {len(self.image_features)} images")
        
    def _parse_matching_file(self, matching_file, base_image_id):
        """
        Parse a single matching file and add to database.
        
        Format: n_matches r g b x1 y1 target_img_id x2 y2 [target_img_id x3 y3 ...]
        """
        with open(matching_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header line
        for line_idx, line in enumerate(lines[1:], 1):
            parts = line.strip().split()
            if len(parts) < 6:
                continue
                
            # Create unique feature ID based on base image and line
            feature_id = f"{base_image_id}_{line_idx:03d}"
            
            # Parse base image coordinates
            x_base, y_base = float(parts[4]), float(parts[5])
            
            if feature_id not in self.features:
                self.features[feature_id] = {}
            
            self.features[feature_id][base_image_id] = (x_base, y_base)
            
            # Parse target image coordinates
            i = 6
            while i < len(parts):
                try:
                    target_img = int(parts[i])
                    x_target, y_target = float(parts[i+1]), float(parts[i+2])
                    self.features[feature_id][target_img] = (x_target, y_target)
                    i += 3
                except (ValueError, IndexError):
                    break
                    
        print(f"  Parsed {matching_file}: {len(lines)-1} features")
        
    def _build_image_feature_lists(self):
        """Build reverse mapping: image_id -> [feature_ids]"""
        self.image_features = {}
        
        for feature_id, views in self.features.items():
            for image_id in views.keys():
                if image_id not in self.image_features:
                    self.image_features[image_id] = []
                self.image_features[image_id].append(feature_id)
                
        # Sort feature lists for consistency
        for image_id in self.image_features:
            self.image_features[image_id].sort()
            
    def get_2d_2d_correspondences(self, image1, image2):
        """
        Get 2D-2D correspondences between two images.
        
        Args:
            image1, image2: Image IDs
            
        Returns:
            pts1, pts2: Arrays of corresponding points
            feature_ids: List of feature IDs for tracking
        """
        pts1, pts2, feature_ids = [], [], []
        
        for feature_id, views in self.features.items():
            if image1 in views and image2 in views:
                pts1.append(views[image1])
                pts2.append(views[image2])
                feature_ids.append(feature_id)
                
        return np.array(pts1), np.array(pts2), feature_ids
    
    def get_2d_3d_correspondences(self, reconstructed_points, new_image_id):
        """
        Get 2D-3D correspondences for PnP.
        
        Args:
            reconstructed_points: Dict {feature_id: 3D_point}
            new_image_id: ID of image to add
            
        Returns:
            points_2d: 2D points in new image
            points_3d: Corresponding 3D points
            feature_ids: Feature IDs for tracking
        """
        points_2d, points_3d, feature_ids = [], [], []
        
        for feature_id, point_3d in reconstructed_points.items():
            if feature_id in self.features and new_image_id in self.features[feature_id]:
                points_2d.append(self.features[feature_id][new_image_id])
                points_3d.append(point_3d)
                feature_ids.append(feature_id)
                
        return np.array(points_2d), np.array(points_3d), feature_ids
    
    def get_common_features(self, image_ids):
        """
        Get features visible in all specified images.
        Args:
            image_ids: List of image IDs
            
        Returns:
            common_features: List of feature IDs visible in all images
        """
        if not image_ids:
            return []
            
        # Start with features from first image
        common_features = set(self.image_features.get(image_ids[0], []))
        
        # Intersect with features from other images
        for image_id in image_ids[1:]:
            image_features = set(self.image_features.get(image_id, []))
            common_features = common_features.intersection(image_features)
            
        return list(common_features)
