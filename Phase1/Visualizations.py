import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import camera_pose_to_projection_matrix

import os


def _camera_center(R, t):
    """World-space camera centre from (R, t)."""
    return (-R.T @ t.reshape(3, 1)).flatten()

def _clip_axes(ax, pts_list):
    """Clip plot axes to 1st-99th percentile of all provided point arrays."""
    combined = np.vstack([p for p in pts_list if len(p) > 0])
    if len(combined) < 2:
        return
    x_lo, x_hi = np.percentile(combined[:, 0], [1, 99])
    z_lo, z_hi = np.percentile(combined[:, 2], [1, 99])
    pad_x = max((x_hi - x_lo) * 0.1, 0.5)
    pad_z = max((z_hi - z_lo) * 0.1, 0.5)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(z_lo - pad_z, z_hi + pad_z)


def _filter_finite(pts):
    mask = np.all(np.isfinite(pts), axis=1)
    return pts[mask]



def set_axes_equal(ax, pts=None):
    
    if pts is not None and len(pts) > 0:
        # Get 5th and 95th percentiles to define robust bounding box
        x_min, x_max = np.percentile(pts[:,0], [10, 90])
        y_min, y_max = np.percentile(pts[:,1], [10, 90])
        z_min, z_max = np.percentile(pts[:,2], [10, 90])
        
        # Make it a cube
        max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
        mid_x = (x_max+x_min) * 0.5
        mid_y = (y_max+y_min) * 0.5
        mid_z = (z_max+z_min) * 0.5
        
        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
    else:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        plot_radius = 0.5*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def draw_matches(img1, img2, pts1, pts2, inliers=None, save_path="matches.png"):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create empty image
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:] = img2
    
    for i in range(len(pts1)):
        pt1 = (int(pts1[i][0]), int(pts1[i][1]))
        pt2 = (int(pts2[i][0] + w1), int(pts2[i][1]))
        
        if inliers is None:
            color = (0, 255, 0)
        else:
            color = (0, 255, 0) if inliers[i] else (0, 0, 255)
            
        cv2.circle(out_img, pt1, 3, color, -1)
        cv2.circle(out_img, pt2, 3, color, -1)
        cv2.line(out_img, pt1, pt2, color, 1)
        
    cv2.imwrite(save_path, out_img)

def visualize_four_poses(poses, pts1, pts2, K, save_path="four_poses.png"):

    R1, t1 = np.eye(3), np.zeros((3, 1))
    P1 = camera_pose_to_projection_matrix(R1, t1, K)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (R2, t2) in enumerate(poses):
        ax = axes[i]
        P2 = camera_pose_to_projection_matrix(R2, t2, K)
        X = LinearTriangulation(P1, P2, pts1, pts2)

        # Cheirality colouring
        colors = []
        for j in range(len(X)):
            x_h = np.append(X[j], 1).reshape(4, 1)
            d1 = (P1 @ x_h)[2, 0]
            d2 = (P2 @ x_h)[2, 0]
            colors.append('g' if (d1 > 0 and d2 > 0) else 'r')

        Xf = _filter_finite(X)
        if len(Xf):
            ax.scatter(Xf[:, 0], Xf[:, 2], s=4,
                       c=[colors[k] for k in range(len(X)) if np.all(np.isfinite(X[k]))],
                       alpha=0.6)

        # Camera centres
        ax.scatter(0, 0, marker='^', s=120, c='blue', zorder=5, label='Cam 1')
        C2 = _camera_center(R2, t2)
        ax.scatter(C2[0], C2[2], marker='^', s=120, c='black', zorder=5, label='Cam 2')

        ax.set_title(f"Pose {i + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.legend(fontsize=7)

    plt.suptitle("Four Camera Pose Hypotheses (green = valid cheirality)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_linear_vs_nonlinear(X_linear, X_nonlinear, camera_poses=None,
                                   save_path="linear_vs_nonlinear.png"):
    
    fig, ax = plt.subplots(figsize=(9, 7))

    Xl = _filter_finite(X_linear)
    Xn = _filter_finite(X_nonlinear)

    if len(Xl):
        ax.scatter(Xl[:, 0], Xl[:, 2], s=6, c='red', alpha=0.6, label='linear')
    if len(Xn):
        ax.scatter(Xn[:, 0], Xn[:, 2], s=6, c='blue', alpha=0.6, label='nonlinear')

    if camera_poses:
        cam_colors = ['blue', 'black', 'green', 'purple', 'orange']
        for imid, poses in camera_poses.items():
            R = poses[0]
            t = poses[1]
            C = _camera_center(R, t)
            ax.scatter(C[0], C[2], marker='^', s=150,
                       c=cam_colors[imid-1 % len(cam_colors)], zorder=5, label=f'Cam {imid}')

    _clip_axes(ax, [p for p in [Xl, Xn] if len(p) > 0])
    ax.set_title("Linear vs Non-linear Triangulation")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.legend()
    ax.set_ylim(bottom=-1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def plot_triangulation_reprojections(img, pts2d, X_lin, X_nonlin, P,
                                     title_prefix, save_path):
    
    def _project(X, P_mat):
        Xh = np.hstack([X, np.ones((len(X), 1))])   # Nx4
        ph = (P_mat @ Xh.T).T                         # Nx3
        # avoid division by zero
        denom = ph[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        return ph[:, :2] / denom

    proj_lin = _project(X_lin, P)
    proj_nonlin = _project(X_nonlin, P)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axs[0].imshow(img_rgb)
    axs[0].scatter(pts2d[:, 0], pts2d[:, 1], c='r', s=15, label='Observed', zorder=3)
    axs[0].scatter(proj_lin[:, 0], proj_lin[:, 1], c='b', s=10,
                   marker='x', label='Linear reprojection', zorder=4)
    axs[0].set_title(f"{title_prefix}: Linear Triangulation Reprojection")
    axs[0].legend()
    axs[0].axis('off')

    axs[1].imshow(img_rgb)
    axs[1].scatter(pts2d[:, 0], pts2d[:, 1], c='r', s=15, label='Observed', zorder=3)
    axs[1].scatter(proj_nonlin[:, 0], proj_nonlin[:, 1], c='g', s=10,
                   marker='x', label='Nonlinear reprojection', zorder=4)
    axs[1].set_title(f"{title_prefix}: Non-linear Triangulation Reprojection")
    axs[1].legend()
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()




def visualize_sfm(points_3d_dict, camera_poses, step_name, save_path):

    fig, ax = plt.subplots(figsize=(10, 8))
    cam_colors = ['red', 'green', 'cyan', 'magenta', 'yellow', 'black']

    if points_3d_dict:
        pts = _filter_finite(np.array(list(points_3d_dict.values())))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 2], s=3, c='blue', alpha=0.5, label='3D points')
            _clip_axes(ax, [pts])

    for imid, pose in camera_poses.items():
        R = pose[0]
        t = pose[1]
        C = _camera_center(R, t)
        ax.scatter(C[0], C[2], marker='^', s=200,
                   c=cam_colors[imid-1 % len(cam_colors)], zorder=5, label=f'Cam {imid}')

    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title(f"3D Reconstruction – {step_name}")
    ax.legend(fontsize=8)
    # ax.set_ylim(top=100)
    ax.set_ylim(bottom=-1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_sfm_before_after_ba(pts_before_dict, pts_after_dict,
                                   camera_poses_before, camera_poses_after,
                                   step_name, save_path):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    all_pts = []

    if pts_before_dict:
        pts = _filter_finite(np.array(list(pts_before_dict.values())))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 2], s=4, c='blue',
                       alpha=0.5, label='before bund adj')
            all_pts.append(pts)

    if pts_after_dict:
        pts = _filter_finite(np.array(list(pts_after_dict.values())))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 2], s=4, c='red',
                       alpha=0.5, label='after bund adj')
            all_pts.append(pts)

    if all_pts:
        _clip_axes(ax, all_pts)

    cam_colors = ['saddlebrown', 'darkorange', 'gold', 'purple', 'teal']
    for imid, poses in camera_poses_after.items():
        R = poses[0]
        t = poses[1]

        C = _camera_center(R, t)
        ax.scatter(C[0], C[2], marker='^', s=200,
                   c=cam_colors[imid-1 % len(cam_colors)], zorder=5, label=f'Cam {imid}')

    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title(f"Bundle Adjustment – {step_name}")
    ax.legend()
    # ax.set_ylim(top=100)
    ax.set_ylim(bottom=-1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pnp_reprojections(img, pts2d, pts3d,
                            R_lin, t_lin, R_nonlin, t_nonlin, K, save_path):
    def _proj(pts3, R, t):
        P = K @ np.hstack([R, t.reshape(3, 1)])
        Xh = np.hstack([pts3, np.ones((len(pts3), 1))])
        ph = (P @ Xh.T).T
        denom = ph[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        return ph[:, :2] / denom

    proj_lin    = _proj(pts3d, R_lin,    t_lin)
    proj_nonlin = _proj(pts3d, R_nonlin, t_nonlin)

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    for ax, proj, color, marker, label, title in [
        (axs[0], proj_lin,    'b', 'x', 'Linear PnP',    "Linear PnP Reprojection"),
        (axs[1], proj_nonlin, 'g', 'x', 'Nonlinear PnP', "Nonlinear PnP Reprojection"),
    ]:
        ax.imshow(img_rgb)
        ax.scatter(pts2d[:, 0], pts2d[:, 1],
                   c='r', s=15, label='Observed', zorder=3)
        ax.scatter(proj[:, 0], proj[:, 1],
                   c=color, s=20, marker=marker, label=label, zorder=4)
        ax.set_title(title)
        ax.legend()

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)   # y-axis inverted (image coords: 0 at top)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()