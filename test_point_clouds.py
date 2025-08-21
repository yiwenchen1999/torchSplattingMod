import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from pathlib import Path

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.point_utils import get_point_clouds, PointCloud
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera, parse_camera

# Additional imports for enhanced visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not available. Install with: pip install open3d")
    OPEN3D_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not available. Install with: pip install pillow")
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not available. Install with: pip install opencv-python")
    CV2_AVAILABLE = False

def test_point_cloud_generation():
    """
    Test the get_point_clouds function with sample data
    """
    print("=== Testing Point Cloud Generation ===")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folder = '../ship_latents_processed_test'
    # folder = 'B075X65R3X'
    
    print(f"Using device: {device}")
    print(f"Loading data from: {folder}")
    
    # Load data (same as in train.py)
    try:
        # data = read_all(folder, resize_factor=256.0/800.0)
        data = read_all(folder, resize_factor=1.0)
        data = {k: v.to(device) for k, v in data.items()}
        data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)
        print(f"Successfully loaded data with {len(data['rgb'])} images")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Print data shapes and statistics
    print("\n=== Data Statistics ===")
    print(f"RGB shape: {data['rgb'].shape}")
    print(f"Depth shape: {data['depth'].shape}")
    print(f"Alpha shape: {data['alpha'].shape}")
    print(f"Camera shape: {data['camera'].shape}")
    # print(f"Latent shape: {data['latent'].shape}")
    #
    print(f"RGB range: [{data['rgb'].min():.3f}, {data['rgb'].max():.3f}]")
    print(f"Depth range: [{data['depth'].min():.3f}, {data['depth'].max():.3f}]")
    print(f"Alpha range: [{data['alpha'].min():.3f}, {data['alpha'].max():.3f}]")
    
    # Generate point clouds
    print("\n=== Generating Point Clouds ===")
    # try:
    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    print(f"Generated point cloud with {len(points.coords)} points")
    print(f"Available channels: {list(points.channels.keys())}")
    # except Exception as e:
    #     print(f"Error generating point clouds: {e}")
    #     return None
    
    return data, points

def analyze_point_cloud(points):
    """
    Analyze the generated point cloud
    """
    print("\n=== Point Cloud Analysis ===")
    
    coords = points.coords
    channels = points.channels
    
    print(f"Total points: {len(coords)}")
    print(f"Coordinate range:")
    print(f"  X: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    print(f"  Y: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    print(f"  Z: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
    
    if 'R' in channels and 'G' in channels and 'B' in channels:
        rgb = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"RGB mean: [{rgb.mean(axis=0)}]")
    
    if 'A' in channels:
        alpha = channels['A']
        print(f"Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
        print(f"Alpha mean: {alpha.mean():.3f}")
        print(f"Points with alpha > 0.5: {np.sum(alpha > 0.5)}")
    
    # Calculate point density statistics
    from scipy.spatial.distance import pdist, squareform
    if len(coords) > 1000:
        # Sample a subset for distance calculation
        sample_indices = np.random.choice(len(coords), 1000, replace=False)
        sample_coords = coords[sample_indices]
        distances = pdist(sample_coords)
        print(f"Average distance between points (sampled): {distances.mean():.4f}")
        print(f"Min distance between points (sampled): {distances.min():.4f}")
        print(f"Max distance between points (sampled): {distances.max():.4f}")

def visualize_point_cloud_2d(points, save_path="point_cloud_2d.png"):
    """
    Create 2D scatter plots of the point cloud from different viewpoints
    """
    print(f"\n=== Creating 2D Visualizations ===")
    
    coords = points.coords
    channels = points.channels
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Point Cloud 2D Projections', fontsize=16)
    
    # XY projection
    if 'R' in channels and 'G' in channels and 'B' in channels:
        colors = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        colors = np.clip(colors, 0, 1)  # Ensure colors are in [0, 1]
    else:
        colors = 'blue'
    
    # XY projection
    axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=colors, s=0.1, alpha=0.6)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('XY Projection')
    axes[0, 0].set_aspect('equal')
    
    # XZ projection
    axes[0, 1].scatter(coords[:, 0], coords[:, 2], c=colors, s=0.1, alpha=0.6)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('XZ Projection')
    axes[0, 1].set_aspect('equal')
    
    # YZ projection
    axes[1, 0].scatter(coords[:, 1], coords[:, 2], c=colors, s=0.1, alpha=0.6)
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    axes[1, 0].set_title('YZ Projection')
    axes[1, 0].set_aspect('equal')
    
    # Color distribution
    if 'R' in channels and 'G' in channels and 'B' in channels:
        rgb = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        axes[1, 1].hist(rgb.flatten(), bins=50, alpha=0.7, label=['R', 'G', 'B'])
        axes[1, 1].set_xlabel('Color Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Color Distribution')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D visualization saved to {save_path}")
    plt.close()

def visualize_point_cloud_3d(points, save_path="point_cloud_3d.png"):
    """
    Create 3D visualization of the point cloud
    """
    print(f"\n=== Creating 3D Visualization ===")
    
    coords = points.coords
    channels = points.channels
    
    # Sample points if there are too many for visualization
    if len(coords) > 10000:
        sample_indices = np.random.choice(len(coords), 10000, replace=False)
        coords = coords[sample_indices]
        if 'R' in channels and 'G' in channels and 'B' in channels:
            colors = np.stack([channels['R'][sample_indices], 
                             channels['G'][sample_indices], 
                             channels['B'][sample_indices]], axis=1)
        else:
            colors = 'blue'
    else:
        if 'R' in channels and 'G' in channels and 'B' in channels:
            colors = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        else:
            colors = 'blue'
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=colors, s=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')
    
    # Set equal aspect ratio
    max_range = np.array([coords[:, 0].max()-coords[:, 0].min(),
                         coords[:, 1].max()-coords[:, 1].min(),
                         coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
    
    mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D visualization saved to {save_path}")
    plt.close()

def create_camera_frustum(camera_params, scale=0.1):
    """
    Create a camera frustum visualization from camera parameters
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available for camera frustum visualization")
        return None
    
    # Parse camera parameters
    H, W, intrinsics, c2w = parse_camera(camera_params.unsqueeze(0))
    H, W = int(H[0]), int(W[0])
    intrinsic = intrinsics[0]
    c2w = c2w[0]
    
    # Camera center in world coordinates
    camera_center = c2w[:3, 3].cpu().numpy()
    
    # Camera coordinate system vectors
    camera_x = c2w[:3, 0].cpu().numpy() * scale
    camera_y = c2w[:3, 1].cpu().numpy() * scale
    camera_z = c2w[:3, 2].cpu().numpy() * scale
    
    # Create frustum points
    focal_x, focal_y = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Near plane corners
    near_z = 0.1
    near_left = -cx * near_z / focal_x
    near_right = (W - cx) * near_z / focal_x
    near_top = -cy * near_z / focal_y
    near_bottom = (H - cy) * near_z / focal_y
    
    # Far plane corners
    far_z = 1.0
    far_left = -cx * far_z / focal_x
    far_right = (W - cx) * far_z / focal_x
    far_top = -cy * far_z / focal_y
    far_bottom = (H - cy) * far_z / focal_y
    
    # Create frustum vertices in camera space
    near_vertices = np.array([
        [near_left, near_top, near_z],
        [near_right, near_top, near_z],
        [near_right, near_bottom, near_z],
        [near_left, near_bottom, near_z]
    ])
    
    far_vertices = np.array([
        [far_left, far_top, far_z],
        [far_right, far_top, far_z],
        [far_right, far_bottom, far_z],
        [far_left, far_bottom, far_z]
    ])
    
    # Transform to world space
    near_vertices_world = (c2w[:3, :3].cpu().numpy() @ near_vertices.T).T + camera_center
    far_vertices_world = (c2w[:3, :3].cpu().numpy() @ far_vertices.T).T + camera_center
    
    # Create Open3D line set for frustum
    frustum_lines = []
    frustum_colors = []
    
    # Near plane
    for i in range(4):
        frustum_lines.append([i, (i + 1) % 4])
        frustum_colors.append([1, 0, 0])  # Red
    
    # Far plane
    for i in range(4):
        frustum_lines.append([i + 4, ((i + 1) % 4) + 4])
        frustum_colors.append([0, 1, 0])  # Green
    
    # Connecting lines
    for i in range(4):
        frustum_lines.append([i, i + 4])
        frustum_colors.append([0, 0, 1])  # Blue
    
    # Create coordinate axes
    axis_lines = []
    axis_colors = []
    
    # X axis (red)
    axis_lines.append([8, 9])
    axis_colors.append([1, 0, 0])
    
    # Y axis (green)
    axis_lines.append([10, 11])
    axis_colors.append([0, 1, 0])
    
    # Z axis (blue)
    axis_lines.append([12, 13])
    axis_colors.append([0, 0, 1])
    
    # Combine all vertices
    all_vertices = np.vstack([
        near_vertices_world,
        far_vertices_world,
        camera_center,
        camera_center + camera_x,
        camera_center,
        camera_center + camera_y,
        camera_center,
        camera_center + camera_z
    ])
    
    # Combine all lines
    all_lines = frustum_lines + axis_lines
    all_colors = frustum_colors + axis_colors
    
    # Create Open3D line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_vertices)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(all_colors)
    
    return line_set

def visualize_point_cloud_with_cameras(points, camera_params, save_path="point_cloud_with_cameras.png"):
    """
    Visualize point cloud with camera poses using Open3D
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available for enhanced visualization")
        return
    
    print(f"\n=== Creating Point Cloud with Camera Visualization ===")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.coords)
    
    # Add colors if available
    if 'R' in points.channels and 'G' in points.channels and 'B' in points.channels:
        colors = np.stack([points.channels['R'], points.channels['G'], points.channels['B']], axis=1)
        colors = np.clip(colors, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create camera frustums
    camera_geometries = []
    for i, camera_param in enumerate(camera_params):
        frustum = create_camera_frustum(camera_param)
        if frustum is not None:
            camera_geometries.append(frustum)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Combine all geometries
    geometries = [pcd, coord_frame] + camera_geometries
    
    # Visualize
    o3d.visualization.draw_geometries(geometries, 
                                     window_name="Point Cloud with Camera Poses",
                                     width=1200, height=800)
    
    # Save screenshot if possible
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1200, height=800)
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Set view to show all geometries
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().line_width = 3.0
        
        # Capture image
        vis.capture_screen_image(save_path, do_render=True)
        vis.destroy_window()
        print(f"Visualization saved to {save_path}")
    except Exception as e:
        print(f"Could not save visualization: {e}")

def render_point_cloud_from_camera(points, camera_params, output_dir="renders"):
    """
    Render the point cloud from each camera viewpoint
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available for rendering")
        return
    
    print(f"\n=== Rendering Point Cloud from Camera Views ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.coords)
    
    # Add colors if available
    if 'R' in points.channels and 'G' in points.channels and 'B' in points.channels:
        colors = np.stack([points.channels['R'], points.channels['G'], points.channels['B']], axis=1)
        colors = np.clip(colors, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Parse camera parameters
    H, W, intrinsics, c2ws = parse_camera(camera_params)
    
    for i, (h, w, intrinsic, c2w) in enumerate(zip(H, W, intrinsics, c2ws)):
        print(f"Rendering camera {i+1}/{len(camera_params)}")
        print(f"Camera params: {c2w}")
        print(f"intrinsic: {intrinsic}")
        
        # Convert to Open3D camera parameters
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=int(w), height=int(h),
            fx=float(intrinsic[0, 0]), fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]), cy=float(intrinsic[1, 2])
        )
        
        # Convert camera-to-world to world-to-camera
        S = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
        )

        c2w[:3, :3] =  c2w[:3, :3] @ S
        w2c = torch.linalg.inv(c2w)
        
        # Create Open3D camera extrinsic matrix
        extrinsic = w2c.cpu().numpy()
        
        # Create visualizer for rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=int(w), height=int(h))
        vis.add_geometry(pcd)
        
        # Set camera parameters
        ctr = vis.get_view_control()
        
        # Set camera pose (simplified approach)
        # Calculate camera position and look direction
        camera_pos = c2w[:3, 3].cpu().numpy()
        look_dir = -c2w[:3, 2].cpu().numpy()  # Negative Z direction
        
        # Set view parameters
        ctr.set_front(look_dir)
        ctr.set_lookat(camera_pos + look_dir)  # Look at point in front of camera
        ctr.set_up(c2w[:3, 1].cpu().numpy())  # Up vector
        
        # Render settings
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0, 0, 0])  # Black background
        
        # Capture image
        output_path = os.path.join(output_dir, f"render_camera_{i:03d}.png")
        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()
        
        print(f"  Saved to {output_path}")
    
    print(f"Rendering complete. Images saved to {output_dir}/")

def create_camera_trajectory_visualization(camera_params, save_path="camera_trajectory.png"):
    """
    Create a 2D visualization of camera trajectory
    """
    print(f"\n=== Creating Camera Trajectory Visualization ===")
    
    # Parse camera parameters
    H, W, intrinsics, c2ws = parse_camera(camera_params)
    
    # Extract camera centers
    camera_centers = []
    for c2w in c2ws:
        center = c2w[:3, 3].cpu().numpy()
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Camera Trajectory Visualization', fontsize=16)
    
    # XY projection
    axes[0, 0].plot(camera_centers[:, 0], camera_centers[:, 1], 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].scatter(camera_centers[:, 0], camera_centers[:, 1], c=range(len(camera_centers)), 
                      cmap='viridis', s=50, alpha=0.8)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Camera Trajectory (XY)')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # XZ projection
    axes[0, 1].plot(camera_centers[:, 0], camera_centers[:, 2], 'r-', linewidth=2, alpha=0.7)
    axes[0, 1].scatter(camera_centers[:, 0], camera_centers[:, 2], c=range(len(camera_centers)), 
                      cmap='viridis', s=50, alpha=0.8)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('Camera Trajectory (XZ)')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # YZ projection
    axes[1, 0].plot(camera_centers[:, 1], camera_centers[:, 2], 'g-', linewidth=2, alpha=0.7)
    axes[1, 0].scatter(camera_centers[:, 1], camera_centers[:, 2], c=range(len(camera_centers)), 
                      cmap='viridis', s=50, alpha=0.8)
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    axes[1, 0].set_title('Camera Trajectory (YZ)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3D trajectory
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], 
             'b-', linewidth=2, alpha=0.7)
    scatter = ax3d.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], 
                          c=range(len(camera_centers)), cmap='viridis', s=50, alpha=0.8)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('Camera Trajectory (3D)')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax3d, label='Camera Index')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Camera trajectory visualization saved to {save_path}")
    plt.close()

def export_point_cloud_for_external_viewer(points, output_path="point_cloud_for_viewer.ply"):
    """
    Export point cloud in a format suitable for external viewers like MeshLab
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available for PLY export")
        return
    
    print(f"\n=== Exporting Point Cloud for External Viewer ===")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.coords)
    
    # Add colors if available
    if 'R' in points.channels and 'G' in points.channels and 'B' in points.channels:
        colors = np.stack([points.channels['R'], points.channels['G'], points.channels['B']], axis=1)
        colors = np.clip(colors, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Export as PLY
    success = o3d.io.write_point_cloud(output_path, pcd)
    if success:
        print(f"Point cloud exported to {output_path}")
        print("You can open this file in MeshLab or other 3D viewers")
    else:
        print(f"Failed to export point cloud to {output_path}")

def test_point_cloud_operations(points):
    """
    Test various point cloud operations
    """
    print("\n=== Testing Point Cloud Operations ===")
    
    # Test random sampling
    print("Testing random sampling...")
    sampled_points = points.random_sample(1000)
    print(f"Sampled {len(sampled_points.coords)} points from {len(points.coords)} original points")
    
    # Test farthest point sampling
    print("Testing farthest point sampling...")
    try:
        fps_points = points.farthest_point_sample(1000)
        print(f"Farthest point sampling: {len(fps_points.coords)} points")
    except Exception as e:
        print(f"Farthest point sampling failed: {e}")
    
    # Test saving and loading
    print("Testing save/load functionality...")
    try:
        points.save("test_point_cloud.npz")
        loaded_points = PointCloud.load("test_point_cloud.npz")
        print(f"Save/load successful: {len(loaded_points.coords)} points loaded")
        
        # Clean up
        os.remove("test_point_cloud.npz")
    except Exception as e:
        print(f"Save/load test failed: {e}")
    
    # Test PLY export
    print("Testing PLY export...")
    try:
        with open("test_point_cloud.ply", "wb") as f:
            points.write_ply(f)
        print("PLY export successful")
        
        # # Clean up
        # os.remove("test_point_cloud.ply")
    except Exception as e:
        print(f"PLY export failed: {e}")

def compare_with_original_data(data, points):
    """
    Compare the generated point cloud with the original data
    """
    print("\n=== Comparing with Original Data ===")
    
    # Check if point cloud coordinates match depth values
    coords = points.coords
    
    # Sample a few points and check their depth values
    if len(coords) > 0:
        print(f"Point cloud coordinate statistics:")
        print(f"  X: mean={coords[:, 0].mean():.3f}, std={coords[:, 0].std():.3f}")
        print(f"  Y: mean={coords[:, 1].mean():.3f}, std={coords[:, 1].std():.3f}")
        print(f"  Z: mean={coords[:, 2].mean():.3f}, std={coords[:, 2].std():.3f}")
        
        # Check if Z coordinates match depth range
        depth_range = data['depth_range']
        print(f"Original depth range: {depth_range[0].cpu().numpy()}")
        print(f"Point cloud Z range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
        
        # Check alpha channel consistency
        if 'A' in points.channels:
            alpha_values = points.channels['A']
            original_alpha = data['alpha'].cpu().numpy()
            print(f"Original alpha range: [{original_alpha.min():.3f}, {original_alpha.max():.3f}]")
            print(f"Point cloud alpha range: [{alpha_values.min():.3f}, {alpha_values.max():.3f}]")

def main():
    """
    Main function to run all tests
    """
    print("Starting Point Cloud Testing Script")
    print("=" * 50)
    
    # Test point cloud generation
    result = test_point_cloud_generation()
    if result is None:
        print("Failed to generate point clouds. Exiting.")
        return
    
    data, points = result
    
    # Analyze the point cloud
    analyze_point_cloud(points)
    
    # Create basic visualizations
    visualize_point_cloud_2d(points)
    visualize_point_cloud_3d(points)
    
    # Enhanced visualization with camera poses
    if OPEN3D_AVAILABLE:
        print("\n=== Enhanced Visualization with Camera Poses ===")
        visualize_point_cloud_with_cameras(points, data['camera'])
        create_camera_trajectory_visualization(data['camera'])
        # render_point_cloud_from_camera(points, data['camera'])
        # export_point_cloud_for_external_viewer(points)
    else:
        print("\n=== Skipping Enhanced Visualization (Open3D not available) ===")
        print("Install Open3D for camera pose visualization and rendering:")
        print("pip install open3d")
    
    # Test point cloud operations
    test_point_cloud_operations(points)
    
    # Compare with original data
    compare_with_original_data(data, points)
    
    print("\n" + "=" * 50)
    print("Point Cloud Testing Complete!")
    print("Generated files:")
    print("- point_cloud_2d.png: 2D projections and color distribution")
    print("- point_cloud_3d.png: 3D visualization")
    if OPEN3D_AVAILABLE:
        print("- point_cloud_with_cameras.png: Point cloud with camera poses")
        print("- camera_trajectory.png: Camera trajectory visualization")
        print("- renders/: Rendered images from each camera viewpoint")
        print("- point_cloud_for_viewer.ply: Point cloud for external viewers")

if __name__ == "__main__":
    main()
