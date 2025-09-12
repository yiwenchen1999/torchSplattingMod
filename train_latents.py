import torch
import numpy as np
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render_latents import GaussRenderer
import cv2
import contextlib

from torch.profiler import profile, ProfilerActivity

USE_GPU_PYTORCH = True
USE_PROFILE = False

class GSSTrainer(Trainer):
    def __init__(self, latent_model=False, image_size=64, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(image_size=image_size, **kwargs.get('render_kwargs', {}))
        self.latent_model = latent_model
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
    
    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        save_path = self.results_folder / f'general_eval'
        save_path.mkdir(exist_ok=True)
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)
        # if self.latent_model:
        #     camera.image_width = 64
        #     camera.image_height = 64

        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera, latent_model=self.latent_model, feature_dim=self.model.feature_dim)
        rgb_pd = out['render'].detach().cpu().numpy()[..., :3]
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        # if the shape does not match, resize the depth_pd
        if depth.shape != depth_pd.shape:
            depth_pd = cv2.resize(depth_pd, (depth.shape[1], depth.shape[0]))
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]
        if rgb.shape != rgb_pd.shape:
            rgb_pd = cv2.resize(rgb_pd, (rgb.shape[1], rgb.shape[0]))
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(save_path / f'image-{self.step}.png'), image)
        
        if self.latent_model:
            rgb_pdnp = out['render'].detach().cpu().numpy()
            # Rearrange from (H,W,C) to (C,H,W)
            rgb_pdnp = np.transpose(rgb_pdnp, (2, 0, 1))
            # save as float16
            print('rgb_pdnp', rgb_pdnp.shape, rgb_pdnp.dtype)
            rgb_pdnp = rgb_pdnp.astype(np.float32)
            np.save(str(save_path / f'image-{self.step}.npy'), rgb_pdnp)

    def evaluate_all_samples(self, step):
        """
        Evaluate on all instances in the dataset instead of randomly sampling one.
        Creates a new folder named after the current step to store results.
        Names results after the sampled index.
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create evaluation folder for this step
        eval_folder = self.results_folder / f'eval_step_{step}'
        eval_folder.mkdir(exist_ok=True)
        
        print(f"Starting comprehensive evaluation at step {step} for all {len(self.data['camera'])} samples...")
        
        # Evaluate on all samples
        for ind in range(len(self.data['camera'])):
            camera = self.data['camera'][ind]
            if USE_GPU_PYTORCH:
                camera = to_viewpoint_camera(camera)
            
            rgb = self.data['rgb'][ind].detach().cpu().numpy()
            out = self.gaussRender(pc=self.model, camera=camera, latent_model=self.latent_model, feature_dim=self.model.feature_dim)
            rgb_pd = out['render'].detach().cpu().numpy()[..., :3]
            depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
            depth = self.data['depth'][ind].detach().cpu().numpy()
            
            # Handle shape mismatches
            if depth.shape != depth_pd.shape:
                depth_pd = cv2.resize(depth_pd, (depth.shape[1], depth.shape[0]))
            if rgb.shape != rgb_pd.shape:
                rgb_pd = cv2.resize(rgb_pd, (rgb.shape[1], rgb.shape[0]))
            
            # Create depth visualization
            depth_combined = np.concatenate([depth, depth_pd], axis=1)
            depth_combined = (1 - depth_combined / depth_combined.max())
            depth_combined = plt.get_cmap('jet')(depth_combined)[..., :3]
            
            # Create combined image
            image = np.concatenate([rgb, rgb_pd], axis=1)
            image = np.concatenate([image, depth_combined], axis=0)
            
            # Save image with sample index
            utils.imwrite(str(eval_folder / f'sample_{ind:04d}.png'), image)
            
            # Save latent output if using latent model
            if self.latent_model:
                rgb_pdnp = out['render'].detach().cpu().numpy()
                rgb_pdnp = np.transpose(rgb_pdnp, (2, 0, 1))
                rgb_pdnp = rgb_pdnp.astype(np.float32)
                np.save(str(eval_folder / f'sample_{ind:04d}.npy'), rgb_pdnp)
            
            if (ind + 1) % 50 == 0:
                print(f"Processed {ind + 1}/{len(self.data['camera'])} samples...")
                
        print(f"Comprehensive evaluation completed at step {step}. Results saved in {eval_folder}")

    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        mask = (self.data['alpha'][ind] > 0.5)
        if self.latent_model:
            latent = self.data['latent'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            # if self.latent_model:
            #     camera.image_width = 64
            #     camera.image_height = 64
            out = self.gaussRender(pc=self.model, camera=camera, latent_model=self.latent_model, feature_dim=self.model.feature_dim)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))


        #^ l1_loss = loss_utils.l1_loss(out['render'], rgb)
        #^ depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        #^ ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)
        # print('out render shape', out['render'].shape, 'latent shape', latent.shape)
        if self.latent_model:
            gt = latent
        else:
            gt = rgb
        l1_loss = loss_utils.l1_loss(out['render'], gt)

        #^ total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        #^ psnr = utils.img2psnr(out['render'], rgb)
        #^ log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr}
        psnr = utils.img2psnr(out['render'], gt)
        total_loss = l1_loss
        log_dict = {'total': total_loss,'l1':l1_loss, 'psnr': psnr}

        # Check if we should do comprehensive evaluation
        if self.step % 50000 == 0:
            print(f"Step {self.step} reached - starting comprehensive evaluation...")
            self.evaluate_all_samples(self.step)

        return total_loss, log_dict

    def save_gaussian_xyz_as_pointcloud(self, step):
        """
        Save the gaussian model's _xyz as a point cloud
        """
        if True:
            # Get the xyz coordinates from the gaussian model
            xyz = self.model._xyz.detach().cpu().numpy()
            
            # Get other gaussian parameters if available
            features = None
            if hasattr(self.model, '_features_dc') and self.model._features_dc is not None:
                features = self.model._features_dc.detach().cpu().numpy()
            
            # Create a simple point cloud with xyz coordinates
            point_cloud_data = {
                'xyz': xyz,
                'features': features
            }
            
            # Save as numpy file
            np.save(str(self.results_folder / f'gaussian_xyz_step_{step}.npy'), point_cloud_data)
            
            # Also save as PLY file for visualization
            self.save_xyz_as_ply(xyz, features, step)
            
            print(f"Saved gaussian xyz point cloud at step {step}")
            
        # except Exception as e:
        #     print(f"Error saving gaussian xyz point cloud: {e}")
    
    def save_xyz_as_ply(self, xyz, features=None, step=None):
        """
        Save xyz coordinates as PLY file for visualization
        """
        if True:
            if features is not None:
                features = features[:, 0, :]
            # Create PLY header
            num_points = len(xyz)
            ply_header = f"""ply
                format ascii 1.0
                element vertex {num_points}
                property float x
                property float y
                property float z"""
            
            # Add color properties if features are available
            if features is not None and len(features.shape) >= 2:
                ply_header += """
                    property uchar red
                    property uchar green
                    property uchar blue"""
            
            ply_header += "\nend_header\n"
            
            # Write PLY file
            ply_filename = f'gaussian_xyz_step_{step}.ply' if step is not None else 'gaussian_xyz.ply'
            ply_path = str(self.results_folder / ply_filename)
            
            with open(ply_path, 'w') as f:
                f.write(ply_header)
                
                for i in range(num_points):
                    x, y, z = xyz[i]
                    line = f"{x:.6f} {y:.6f} {z:.6f}"
                    
                    # Add colors if features are available
                    if features is not None and len(features.shape) >= 2:
                        # Convert features to RGB (assuming first 3 components)
                        
                        if features.shape[1] >= 3:
                            r = int(max(0, min(255, features[i, 0] * 255)))
                            g = int(max(0, min(255, features[i, 1] * 255)))
                            b = int(max(0, min(255, features[i, 2] * 255)))
                            line += f" {r} {g} {b}"
                        else:
                            # Use grayscale if less than 3 features
                            val = int(max(0, min(255, features[i, 0] * 255)))
                            line += f" {val} {val} {val}"
                    
                    f.write(line + "\n")
            
            print(f"Saved PLY file: {ply_path}")
            
        # except Exception as e:
        #     print(f"Error saving PLY file: {e}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train Gaussian Splatting model with latents')
    parser.add_argument('--folder', type=str, default='../objaverse_synthetic',
                       help='Path to dataset folder')
    parser.add_argument('--scene_name', type=str, default=None,
                       help='Scene name for output folder')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size for training')
    parser.add_argument('--input_folder', type=str, default=None, required=False,
                        help='specify what folder to use')
    parser.add_argument('--feature_dim', type=int, default=4,
                       help='Feature dimension for training')
    
    args = parser.parse_args()
    
    device = 'cuda'
    if args.input_folder is not None:
        folder = args.input_folder
    else:
        folder = os.path.join(args.folder, f'{args.scene_name}_processed_train')
    image_size = args.image_size
    
    if args.scene_name is None:
        scene_name = f'scene_{image_size}'
    else:
        scene_name = f'{args.scene_name}_{image_size}'
    
    # folder = 'B075X65R3X'
    # scene_name = 'chair_rgb'
    latent_model = True
    data = read_all(folder, resize_factor=4*image_size/512.0, latent_model=latent_model, image_size=image_size)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)

    if 'nerf' in folder:
        points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'], regularize_rays=True)
    else:
        points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    raw_points = points.random_sample(2**14)
    # raw_points.write_ply(open('points.ply', 'wb'))

    gaussModel = GaussModel(sh_degree=4, debug=False, latent_model=latent_model, feature_dim=args.feature_dim)
    gaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': True,
    }

    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=100001,
        i_image =1000,
        train_lr=1e-3, 
        amp=False,
        fp16=False,
        results_folder=f'result/{scene_name}',
        latent_model=latent_model,
        image_size=image_size,
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()
    trainer.train()