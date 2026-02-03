import os
import re
import sys
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import packaging
import torch
from scripts.common import read_image, write_image
import trimesh
from util import save_mesh
import pysdf

class DataSampler:
    def sample_batch(self, n_samples: int, device: str):
        """
        samples input positions and corresponding field values
        
        Args
            - n_samples: number of samples
            - device: name of device (e.g. "cpu", "cuda:0") on which the sampled data will live

        Returns:
            - inputs: sample positions. torch tensor of shape [n_samples, data_input_dim], e.g., [n_samples, 2] for image xy positions
            - targets: values of the field sampled at inputs. torch tensor of shape [n_samples, data_output_dim], e.g., [n_samples, 3] for image RGB values
        """
        raise NotImplementedError
    def save_model_output(self, model: torch.nn.Module, save_path: str):
        """
        saves a reconstruction of the given neural field at the given path (without extension). assumes that the directory exists

        Args: 
            - model: a neural field nn.Module()
        
        Returns nothing
        """
        raise NotImplementedError
    
class Image(DataSampler, torch.nn.Module):
    def __init__(self, filename, device):
        super().__init__()
        self.data = read_image(filename)
        # remove alpha channel
        if self.data.ndim > 2 and self.data.shape[2] == 4:
            self.data = self.data[:,:,:3] # keep RGB
        self.orig_data_npy = self.data
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)
        self.device = device

        # for model output reconsturction
        resolution = self.data.shape[0:2]
        # n_pixels = resolution[0] * resolution[1]
        half_dx =  0.5 / resolution[0] # half pixel size
        half_dy =  0.5 / resolution[1]
        xs = torch.linspace(half_dx, 1-half_dx, resolution[0])
        ys = torch.linspace(half_dy, 1-half_dy, resolution[1])
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        self.img_shape = resolution + torch.Size([self.data.shape[2]])
        self.xy = torch.stack((yv.flatten(), xv.flatten())).t()

    def forward(self, xs, interpolate=True):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()

            x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
            if interpolate:
                lerp_weights = xs - indices.float()
                x1 = (x0 + 1).clamp(max=shape[1]-1)
                y1 = (y0 + 1).clamp(max=shape[0]-1)

                return (
                    self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
                    self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
                    self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
                    self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
                )
            
            return (self.data[y0, x0]) # no interpolation
    
    def sample_batch(self, n_samples, device):
        assert device == self.device
        input_xy = torch.rand([n_samples, 2], dtype=torch.float) 
        image_rgb = self.forward(input_xy)
        return input_xy, image_rgb

    def save_model_output(self, model, save_path):
        write_image(f"{save_path}.png", model(self.xy).reshape(self.img_shape).clamp(0.0, 1.0).detach().cpu().numpy())


class SDF(DataSampler):
    def __init__(self, path, device, num_samples=2**18, clip_sdf=None, transformation_save_dir=""):
        super().__init__()
        self.path = path
        self.device = self.device

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1]) via scaling and translation
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95

        # TODO: save normalizing transformation of base model and apply same transformation to deformed model 
        transformation_path = f"{transformation_save_dir}/data_normalization_transformation.npz"
        if not os.path.exists(transformation_path):
            print("saving base mesh normalization transformation")
            np.savez(transformation_path, v_scale=v_scale, v_center=v_center)
        else:
            print("######################")
            print("####### TEMP DISABLED CONSISTENT SDF NORMALIZATION")
            print("######################")
            print("######################")
            print("######################")
            print("######################")
        #     print("using existing normalization transformation")
        #     loaded_transformation = np.load(transformation_path)
        #     v_scale = loaded_transformation["v_scale"]
        #     v_center = loaded_transformation["v_center"]

        print("scale: ", v_scale)
        print("center: ", v_center)
        # apply transformation to verts
        vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh verts & faces: {self.mesh.vertices.shape} & {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

    def sample_batch(self, n_samples, device):
        assert device == self.device
        # online sampling
        sdfs = np.zeros((n_samples, 1))
        # surface query points (7/8 points for surface and near-surface)
        points_surface = self.mesh.sample(n_samples * 7 // 8)
        
        # near-surface points
        points_surface[n_samples // 2:] += 0.01 * np.random.randn(n_samples * 3 // 8, 3)
        # random uniform points (1/8 of points)
        points_uniform = np.random.rand(n_samples // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)
        sdfs[n_samples // 2:] = -self.sdf_fn(points[n_samples // 2:])[:,None].astype(np.float32)
 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        return torch.tensor(points, device=device), torch.tensor(sdfs, device=device)

    def save_model_output(self, model: torch.nn.Module, save_path: str):
        save_mesh(save_path, model, self.device, resolution=256)