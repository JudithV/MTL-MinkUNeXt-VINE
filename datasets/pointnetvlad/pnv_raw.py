import copy
import time
import numpy as np
import os
import open3d as o3d
import open3d.core as o3c
import pandas as pd
import cv2
#import cupy as cp
import matplotlib.pyplot as plt
from skimage import exposure
import torch
from config import PARAMS 

from datasets.base_datasets import PointCloudLoader

class PNVPointCloudLoader(PointCloudLoader):
    def global_normalize(self, pcd):
        """
        Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
        """
        pcd = copy.deepcopy(pcd)
        points = np.asarray(pcd.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)
        norm_factor = 60.0
        
        x = x - x_mean
        y = y - y_mean
        z = z - z_mean

        x = x / norm_factor
        y = y / norm_factor
        z = z / norm_factor

        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z

        return points
    
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None
    
    def read_pc(self, device, file_pathname: str) -> np.ndarray:
        if PARAMS.format_point_cloud == 'csv':
            # SE RECIBE LA PC EN CSV
            if not PARAMS.use_downsampled:
                df = pd.read_csv(file_pathname)
            else:
                file_pathname = file_pathname.replace(PARAMS.protocol, PARAMS.protocol+"_downsample")
                df = pd.read_csv(file_pathname)
            df.columns = df.columns.str.lower().str.strip()
            df = df.query('x != 0 and y != 0 and z != 0 and intensity != 0')
            points = df[["x", "y", "z"]].to_numpy() 
            #print(len(points)) # Number of points in cloud
            intensity = df["intensity"].to_numpy()
            if not PARAMS.use_downsampled:
                if PARAMS.protocol == 'vmd' or PARAMS.protocol == 'blt':
                    # Remove noisy points
                    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                    r2 = x ** 2 + y ** 2
                    idx = np.where((r2 < PARAMS.max_distance ** 2))
                    points = points[idx]
                    intensity = intensity[idx]
            if PARAMS.normalize:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                points = self.global_normalize(pcd)
            return points, intensity
