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
from datasets.spherical_coords import SphericalCoords

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
    

    def correct_intensity(self, points, intensity):
        """ This function corrects the intensity value, removing the bias of the distance between point and sensor. """
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        cos_phi = points[:, 2] / r
        intensity_corr = intensity / (r**2 * cos_phi)
        return intensity_corr

    def dsor_filter(self, points, base_k=10, scaling_factor=0.5, std_ratio=2.0):
        """
        DSOR: Dynamic Statistical Outlier Removal.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        distances = []
        neighbors_list = []

        for i, point in enumerate(points):
            dist = np.linalg.norm(point)
            k = int(base_k + scaling_factor * dist)  # k dinámico según distancia
            k = min(k, len(points) - 1)  # para evitar exceder el tamaño de la nube

            [_, idxs, dists] = kdtree.search_knn_vector_3d(point, k)
            if len(dists) > 0:
                avg_dist = np.mean(np.sqrt(dists[1:]))  # quitamos el primero (distancia a sí mismo)
                distances.append(avg_dist)
                neighbors_list.append(i)

        # Calcular media y desviación globales de distancias promedio
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dev = np.std(distances)

        threshold = mean_dist + std_ratio * std_dev

        # Elegimos solo los puntos cuya distancia media a vecinos es razonable
        inlier_indices = [neighbors_list[i] for i in range(len(distances)) if distances[i] < threshold]
        filtered_pcd = pcd.select_by_index(inlier_indices)

        return filtered_pcd, inlier_indices

    def read_pc(self, device, file_pathname: str) -> np.ndarray:
        if PARAMS.format_point_cloud == 'csv':
            # SE RECIBE LA PC EN CSV
            if not PARAMS.use_downsampled:
                df = pd.read_csv(file_pathname)
            else:
                if PARAMS.protocol == 'usyd' and PARAMS.spherical_coords:
                    file_pathname = file_pathname.replace("USyd/","USyd_downsample/").replace(".bin", ".csv")
                elif PARAMS.protocol == 'usyd' and not PARAMS.spherical_coords:
                    file_pathname = file_pathname.replace("USyd/","USyd_downsample/").replace(".bin", "_NO_SP.csv")
                else:
                    file_pathname = file_pathname.replace(PARAMS.protocol, PARAMS.protocol+"_downsample")
                df = pd.read_csv(file_pathname)
            df.columns = df.columns.str.lower().str.strip()
            if PARAMS.protocol == 'blt' and PARAMS.use_2D:
                points = np.column_stack([df['x'], df['y'], np.zeros(len(df))])
            else:
                df = df.query('x != 0 and y != 0 and z != 0 and intensity != 0')
                points = df[["x", "y", "z"]].to_numpy() 
            #print(len(points)) # Number of points in cloud
            if PARAMS.protocol == 'blt' and PARAMS.use_2D:
                intensity = np.zeros(len(df))
            else:
                intensity = df["intensity"].to_numpy()
            if not PARAMS.use_downsampled:
                if PARAMS.correct_intensity:
                    intensity = self.correct_intensity(points, intensity)
                if PARAMS.protocol == 'vmd' or PARAMS.protocol == 'blt':
                    # Remove noisy points
                    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                    r2 = x ** 2 + y ** 2
                    idx = np.where((r2 < PARAMS.max_distance ** 2))
                    points = points[idx]
                    intensity = intensity[idx]
                if PARAMS.protocol == 'arvc': # Remove noise with minimum and maximum radius
                    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                    r2 = x ** 2 + y ** 2
                    idx = np.where(r2 < PARAMS.max_distance ** 2) and np.where(r2 > PARAMS.min_distance ** 2)
                    points = points[idx]
                    intensity = intensity[idx]
                if PARAMS.protocol != 'usyd' and PARAMS.spherical_coords:
                    points = np.concatenate((points, intensity[:, np.newaxis]), axis=1)
                    points = SphericalCoords.to_spherical(points, PARAMS.protocol)
                    intensity = points[:, 3]
                    points = points[:, :3]
            if PARAMS.correct_intensity:
                intensity = self.correct_intensity(points, intensity)
            if not PARAMS.use_downsampled:
                if PARAMS.equalize_intensity:
                    intensity = exposure.equalize_hist(intensity)
            if PARAMS.normalize:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                points = self.global_normalize(pcd)
            return points, intensity
        
        else:
            # SE RECIBE LA PC COMO BINARIO:
            # Load point cloud, does not apply any transform
            # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
            start = time.time()
            file_path = os.path.join(file_pathname)
            if PARAMS.protocol == 'nclt':
                dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                data = np.fromfile(file_path, dtype=dtype)
                points = np.stack([data['x'] * 0.005 - 100, data['y'] * 0.005 - 100, (data['z'] * 0.005 - 100)], axis=-1)

                intensity =  data['intensity']
                # Parámetro: umbral de altura para eliminar el suelo
                ground_threshold = 0.5  # metros
                [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                r2 = x ** 2 + y ** 2
                idx = (r2 < 80**2) & (z < -ground_threshold) # Procesado según paper: DiSCO
                points = points[idx]
                intensity = intensity[idx]
                pc = np.column_stack((points, intensity))
            elif PARAMS.protocol == 'intensityOxford':
                pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 4])
            else:
                pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4]) #USYD, NCLT: float32, OXFORD: float64
            #pc = pc[np.linalg.norm(pc[:, :3], axis=1) < PARAMS.max_distance]
            np.random.shuffle(pc)
            if pc.shape[0] == 0:
                pc = np.zeros((1, 4), dtype=np.float32)
            else:
                if PARAMS.correct_intensity:
                    intensity = self.correct_intensity(pc[:, :3], pc[:, 3])
                    pc = np.concatenate((pc[:, :3], intensity[:, np.newaxis]), axis=1)
                if PARAMS.spherical_coords:
                    pc = SphericalCoords.to_spherical(pc, PARAMS.protocol)
                if PARAMS.equalize_intensity:
                    pc[:, 3] = exposure.equalize_hist(pc[:, 3])
            return pc[:, :3], pc[:, 3]

    def dror_filter(self, points, base_radius=0.1, scaling_factor=0.03, min_neighbors=3):
        """
        DROR: Dynamic Radius Outlier Removal.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        inlier_indices = []

        for i, point in enumerate(points):
            dist = np.linalg.norm(point)
            dynamic_radius = base_radius + scaling_factor * dist

            [_, idxs, _] = kdtree.search_radius_vector_3d(point, dynamic_radius)

            if len(idxs) > min_neighbors:
                inlier_indices.append(i)

        filtered_pcd = pcd.select_by_index(inlier_indices)
        return filtered_pcd, inlier_indices
    
    
    def read_intensity(self, intensity):
        intensity_norm = []
        for i in intensity:
            if i == 0:
                intensity_norm.append(0)
            else:
                intensity_norm.append(1)
        intensity_norm = np.array(intensity_norm)
        return intensity_norm
