# Judith Vilella Cantos. Miguel Hernández University of Elche.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import open3d as o3d
from skimage import exposure
from spherical_coords import SphericalCoords
from config.config import PARAMS

# Ruta al dataset original y carpeta destino
input_folder = "/media/arvc/HDD4TB/Judith/MinkUNeXt-main/blt/ktima"
output_folder = "/media/arvc/HDD4TB/Judith/MinkUNeXt-main/blt_downsample/ktima"

# Crear la carpeta destino si no existe
os.makedirs(output_folder, exist_ok=True)

# Parámetro para reducción de resolución
voxel_size = 0.1  # Cambia según la densidad que desees

def correct_intensity(points, intensity):
    """ This function corrects the intensity value, removing the bias of the distance between point and sensor. """
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    cos_phi = points[:, 2] / r
    intensity_corr = intensity / (r**2 * cos_phi)
    return intensity_corr

def dror_filter(points, base_radius=0.1, scaling_factor=0.03, min_neighbors=3):
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

def read_bin(file_path):
    """Lee un archivo .bin con formato de nubes de puntos."""
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # X, Y, Z, intensidad

def write_bin(file_path, pcd):
    """Escribe un archivo .bin con formato de nubes de puntos."""
    #points.astype(np.float32).tofile(file_path)
    """Para guardar la pointcloud como .pcd"""
    pcd.to_csv(file_path, index=False, float_format="%.6f")

def reduce_resolution(points, intensity):
    #intensity = correct_intensity(points, intensity)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)
    pcd, ind = dror_filter(pcd.points)
    points = np.asarray(pcd.points)

    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
    r2 = x ** 2 + y ** 2
    idx = np.where((r2 < PARAMS.max_distance ** 2))
    points = points[idx]
    intensity = intensity[idx]
    points = np.concatenate((points, intensity[:, np.newaxis]), axis=1)
    points = SphericalCoords.to_spherical(points, "blt")
    intensity = exposure.equalize_hist(points[:, 3])

    return points[:, :3], intensity

if __name__ == '__main__':
    for folder in os.listdir(input_folder):
        folder_to_process = os.path.join(input_folder,folder,"robot0/lidar_3d/data")
        for file_name in os.listdir(folder_to_process):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_folder,folder,"robot0/lidar_3d/data", file_name)
                output_folder_part = os.path.join(output_folder,folder,"robot0/lidar_3d/data")
                output_path = os.path.join(output_folder_part, file_name)
                #output_path = output_path.replace(".bin", "_NO_SP.csv")

                os.makedirs(output_folder_part, exist_ok=True)
                # Leer nube de puntos
                df = pd.read_csv(input_path)
                #pc = read_bin(input_path)
                df = df.query('x != 0 and y != 0 and z != 0 and intensity != 0')

                points = df[['x', 'y', 'z']].values
                intensity = df['intensity'].values
                #points = pc[:, :3]
                #intensity = pc[:, 3]
                # Reducir resolución
                points, intensity = reduce_resolution(points, intensity)
                if points.size > 0:
                    df = pd.DataFrame(points, columns=["x", "y", "z"])
                    df["intensity"] = intensity

                    # Guardar nube de puntos reducida
                    write_bin(output_path, df)
                    print(f"Procesado: {file_name}")
