import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy.spatial import Delaunay

# Ruta al dataset original y carpeta destino
input_folder = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/vmd/vineyard"
output_folder = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/vmd_interpolated/vineyard"

# Crear la carpeta destino si no existe
os.makedirs(output_folder, exist_ok=True)

def interpolate_pointcloud(xyz):
    # Usamos Delaunay para dividir el espacio en tetraedros
    triangulation = Delaunay(xyz)

    # Tomar un subconjunto aleatorio del 30% de los tetraedros
    num_tetraedros = len(triangulation.simplices)
    num_tetraedros_a_usar = int(num_tetraedros * 0.3)  # Solo usamos el 30% de los tetraedros
    tetraedros_indices = np.random.choice(num_tetraedros, num_tetraedros_a_usar, replace=False)


    # Generar puntos dentro de los tetraedros
    nuevos_puntos = []
    for simplex_idx in tetraedros_indices:
        vertices = xyz[triangulation.simplices[simplex_idx]]
        # Puntos de muestra dentro de cada tetraedro
        random_weights = np.random.rand(4)
        random_weights /= random_weights.sum()  # Normalizar para obtener pesos válidos
        punto_nuevo = np.dot(random_weights, vertices)  # Extrapolación lineal
        nuevos_puntos.append(punto_nuevo)

    # Convertir los nuevos puntos a un array numpy
    nuevos_puntos = np.array(nuevos_puntos)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nuevos_puntos)
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)
    pc = np.asarray(pcd.points)
    #pc = points[:, :3]
    min_radius = 1.0
    max_radius = 60.0
    [x, y, z] = pc[:, 0], pc[:, 1], pc[:, 2]
    r2 = x ** 2 + y ** 2
    idx = np.where(r2 < max_radius ** 2) and np.where(r2 > min_radius ** 2)
    pcd.points = o3d.utility.Vector3dVector(pc[idx])
    nuevos_puntos = pcd.points
    # Visualizar la nube de puntos
    #o3d.visualization.draw_geometries([pcd])

    # Verificar cuántos valores no son NaN después de la interpolación
    print("Número de píxeles válidos en ZI después de la interpolación:", np.count_nonzero(~np.isnan(nuevos_puntos)))

    return np.asarray(nuevos_puntos)

if __name__ == '__main__':
    for folder in os.listdir(input_folder):
        folder_to_process = os.path.join(input_folder+"/"+folder+"/pointcloud/lidar3d_1")
        for file_name in os.listdir(folder_to_process):
            if file_name.endswith(".csv"):     
                input_path = os.path.join(input_folder+"/"+folder+"/pointcloud/lidar3d_1", file_name)
                output_path = os.path.join(output_folder+"/"+folder+"/pointcloud/lidar3d_1", file_name)

                # Leer nube de puntos       
                df = pd.read_csv(input_path)
                df = df.query('x != 0 and y != 0 and z != 0')
                # Extraer las coordenadas (x, y, z) y la intensidad
                # Obtener las coordenadas x, y, z
                xyz = df[['x', 'y', 'z']].values
                intensity = df['intensity'].values

                # Reducir resolución
                new_intensity = []
                nuevos_puntos = interpolate_pointcloud(xyz)
                
                neighbors = NearestNeighbors(n_neighbors=3)
                neighbors.fit(xyz)

                for punto in nuevos_puntos:
                    dist, indices = neighbors.kneighbors([punto])
                    intensidad_promedio = intensity[indices].mean()
                    new_intensity.append(intensidad_promedio)
                new_intensity = np.array(new_intensity)
                if nuevos_puntos.size > 0:
                    df = pd.DataFrame(nuevos_puntos, columns=['x', 'y', 'z', 'intensity'])
                    df['intensity'] = new_intensity
                    # Añadir las intensidades a los nuevos puntos
                    df.to_csv(output_path, index=False)
                    print(f"Procesado: {file_name}")
