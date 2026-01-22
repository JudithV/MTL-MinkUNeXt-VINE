# Miguel Hernández University of Elche
# Institute for Engineering Research of Elche (I3E)
# Automation, Robotics and Computer Vision lab (ARCV)
# Author: Judith Vilella Cantos
import pandas as pd
import numpy as np
import utm
import os
from sklearn.decomposition import PCA
from config import PARAMS

# Load ground truth
df = pd.read_csv(os.path.join(PARAMS.dataset_folder, 'vmd/vineyard/run1_04_v/gps.csv')) # blt/ktima/2022-04-06-11-02-34/robot0/gps0/data.csv
utm_coords = np.array([utm.from_latlon(lon, lat)[:2] for lat, lon in zip(df['latitude'], df['longitude'])])
x_coords, y_coords = utm_coords[:, 0], utm_coords[:, 1]

# PCA for vineyard alignment
pca = PCA(n_components=2)
coords_pca = pca.fit_transform(utm_coords)
main_axis = coords_pca[:, 0]  # eje longitudinal del viñedo (a lo largo de las filas)
perp_axis = coords_pca[:, 1]  # eje transversal (a través de las filas)

# Define extremes
main_min, main_max = np.percentile(main_axis, [2, 98])
margin = 5.0  # meters

lower_extreme = main_axis < (main_min + margin)
upper_extreme = main_axis > (main_max - margin)
intra_row = ~(lower_extreme | upper_extreme)

# Asign labels to all timestamps
segment_labels = np.zeros(len(x_coords), dtype=int)
segment_labels[lower_extreme] = 0
segment_labels[upper_extreme] = 1
segment_labels[intra_row] = 2

# Save labeled data
df['segment'], df['type'] = segment_labels, "V" # Type "V" for trellis, "P" for pergola
df.to_csv(os.path.join(PARAMS.dataset_folder, 'vmd/vineyard/run1_04_v/gps.csv'), index=False) # blt/ktima/2022-04-06-11-02-34/robot0/gps0/data.csv

