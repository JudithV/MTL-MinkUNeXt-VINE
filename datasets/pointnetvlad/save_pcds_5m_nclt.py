import math
import numpy as np
# import matplotlib.pyplot as plt
import sys

sys.path.append(PARAMS.dataset_folder)
from tools.read_times import DataReader
import csv
from tools.read_pcds import PCDReader
from tools.read_features import ReadFeatures
import pandas as pd
from nclt.utils.plot_tools import plot_data
import os

RUNS_FOLDER = "nclt/"
GPS_FOLS ="gps/"
FILENAME = "data.csv"
POINTCLOUD_FOLS = "/velodyne_sync/"


def write_matrix_csv(directory, tr):

    scan_times, x, y, z= tr

    columns = ['id', 'timestamp', 'x', 'y', 'z']

    with open(directory, mode='w', newline='') as archivo_csv:
        write_csv = csv.writer(archivo_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        write_csv.writerow(columns)
        j=0
        for i in range(len(scan_times)):
            if str(x[i]) == 'nan':
                continue
            matrix = [j, scan_times[i], x[i], y[i], z[i]]
            # matrix = [scan_times[i], x[i], y[i], z[i], alpha[i], beta[i], gamma[i]]
            write_csv.writerow(matrix)
            j+=1
    print(f"Successfully saved: {len(scan_times)} arrays in {directory}.")



if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders))
    print("Number of runs: " + str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    print(folders)


    ruta= '2012-08-04'
    gt_dir = '/media/arvc/Seagate Expansion Drive/NCLT/'+ruta+ '/groundtruth_' + ruta + '.csv'
    
    gt = pd.read_csv(gt_dir, header=None, low_memory=False).to_dict('records')
    timestamps_clouds = pd.Series([int(dic[0]) for dic in gt])
    
    x_gt = []
    y_gt = []
    z_gt = []

    for ind in range(len(gt)):
        x_gt.append(float(gt[ind][1]))
        y_gt.append(float(gt[ind][2]))
        z_gt.append(float(gt[ind][3]))
    

    timestamps_f = []
    x_f = []
    y_f = []
    z_f = []
    timestamps_f.append(timestamps_clouds[0])
    x_f.append(x_gt[0])
    y_f.append(y_gt[0])
    z_f.append(z_gt[0])

    for i in range(len(x_gt)):
        pos_ant = np.array([x_f[-1], y_f[-1]])
        pos_act =  np.array([x_gt[i], y_gt[i]])
        if np.linalg.norm(pos_act-pos_ant)>5:
            timestamps_f.append(timestamps_clouds[i])
            x_f.append(x_gt[i])
            y_f.append(y_gt[i])
            z_f.append(z_gt[i])

   
    # GUARADAR SOLO idx_gps
    tr = (np.array(timestamps_f), np.array(x_f), np.array(y_f), np.array(z_f))

    # directory = '/home/arvc/Miriam/Repositorios/MonteCarlo/MonteCarlo/mcl_nclt/datos/' + trial+ '.csv'
    directory = '/home/arvc/Miriam/Repositorios/MonteCarlo/MonteCarlo/mcl_umh/datos/trial.csv'
    write_matrix_csv(directory, tr)

