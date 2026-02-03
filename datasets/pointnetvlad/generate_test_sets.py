# Judith Vilella Cantos. Miguel Hernández University of Elche.
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS

# For training and test data splits, BLT = 13, VMD = 23
X_WIDTH = 13
Y_WIDTH = 13

# BLT
#KTIMA
P16 = [663182.204586871, 4479512.96605084]
P17 = [663141.686467221, 4479464.873340834]
P18 = [663116.3699656122, 4479498.777999125]

#RISEHOLME
P19 = [665082.1563399071, 5904963.1919269]
P20 = [665100.1582773875, 5904972.71202409]

# VMD
P26 = [405156.7520412804, 5025041.790304738]
#P27 = [405107.0985735883, 5025050.372015398]
P28 = [405154.7637688267, 5025158.401943873]

P27 = [405104.76748520625, 5025096.011784253]
P29 = [405179.49875482655, 5025142.53828079]
#P28 = [405178.6821447582, 5025143.267948912]

# KITTI
P30 = [457190.21529351856, 5428835.785329838]
P31 = [457323.8517238996, 5428045.37636426]

P_DICT = {"blt-ktima": [P16, P17, P18], "blt-riseholme": [P19, P20], "vmd": [P26, P27, P28, P29]}


def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set

def filter_first_visits(df, radius=10.0):
    coords = df[['northing', 'easting']].values
    tree = KDTree(coords)

    mask = np.ones(len(df), dtype=bool)
    visited = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if not visited[i]:
            # Encuentra todos los puntos dentro del radio
            idxs = tree.query_radius([coords[i]], r=radius)[0]
            visited[idxs] = True  # Marca como visitados
        else:
            mask[i] = False  # Este ya se ha visitado antes

    return df[mask].reset_index(drop=True)


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    database_trees = []
    test_trees = []
    ind = 0

    for folder in folders:
        #print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        i = 0
        #df_locations = filter_first_visits(df_locations, radius=2.0)
        #df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        #df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            if output_name == "vmd":
                if check_in_test_set(row['easting'], row['northing'], p):
                    df_test = df_test.append(row, ignore_index=True)
                """if i % 2 == 0:
                    df_test = df_test.append(row, ignore_index=True)"""
                """if "run2" in folder:
                    df_test = df_test.append(row, ignore_index=True)"""
            else:
                if check_in_test_set(row['northing'], row['easting'], p):
                    df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)
            i += 1
            #df_database = pd.concat([df_database, pd.DataFrame([row])], ignore_index=True)
        database_tree = KDTree(df_database[['northing', 'easting']])
        database_trees.append(database_tree)
        #if ind == (len(all_folders) - 1):
        if not df_test.empty:
            test_tree = KDTree(df_test[['northing', 'easting']])
            test_trees.append(test_tree)
        ind += 1

    test_sets = []
    database_sets = []
    ind = 0
    lidar_i = 0
    i = 0
    for folder in folders:
        database = {}
        test = {}
        i = 0
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        if output_name == 'blt' or output_name == 'vmd':
            df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.csv' 
        else:
            df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin' 
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        total_rows = len(df_locations)
        for index, row in df_locations.iterrows():
            if 'segment' in df_locations.columns:
                label = row['segment']
            else:
                label = -1
            if output_name == "vmd":
                if check_in_test_set(row['easting'], row['northing'], p):
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'label': label}
                """if i % 2 == 0:
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'label': label}"""
                """if "run2" in folder:
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'label': label}"""
            else:
                if check_in_test_set(row['northing'], row['easting'], p):
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'label': label}
       
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting'], 'label': label}
            i += 1
        database_sets.append(database)
        #if ind == (len(all_folders) - 1):
        if len(test) > 0:
            test_sets.append(test)
        ind += 1

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            """if i == j:
                continue"""
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=5) 
                # indices of the positive matches in database i of each query (key) in test set j
                #test_sets[j][key][i] = index[0].tolist()
                index_list = [int(k) for k in index[0]]  # asegura que son ints y no arrays
                intersection = list(set(index_list) & set(test_sets[j].keys()))
                test_sets[j][key][i] = intersection

    output_to_file(database_sets, base_path+"train_test_sets/"+PARAMS.protocol, "minkloc_" + output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, base_path+"train_test_sets/"+PARAMS.protocol, "minkloc_" + output_name + '_evaluation_query.pickle')


if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder
                                      
    # For BLT
    folders = []
    runs_folder = "blt/"
    
    # Process the two different scenarios within the dataset
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder + "ktima/"))) #+ sorted(os.listdir(os.path.join(base_path, runs_folder + "riseholme/")))
    for folder in all_folders:
        if "2022-07-" not in folder:
            continue
        if os.path.exists(os.path.join(base_path, runs_folder + "ktima/" + folder)):
            folders.append("ktima/"+folder)
    
    construct_query_and_database_sets(base_path, runs_folder, folders, "/robot0/lidar_3d/data/",
                                      "data.csv", P_DICT["blt-ktima"], "blt")
    """folders = []
    for folder in all_folders:
            if folder != "session0" and folder != "session1" and os.path.exists(os.path.join(base_path, runs_folder + "riseholme/" + folder)):
                folders.append("riseholme/"+folder)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/robot0/lidar/data/",
                                      "data.csv", P_DICT["blt-riseholme"], "blt")"""
    

    # For VMD (TEMPO-VINE)
    """folders = []
    runs_folder = "vmd/"
    
    # To process the two different scenarios within the dataset use the following line
    #all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder + "pergola/"))) + sorted(os.listdir(os.path.join(base_path, runs_folder + "vineyard/")))
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder + "vineyard/")))
    for folder in all_folders:
        if "run2_02_p" in folder: #run3
            continue
        if "_07_" not in folder:
            continue
        if os.path.exists(os.path.join(base_path, runs_folder + "pergola/" + folder)):
            folders.append("pergola/"+folder)
        else:
            folders.append("vineyard/"+folder)
    
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud/lidar3d_0/",
                                      "data.csv", P_DICT["vmd"], "vmd")"""
