# Judith Vilella Cantos. Miguel Hernández University of Elche.
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import csv
import numpy as np
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import utm
import tqdm
import random

from datasets.base_datasets import TrainingTuple
# Import test set boundaries
from datasets.pointnetvlad.generate_test_sets import P16, P17, P18, P19, P20, check_in_test_set

# Test set boundaries
P_K = [P16, P17, P18]
P_R = [P19, P20]

RUNS_FOLDER = "blt/"
FILENAME = "data.csv"
POINTCLOUD_TS = "robot0/lidar_3d"
POINTCLOUD_FOLS = "robot0/lidar_3d/data" #/data
GPS_FOLS = "robot0/gps0"

ZONE_KTIMA = 34
ZONE_RISEHOLME = 30
LIDAR_2D = False
USE_SEGMENT = True

def construct_query_dict(df_centroids, base_path, filename):
    queries = {}
    tree = KDTree(df_centroids[['northing', 'easting']])
    coords = df_centroids[['northing', 'easting']].values
    segments = df_centroids['segment'].values
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=5) # r=7
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=12) # r=15
    # ORIGINAL:
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.csv', f"Expected .csv file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        # Conventional positives and non-negatives division by distance
        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        """if USE_SEGMENT:
            anchor_segment = segments[anchor_ndx]
            positives = [p for p in positives if segments[p] == anchor_segment]"""
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        if USE_SEGMENT:
            anchor_segment = segments[anchor_ndx]
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, position=anchor_pos, segment=anchor_segment)
        else:
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos, segment=-1)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done", filename)
    return queries


def construct_query_dict_pnv(df_centroids, base_path, filename):
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=5)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=12)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":query, "positives":positives,"negatives":negatives}
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


def get_df_at_times(file_name, df_data, time_list, time_pcs):
    """
    Build a pandas df from exaclty the times specified
    """
    # now find odo corresponding to closest times
    data = []
    index_pcs = 0
    for timestamp in time_list:
        # find the exact timestamp in df
        ind = df_data['#timestamp [ns]'] == timestamp
        row = df_data.loc[ind]
        UTMx, UTMy, _, _ = utm.from_latlon(row['latitude'], row['longitude'])
        file_name.writerow([time_pcs[index_pcs], UTMx[0], UTMy[0]])
        #data.append([time_pcs[index_pcs], UTMx[0], UTMy[0]])
        index_pcs += 1

def gps2utm(df_gps, zone_name):
    latitude = df_gps['latitude']
    longitude = df_gps['longitude']
    lat = np.array(latitude)
    lon = np.array(longitude)

    UTMx, UTMy, _, _ = utm.from_latlon(lat, lon)

    return UTMx, UTMy

def sample_gps(deltaxy, UTMx, UTMy, s, timestamp):
    gps_times, segments = [], []
    utm_positions = []
    gpsi = gpsi1 = gps = []
    for ind in timestamp.index:
        gps = np.array([UTMx[ind], UTMy[ind]])
        current_time = timestamp[ind]
        if ind == 0:
            gps_times.append(current_time)
            utm_positions.append([UTMx[ind], UTMy[ind]])
            segments.append(s[ind])
            gpsi = gps

        gpsi1 = gps
        dxy = np.linalg.norm(gpsi1[0:2] - gpsi[0:2])
        if dxy > deltaxy:
            gps_times.append(current_time)
            utm_positions.append([UTMx[ind], UTMy[ind]])
            segments.append(s[ind])
            gpsi = gpsi1
    return np.array(gps_times), np.array(utm_positions), np.array(segments)


def get_closest_data(df_data, time_list, gps_mode='utm'):
    positions = []
    orientations = []
    corresp_time_list = []

    for timestamp in time_list:
        ind = df_data['#timestamp [ns]'].sub(timestamp).abs().idxmin()
        try:
            position = [df_data['x'][ind], df_data['y'][ind], df_data['z'][ind]]
            positions.append(position)
        except:
            pass
        try:
            latitude = df_data['latitude'][ind]
            longitude = df_data['longitude'][ind]
            altitude = df_data['altitude'][ind]

            lat = np.array(latitude)
            lon = np.array(longitude)
            alt = np.array(altitude)


            if gps_mode == 'utm':
                x, y, _, _ = utm.from_latlon(lat, lon)
                position = [x, y] #, 0]
            else:
                position = [lat, lon, alt]
            positions.append(position)
        except:
            pass

        corresp_time = df_data['#timestamp [ns]'][ind]
        corresp_time_list.append(corresp_time)
    return np.array(corresp_time_list), np.array(positions), np.array(orientations)

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

def process_dataset(base_path, RUNS_FOLDER, folder, site):
    run_path = os.path.join(base_path, RUNS_FOLDER, site, folder)
    print(run_path)
    files = os.listdir(os.path.join(base_path,run_path, POINTCLOUD_FOLS))
    scantimes_pcds = []
    for f in files:
        timestamp = int(f.split('.')[0])
        scantimes_pcds.append(timestamp)
    scantimes_pcds = list(set(scantimes_pcds))
    scan_data = pd.DataFrame({"#timestamp [ns]": scantimes_pcds})
    scan_data.to_csv(os.path.join(run_path, "scan_times.csv"), index = False)
    if not os.path.exists(run_path):
        return False, None  
    with open(os.path.join(run_path, FILENAME), 'w', newline='') as file:
        escritor_csv = csv.writer(file)
        escritor_csv.writerow(['timestamp','northing','easting', 'segment'])
        scan_data = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, site, folder, "scan_times.csv"), sep=',')
        gps_data = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, site, folder, GPS_FOLS, FILENAME), sep=',')
        UTMx, UTMy = gps2utm(gps_data, site)
        gps_times = gps_data['#timestamp [ns]']
        ref_times, _, segment_id = sample_gps(deltaxy=1.0,UTMx=UTMx, UTMy=UTMy, s=gps_data['segment'], timestamp=gps_times)
        scan_times, _, _ = get_closest_data(scan_data, ref_times) # scan_data, ref_times
        #print(scan_data)
        _, utm_pos, _ = get_closest_data(gps_data, scan_times, gps_mode='utm') #gps_data, scan_times
        ind = 0
        """print("\n=== DEPURACIÓN DE SCAN TIMES ===")
        print(f"Total ficheros en carpeta: {len(files)}")
        print(f"Total scan_times devueltos: {len(scan_times)}")

        # 1. Mira algunos valores de scan_times
        print("Ejemplo scan_times:", scan_times[:20])

        # 2. Mira algunos nombres reales de ficheros
        print("Ejemplo files:", files[:20])

        # 3. Comprueba cuántos scan_times están realmente en los nombres de ficheros
        scan_times_str = [str(scan) + ".csv" for scan in scan_times]
        coinciden = [f for f in scan_times_str if f in files]
        print(f"Coincidencias exactas: {len(coinciden)} / {len(scan_times)}")

        # 4. Revisa duplicados en scan_times
        print(f"Duplicados en scan_times: {len(scan_times) - len(set(scan_times))}")"""

        for ts in scan_times: # ts in scan_times
            escritor_csv.writerow([ts, utm_pos[ind][0], utm_pos[ind][1], segment_id[ind]])
            ind += 1

    return True, scan_data['#timestamp [ns]'] # True scan_times

def process_locations(base_path, RUNS_FOLDER, folder, site, P, df_train, df_test):
    run_path = os.path.join(RUNS_FOLDER, site, folder)
    if not os.path.exists(run_path):
        return df_train, df_test 
    
    df_locations = pd.read_csv(
        os.path.join(base_path, RUNS_FOLDER, site, folder, FILENAME),
        sep=','
    )

    df_locations['timestamp'] = (
        os.path.join(RUNS_FOLDER, site, folder, POINTCLOUD_FOLS) + "/" +
        df_locations['timestamp'].astype(str) + '.csv'
    )
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    # train/test
    for _, row in df_locations.iterrows():
        if check_in_test_set(row['northing'], row['easting'], P):
            df_test = pd.concat([df_test, pd.DataFrame([row])], ignore_index=True)
        else:
            df_train = pd.concat([df_train, pd.DataFrame([row])], ignore_index=True)

    return df_train, df_test

if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER + "ktima/"))) #+ sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER + "riseholme/")))

    folders = []

    # All runs are used for training (both full and partial)
    index_list = len(all_folders)
    print("Number of runs: " + str(index_list))
    for index in range(index_list):
        if all_folders[index] != "session0" and all_folders[index] != "session1": #Both routes lack GPS information
            folders.append(all_folders[index])
    print(folders)

    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    for folder in tqdm.tqdm(folders):
        process, scan_times = process_dataset(base_path, RUNS_FOLDER, folder, "ktima")
        run_path = os.path.join(base_path, RUNS_FOLDER, "ktima", folder) + "/"
        files = os.listdir(os.path.join(base_path,RUNS_FOLDER, "ktima", folder, POINTCLOUD_FOLS))
        print(files[:5])
        if not process:
            run_path = os.path.join(base_path, RUNS_FOLDER, "riseholme", folder) + "/"
            _, scan_times = process_dataset(base_path, RUNS_FOLDER, folder, "riseholme")
            files = os.listdir(os.path.join(base_path,RUNS_FOLDER, "riseholme", folder, POINTCLOUD_FOLS))
        used_scans = [str(scan).strip() + ".csv" for scan in scan_times]
        unused_scans = list(set(files) - set(used_scans))
        print(f"Len used: {len(used_scans)}")
        print(f"Len unused: {len(unused_scans)}")
        print(f"Total files: {len(files)}")

        """for f in unused_scans:
            try:
                os.remove(run_path + POINTCLOUD_FOLS + f)
            except Exception as e:
                print(f"Error deleting {f}: {e}")"""
    
    for folder in tqdm.tqdm(folders):
        if os.path.exists(RUNS_FOLDER + "ktima/" + folder):
            df_train, df_test = process_locations(base_path, RUNS_FOLDER, folder, "ktima", P_K, df_train, df_test)
        elif os.path.exists(RUNS_FOLDER + "riseholme/" + folder):
            df_train, df_test = process_locations(base_path, RUNS_FOLDER, folder, "riseholme", P_R, df_train, df_test)

    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path+"train_test_sets/blt", "training_queries_blt_Ktima_3D_segment.pickle") # ind_nn_r=5
    construct_query_dict(df_test, base_path+"train_test_sets/blt", "test_queries_blt_Ktima_3D_segment.pickle") # ind_nn_r=5

    #construct_query_dict_pnv(df_train, base_path+"train_test_sets/blt", "training_queries_blt_Ktima_3D_PNV_autumn.pickle") # ind_nn_r=5
    #construct_query_dict_pnv(df_test, base_path+"train_test_sets/blt", "test_queries_blt_Ktima_3D_PNV_autumn.pickle")
