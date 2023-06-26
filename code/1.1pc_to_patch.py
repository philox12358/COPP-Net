import os
import os.path
import numpy as np
from plyfile import PlyData
import pandas as pd                   
import argparse
from multiprocessing import Pool, current_process
import xlrd
from sklearn.neighbors import KDTree
import open3d as o3d


def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    return dataFile



def xyz_1_2001(xyz):
  new_xyz = np.copy(xyz)
  # new_xyz[:0] = xyz[:0] - xyz[:0].min()   # Translate coordinate
  # new_xyz[:1] = xyz[:1] - xyz[:1].min()
  # new_xyz[:2] = xyz[:2] - xyz[:2].min()
  new_xyz = new_xyz - new_xyz.min()   # Set the minimum value to 0
  scale = xyz.max() - xyz.min()       # Difference between maximum and minimum
  new_xyz = new_xyz / scale           # normalize
  new_xyz = new_xyz*2000 + 1          # to 1-2001
  return new_xyz



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D], 
        npoint: number of samples (1024)
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape                  
    xyz = point[:,:3]                   
    centroids = np.zeros((npoint,))    
    distance = np.ones((N,)) * 1e10    
    farthest = np.random.randint(0, N)  
    for i in range(npoint):             
        centroids[i] = farthest       
        centroid = xyz[farthest, :]     
        dist = np.sum((xyz - centroid) ** 2, -1)   
        mask = dist < distance                     
        distance[mask] = dist[mask]                
        farthest = np.argmax(distance, -1)         
    point = point[centroids.astype(np.int32)]      
    return point                                   



def knearest(point, center, k):
    res = np.zeros((k,))                           
    xyz = point[:,:3]                              
    dist = np.sum((xyz - center) ** 2, -1)        
    order = [(dist[i],i) for i in range(len(dist))]      
    order = sorted(order)
    for j in range(k):
        res[j] = order[j][1]
    point = point[res.astype(np.int32)]            
    return point

    
def create_patch(id , path, args):
    ply_str = path.strip().split('.')[0]       # folder name
    folder_big = os.path.join(args.data_dir, args.patch_dir_big, ply_str)
    folder_small = os.path.join(args.data_dir, args.patch_dir_small, ply_str)
    if not os.path.exists(folder_big):
        os.mkdir(folder_big)
        os.mkdir(folder_small)  
    else:
        print(f'stride the {id}_th file.........')
        return

    PC_dir = os.path.join(args.data_dir, args.ply_dir, path)

    pcd = o3d.io.read_point_cloud(PC_dir)
    PC_points = np.asarray(pcd.points)                         # 拿出点
    PC_colors = np.asarray(pcd.colors)*255
    point_cloud = np.concatenate((PC_points, PC_colors), axis=1)
    
    point_cloud[:,0:3] = xyz_1_2001(point_cloud[:,0:3])        # normalize xyz to 1-2001
    points = point_cloud[:,0:3]
    kd_tree = KDTree(points)

    centers = farthest_point_sample(point_cloud, args.center_points)     # FPS

    for m, center in enumerate(centers):
        center = center[0:3]
        filename = ply_str + '__' + str(m)             
        filename_big = os.path.join(folder_big, filename)
        filename_small = os.path.join(folder_small, filename)
        if point_cloud.shape[0]<10001:
            kpoint_big = point_cloud
        else:
            # kpoint = knearest(point_cloud, center, args.k_nearest)[:,:6]   # 普通KNN，太慢了
            dist_to_knn, knn_idx_big = kd_tree.query(X=[center], k=args.knn_big)   # 0为中心点索引，k为邻居数
            dist_to_knn, knn_idx_small = kd_tree.query(X=[center], k=args.knn_small)   # 0为中心点索引，k为邻居数
            kpoint_big = point_cloud[knn_idx_big[0]][:,:6]
            kpoint_small = point_cloud[knn_idx_small[0]][:,:6]
        np.save(filename_big, kpoint_big)
        np.save(filename_small, kpoint_small)
    print(f'The {id+1}_th file completed, name:  {ply_str}.ply')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expriment setting')
    parser.add_argument('--knn_big', type=int, default=14900, help='points numbers of each patch have?')
    parser.add_argument('--knn_small', type=int, default=8192, help='points numbers of each patch have?')
    parser.add_argument('--data_dir', type=str, default='../data/WPC', help='What dataset to use?')

    parser.add_argument('--ply_dir', type=str, default='Distortion_ply', help='Where does ply file exist?')
    parser.add_argument('--patch_dir_big', type=str, default='patch_16_big', help='Where to store patch?')
    parser.add_argument('--patch_dir_small', type=str, default='patch_16_small', help='Where to store patch?')
    parser.add_argument('--center_points', type=int, default=16, help='number of patches?')

    parser.add_argument('--parallel_cpu', type=int, default=12, help='CPU Core number?')

    args = parser.parse_args()

    # Create train patch file ----------------------------------------------------------
    try:
        os.mkdir(os.path.join(args.data_dir, args.patch_dir_big))
        os.mkdir(os.path.join(args.data_dir, args.patch_dir_small))
    except:
        print(f'Patch folder exits, Please check it...')
        exit()


    # Create all patch file ----------------------------------------------------------
    exle_file = read_xlrd(os.path.join(args.data_dir, 'mos.xls'))
    print(f'Save dir: {args.data_dir}/{args.patch_dir_big} and {args.data_dir}/{args.patch_dir_small}')
    print(f'There are {len(exle_file)} files waiting for process... ')
    print(f'USE {args.parallel_cpu} CPU core to process the task in parallel.')

    pool = Pool(args.parallel_cpu)         # Create a process pool
    for id, [name, mos] in enumerate(exle_file):
        pool.apply_async(func=create_patch, args=(id, name, args,))
    pool.close()     # close process pool
    pool.join()      # waits for all the child process
    
    # create_patch(0, 'bag_tsl_8_tqs_64.ply', args)
    print(f'All files done...')


