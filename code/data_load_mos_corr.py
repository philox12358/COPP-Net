import os
import torch
import numpy as np
from torch.utils.data import Dataset
import math


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):                
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud[:,0:3], xyz1), xyz2).astype('float32')      
    x = np.concatenate((translated_pointcloud, pointcloud[:,3:]), axis=1)
    return x

def knearest(point, center, k):
    res = np.zeros((k,))                           # init the index
    xyz = point[:,:3]                              #  
    dist = np.sum((xyz - center) ** 2, -1)         # calcu distance
    order = [(dist[i],i) for i in range(len(dist))]      
    order = sorted(order)
    for j in range(k):
        res[j] = order[j][1]
    point = point[res.astype(np.int32)]            # get k nearest point
    return point     

def read_data_list(args, pattern):
    if pattern == 'train':
        txtfile = os.path.join(args.data_dir, 'mos_corred_train_16_patch.txt')
    else:
        txtfile = os.path.join(args.data_dir, 'mos_corred_test_16_patch.txt')

    shape_ids = [line.rstrip() for line in open(txtfile)]
    data_list = [None] * len(shape_ids)
    for i, shape_id in enumerate(shape_ids):
        data_list[i] = shape_id.split(', ')

    data_list = np.array(data_list).reshape((len(shape_ids)//args.patch_num, args.patch_num, -1))

    # print(f'In \"{args.patch_dir}/{pattern}\", patch: {len(data_list)}, equal to ply: {int(len(data_list)/args.patch_num)}')
    # disorder = np.random.permutation(data_list.shape[1])
    # print(f'patch顺序已打乱：{disorder}')
    # data_list = data_list[:,disorder,:]
    return data_list
    
def load_data(message, args, pattern):
    npy_dir_big = f'{args.data_dir}/{args.patch_dir_big}/{message[0]}/{message[1]}'
    npy_dir_small = f'{args.data_dir}/{args.patch_dir_small}/{message[0]}/{message[1]}'          #! big or small ?
    point_set_big = np.load(npy_dir_big)
    point_set_small = np.load(npy_dir_small)
    point_set_big = point_set_big[:,0:6]             # @@@@@@@@@@@@ Limit data dimension 6
    point_set_small = point_set_small[:,0:6]
    index_b = np.arange(point_set_big.shape[0])
    index_s = np.arange(point_set_small.shape[0])
    if pattern == 'train':
        index_big = np.random.choice(index_b, args.point_num_big, replace=False)
        index_small = np.random.choice(index_s, args.point_num_small, replace=False)
    elif pattern == 'test':
        np.random.seed(0)
        index_big = np.random.choice(index_b, args.point_num_big, replace=False)
        np.random.seed(0)
        index_small = np.random.choice(index_s, args.point_num_small, replace=False)
    point_big = point_set_big[index_big]
    point_small = point_set_small[index_small]
    mos = torch.tensor(float(message[2])).float()
    filenum = torch.tensor(int(message[3]))
    patch_num = torch.tensor(int(message[4]))
    patch_corr = torch.tensor(int(message[5]))
    pre_p_mos = torch.tensor(float(message[6]))

    return point_big, point_small, mos, filenum, patch_num, patch_corr, pre_p_mos






def xyzrgb_normalize(point):
    point[:,0:3] = point[:,0:3] - np.mean(point[:,0:3],axis=0)
    point[:,3:6] = point[:,3:6] - np.mean(point[:,3:6],axis=0)   #! 去掉颜色中心化
    return point


def get_normal_vector(point_3):
    vec_1 = point_3[0] - point_3[1]
    vec_2 = point_3[0] - point_3[2]
    # print(p1, p2)
    normal = np.cross(vec_1, vec_2)
    scale = np.sqrt(np.square(normal[0]) + np.square(normal[1]) + np.square(normal[2]))
    normal = normal / scale   # 单位球坐标
    return normal


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
    # farthest = np.random.randint(0, N)     # 这是随机
    farthest = np.argmin(xyz, 0)[0]          # 这是找最小x的坐标
    for i in range(npoint):             
        centroids[i] = farthest       
        centroid = xyz[farthest, :]     
        dist = np.sum((xyz - centroid) ** 2, -1)   
        mask = dist < distance                     
        distance[mask] = dist[mask]                
        farthest = np.argmax(distance, -1)         
    point = point[centroids.astype(np.int32)]      
    return point  

def matrix(angle, coo):
    matrix_x = np.array([[1,                0,                0],
                         [0,  math.cos(angle), -math.sin(angle)],
                         [0,  math.sin(angle),  math.cos(angle)]])

    matrix_y = np.array([[ math.cos(angle),   0,   math.sin(angle)],
                         [               0,   1,                 0],
                         [-math.sin(angle),   0,   math.cos(angle)]])

    matrix_z = np.array([[math.cos(angle), -math.sin(angle),  0],
                         [math.sin(angle),  math.cos(angle),  0],
                         [              0,                0,  1]])
    matrix_one = {'x':matrix_x, 'y':matrix_y, 'z':matrix_z}
    return matrix_one[coo]

def rotation_point(points):
    plane_point = farthest_point_sample(points, 3)
    # plane_point = plane_point[1:]                   # 3个点
    normal_vec = get_normal_vector(plane_point)     # 得到法向量

    target_vector = np.array([0,0,1])

    angle_rotation_x = normal_vec.copy()
    angle_rotation_x[0] = 0.0
    dot_product = np.sum(angle_rotation_x * target_vector)
    die_length = np.sqrt(np.sum(np.square(angle_rotation_x)))
    theta_x = np.arccos(round(dot_product, 6) / round(die_length, 6))
    theta_x = np.abs(theta_x) if angle_rotation_x[1]<0 else -np.abs(theta_x)

    angle_rotation_y = normal_vec.copy()
    angle_rotation_y[1] = 0.0
    dot_product_y = np.sum(angle_rotation_y * target_vector)
    die_length_y = np.sqrt(np.sum(np.square(angle_rotation_y)))
    theta_y = np.arccos(round(dot_product_y, 6) / round(die_length_y, 6))
    theta_y = np.abs(theta_y) if angle_rotation_y[0]>0 else -np.abs(theta_y)

    points = np.dot(points, matrix(theta_x, 'x'))      # x轴旋转
    points = np.dot(points, matrix(theta_y, 'y'))      # y轴旋转
    return points



class WPC_SD(Dataset):
    def __init__(self, args, pattern):      
        # self.num_points = args.point_num
        self.pattern = pattern
        self.data_list = read_data_list(args, pattern)
        self.data_len = len(self.data_list)
        self.args = args
        
    def __getitem__(self, item):
        message = self.data_list[item]
        np.random.shuffle(message)   #! 对patch顺序进行随机化
        # point_big, mos, filenum, patch_num, patch_corr= load_data(message, self.args)


        point_big, point_small = [], []
        mos, filenum, patch_num, patch_corr, pre_p_mos = [], [], [], [], []
        for msg in message:
            p_b, p_s, m, fn, pn, pcorr, pre_mos = load_data(msg, self.args, self.pattern)
            p_b = xyzrgb_normalize(p_b)
            p_s = xyzrgb_normalize(p_s)
            # p[:,0:3] = rotation_point(p[:,0:3])          #! 中心化投影
            if self.pattern == 'train':
                np.random.shuffle(p_b)
                np.random.shuffle(p_s)
            point_big.append(p_b)
            point_small.append(p_s)
            mos.append(m)
            filenum.append(fn)
            patch_num.append(pn)
            patch_corr.append(pcorr)
            pre_p_mos.append(pre_mos)
        point_big = np.array(point_big)
        point_big = torch.tensor(point_big)
        point_small = np.array(point_small)
        point_small = torch.tensor(point_small)
        mos = torch.tensor(mos)
        filenum = torch.tensor(filenum)
        patch_num = torch.tensor(patch_num)
        patch_corr = torch.tensor(patch_corr)
        pre_p_mos = torch.tensor(pre_p_mos)
        return point_big, point_small, mos, filenum, patch_num, patch_corr, pre_p_mos

    def __len__(self):
        return self.data_len
