import os
import torch
import numpy as np
from torch.utils.data import Dataset



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
        txtfile = os.path.join(args.data_dir, 'train_data_list.txt')
    else:
        txtfile = os.path.join(args.data_dir, 'test_data_list.txt')

    shape_ids = [line.rstrip() for line in open(txtfile)]
    data_list = [None] * len(shape_ids)
    for i, shape_id in enumerate(shape_ids):
        data_list[i] = shape_id.split(', ')

    print(f'In \"{args.patch_dir}/{pattern}\", patch: {len(data_list)}, equal to ply: {int(len(data_list)/args.patch_num)}')
    return data_list
    
def load_data(message, args, pattern):
    npy_dir_big = f'{args.data_dir}/{args.patch_dir_big}/{message[0]}/{message[1]}'
    npy_dir_small = f'{args.data_dir}/{args.patch_dir_small}/{message[0]}/{message[1]}'
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

    return point_big, point_small, mos, filenum, patch_num


def xyzrgb_normalize(point):
    point[:,0:3] = point[:,0:3] - np.mean(point[:,0:3],axis=0)
    point[:,3:6] = point[:,3:6] - np.mean(point[:,3:6],axis=0)
    return point


class WPC_SD(Dataset):
    def __init__(self, args, pattern):
        # self.num_points = args.point_num
        self.pattern = pattern
        self.data_list = read_data_list(args, pattern)
        self.data_len = len(self.data_list)
        self.args = args
        
    def __getitem__(self, item):
        message = self.data_list[item]
        point_big, point_small, mos, filenum, patch_num= load_data(message, self.args, self.pattern)
        point_big = xyzrgb_normalize(point_big)
        point_small = xyzrgb_normalize(point_small)
        if self.pattern == 'train':
            np.random.shuffle(point_big)
            np.random.shuffle(point_small)
        return point_big, point_small, mos, filenum, patch_num

    def __len__(self):
        return self.data_len


