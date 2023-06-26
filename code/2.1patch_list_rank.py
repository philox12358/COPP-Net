import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_load_mos import WPC_SD
from model_ARKP import ARKP_Double

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
import time
from data_load_mos import WPC_SD, read_data_list



def get_train_rank(pre_mos, data_list, true_mos, pattern):
    patch_abs_mos = torch.zeros([592, int(args.patch_num)])
    patch_abs_rank = torch.zeros([592, int(args.patch_num)])
    # get the abs value
    for i, t_mos in enumerate(true_mos):
        for j, patch_mos in enumerate(pre_mos[i]):
            patch_abs_mos[i][j] = abs(t_mos-patch_mos)
    # get the rank


    for i, patch_mos_list in enumerate(patch_abs_mos):
        patch_ranked_idx = torch.argsort(patch_mos_list)
        rank = torch.argsort(patch_ranked_idx)
        for j, r in enumerate(rank):
            if r <= 5:
                rank[j] = 2
            elif r <= 10:
                rank[j] = 1
            else:
                rank[j] = 0
        patch_abs_rank[i] = rank

    # add rank to train_list
    file_file = open(f'../data/WPC/mos_corred_{pattern}_{str(int(args.patch_num))}_patch.txt',"a+")
    file_file.truncate(0)            # 清空文件
    all_pre_corr = np.zeros(len(data_list))
    for i, message in enumerate(data_list):
        patch_folder = message[0]
        patch_name = message[1]
        mos = message[2]
        filenum = message[3]
        patch_num = message[4]
        patch_rank = int(patch_abs_rank[int(filenum)][int(patch_num)])      # 获得patch的排名
        pre_patch_mos = pre_mos[int(filenum)][int(patch_num)]
        file_temp = f'{patch_folder}, {patch_name}, {mos}, {filenum}, {patch_num}, {patch_rank}, {pre_patch_mos:.4f}\n'
        all_pre_corr[i] = patch_rank
        file_file.write(file_temp)
    file_file.close()
    all_pre_corr = all_pre_corr.reshape((len(data_list)//args.patch_num, args.patch_num))
    for i, pre_corr in enumerate(all_pre_corr):
        print(f'{i}: {pre_corr}')



# 这里采用训练好的参数，对训练集的patch进行排名
def test(args, train_or_test):

    print('start test...')

    test_data = WPC_SD(args, pattern=train_or_test)
    test_loader = DataLoader(test_data, num_workers=4,
                            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    mos_model = ARKP_Double(args).to(device)
    mos_model = nn.DataParallel(mos_model)
    model_path = f'./checkpoints/Train_ARKP/model_ARKP.pth'
    mos_model.load_state_dict(torch.load(model_path))
    mos_model = mos_model.eval()       #  training turn off

    test_ply_num = int(len(test_data)/args.patch_num)
    test_count = 0.0
    filenum_mos_true = [0]*test_ply_num
    filenum_mos_pred = [0]*test_ply_num
    show_all_mos = torch.zeros([test_ply_num, int(args.patch_num)])   # [592, 16]
    
    for id, (data_b, data_s, mos, filenum, patch_num) in tqdm(enumerate(test_loader, 0), 
        total=len(test_loader), smoothing=0.9, desc =f'Just test', colour = 'green'):
        data_b, data_s = data_b.to(device), data_s.to(device)
        mos = mos.to(torch.float64).to(device).squeeze()
        mos = (mos / args.mos_scale)*100     # scale to 0-100
        data_b = data_b.permute(0, 2, 1)
        data_b = data_b.type(torch.FloatTensor)
        data_s = data_s.permute(0, 2, 1)
        data_s = data_s.type(torch.FloatTensor)
        batch_size = data_b.size()[0]
        pre_mos, corr = mos_model(data_b, data_s)              #*@@@@@@@@@@@@@@@@@@@@@@@  evaluate
        pre_mos = pre_mos.to(torch.float64).view(batch_size)
        pre_mos_cpu = (pre_mos).detach().cpu().numpy()
        true_mos_cpu = (mos).cpu().numpy()
        # preds = logits.max(dim=1)[1]
        test_count += batch_size

        for i in range(batch_size):
            filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
            filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]


            show_all_mos[int(filenum[i])][patch_num[i]] = pre_mos_cpu[i]
                
    data_list = read_data_list(args, train_or_test)
    get_train_rank(show_all_mos, data_list, filenum_mos_true, train_or_test)    #! 生成corr

    show_all_mos = show_all_mos.permute(1,0)

    filenum_mos_true = torch.tensor(filenum_mos_true)
    filenum_mos_pred = torch.tensor(filenum_mos_pred)
    filenum_mos_pred = filenum_mos_pred / args.patch_num
    ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]   # calculate corelation
    ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_KRCC = stats.mstats.kendalltau(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_mae = torch.abs(filenum_mos_true - filenum_mos_pred).mean()
    ply_test_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())
    print(f'\033[1;35mTrain (ply) {test_ply_num},  PLCC:{ply_test_PLCC:.4f},  SRCC:{ply_test_SRCC:.4f}', end='')
    print(f',  KRCC:{ply_test_KRCC:.4f},  mae:{ply_test_mae:.4f},  rmse:{ply_test_rmse:.4f}\n')

    print(f'Time now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
    print(f'filenum_mos_true:{filenum_mos_true}')
    print(f'filenum_mos_pred:{filenum_mos_pred}\033[0m')




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    
    
    parser.add_argument('--data_dir', type=str, default='../data/WPC', metavar='N', help='Where does dataset exist?')
    parser.add_argument('--patch_dir', type=str, default='patch_16', help='Where does patches exist?')
    parser.add_argument('--patch_dir_big', type=str, default='patch_16_big', help='Where to store patch?')
    parser.add_argument('--patch_dir_small', type=str, default='patch_16_small', help='Where to store patch?')
    parser.add_argument('--point_num_big', type=int, default=1024, help='num of points to use')
    parser.add_argument('--point_num_small', type=int, default=8192, help='num of points to use')

    parser.add_argument('--patch_num', type=int, default=16, metavar='patches num', help='How many patchs each PC have?')
    parser.add_argument('--mos_scale', type=int, default=100, metavar='MAX MOS', help='Maximum value of MOS label?')

    args = parser.parse_args()


    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)

    if args.cuda:
        print(f'Using GPU :{torch.cuda.current_device()} from {torch.cuda.device_count()}devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    test(args, 'test')       # evaluate begin
    test(args, 'train')       # evaluate begin



