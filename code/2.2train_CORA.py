import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_load_mos_corr import WPC_SD
from model_ARKP import ARKP_Feature
from model_CORA import CORA
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from scipy import stats
from datetime import datetime
import time




def train(args):

    train_data = WPC_SD(args, pattern='train')
    train_loader = DataLoader(train_data, num_workers=4,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = WPC_SD(args, pattern='test')
    test_loader = DataLoader(test_data, num_workers=4,
                            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    ARKP_model = ARKP_Feature(args).to(device)
    CORA_model = CORA(args).to(device)
    # print(str(model))
    ARKP_model = nn.DataParallel(ARKP_model)
    CORA_model = nn.DataParallel(CORA_model)

    if args.use_sgd:
        print("Use SGD...")
        optimizer = optim.SGD(CORA_model.parameters(), lr=args.lr, 
                    momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(CORA_model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    mse_criterion = nn.MSELoss()
    best_test_plcc = -10000.0
    best_record = 'no info'
    ARKP_model_path = f'./checkpoints/Train_ARKP/model_ARKP.pth'
    CORA_model_path = f'{args.results_dir}/model_CORA.pth'
    ARKP_model.load_state_dict(torch.load(ARKP_model_path))
    print('\033[1;35mUse pretrained ARKP_model... \033[0m')
    if args.pre_train:
        try:
            CORA_model.load_state_dict(torch.load(CORA_model_path))
            print('\033[1;35mUse pretrained CORA_model... \033[0m')
        except:
            print(f'\033[1;33mThere\'s no pre_trained CORA_model, training from scrach.\033[0m')

    begin_time = time.time()
    for epoch in range(args.epochs):
        #?###################
        #? Train
        #?###################
        train_plcc_loss = 0.0
        train_mae_loss = 0.0
        train_total_loss = 0.0
        train_count = 0.0
        ARKP_model.eval()
        CORA_model.train()    # training turn on
        train_ply_num = int(len(train_data)/args.patch_num)
        filenum_ARKP_true = [0]*train_ply_num
        filenum_ARKP_pred = [0]*train_ply_num

        for id, (data_b, data_s, mos, filenum, patch_num, patch_corr, pre_p_mos) in tqdm(enumerate(train_loader, 0), 
                total=len(train_loader), smoothing=0.9, desc =f'train epoch: {epoch}', colour = 'blue'):
            data_b, data_s = data_b.to(device), data_s.to(device)
            mos = mos.to(torch.float64).to(device).squeeze()
            mos = (mos / args.mos_scale)*100   # scale to 0-100
            data_b = data_b.reshape(-1, args.point_num_big, 6)
            data_b = data_b.permute(0, 2, 1)
            data_b = data_b.type(torch.FloatTensor)

            data_s = data_s.reshape(-1, args.point_num_small, 6)
            data_s = data_s.permute(0, 2, 1)
            data_s = data_s.type(torch.FloatTensor)

            batch_size = args.batch_size
            optimizer.zero_grad()
            with torch.no_grad():    #! 有效减少显存占用
                feature = ARKP_model(data_b, data_s)              
            pre_corr = CORA_model(feature)                #?@@@@@@@@@@@@@@@@@@@@@@@  train forward

            true_corr = torch.tensor(patch_corr).to(device).long()
            corr_loss = torch.tensor(0.0).to(device)

            for b_i in range(batch_size):
                corr_loss += F.cross_entropy(pre_corr[b_i], true_corr[b_i], reduction='sum')
            loss = corr_loss
            loss.backward()
            optimizer.step()

            train_count += batch_size
            train_total_loss += loss.item()
    
        scheduler.step()       
        train_loss =  train_total_loss*1.0/train_count
        print(f'Train(ply) {epoch},  loss:{train_loss:.6f}')
        
        time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        with open(args.results_dir + '/train_CORA_log.txt', 'a+') as txt:
            txt.write(f'\n{time_now}        Train {epoch:3d}    {train_loss:.8f}')


        #*###################
        #* Test
        #*###################

        ARKP_model.eval()     # training turn off
        CORA_model.eval()
        test_ply_num = int(len(test_data))
        all_pre_mos = np.zeros((test_ply_num, args.patch_num))
        all_pre_corr = np.zeros((test_ply_num, args.patch_num))
        all_true_corr = np.zeros((test_ply_num, args.patch_num))
        all_weight = np.zeros((test_ply_num))
        filenum_ARKP_true = [0]*test_ply_num

        for id, (data_b, data_s, mos, filenum, patch_num, patch_corr, pre_p_mos) in tqdm(enumerate(test_loader, 0), 
                total=len(test_loader), smoothing=0.9, desc =f'test  epoch：{epoch}', colour = 'green'):

            data_b, data_s = data_b.to(device), data_s.to(device)
            true_mos_cpu = (mos).cpu().numpy()
            mos = mos.to(torch.float64).to(device).squeeze()
            mos = (mos / args.mos_scale)*100   # scale to 0-100
            data_b = data_b.reshape(-1, args.point_num_big, 6)
            data_b = data_b.permute(0, 2, 1)
            data_b = data_b.type(torch.FloatTensor)

            data_s = data_s.reshape(-1, args.point_num_small, 6)
            data_s = data_s.permute(0, 2, 1)
            data_s = data_s.type(torch.FloatTensor)

            batch_size = args.test_batch_size

            with torch.no_grad():    #! 有效减少显存占用
                feature = ARKP_model(data_b, data_s)              
            pre_corr = CORA_model(feature)        #*@@@@@@@@@@@@@@@@@@@@@@@  test forward

            pre_corr_max = pre_corr.max(dim=2)[1].cpu().numpy()         # for classfication
            pre_corr_max = pre_corr_max.astype(float)
            for i, bach_corr in enumerate(pre_corr_max):
                for j, corr in enumerate(bach_corr):
                    if corr==2.0:
                        pre_corr_max[i][j] = 2
                    elif corr==1.0:
                        pre_corr_max[i][j] = 1
                    elif corr==0.0: 
                        pre_corr_max[i][j] = 0.1
            for i, num_list in enumerate(filenum):
                file_n = num_list[0]  
                all_pre_mos[file_n] = pre_p_mos[i]
                all_pre_corr[file_n] = pre_corr_max[i]
                all_true_corr[file_n] = patch_corr[i]
                all_weight[file_n] = pre_corr_max[i].sum()
                filenum_ARKP_true[file_n] = true_mos_cpu[i][0]


        filenum_ARKP_true = torch.tensor(filenum_ARKP_true)           # list2Tensor
        filenum_ARKP_pred = all_pre_mos.mean(axis=1)

        filenum_CORA_mos = (all_pre_mos * all_pre_corr).sum(axis=1)
        filenum_CORA_mos = filenum_CORA_mos/all_weight

        ARKP_test_PLCC = stats.mstats.pearsonr(filenum_ARKP_true, filenum_ARKP_pred)[0]    # PLCC
        ARKP_test_SRCC = stats.mstats.spearmanr(filenum_ARKP_true, filenum_ARKP_pred)[0]   # SRCC
        ARKP_test_rmse = torch.sqrt(((filenum_ARKP_true - filenum_ARKP_pred)**2).mean())   # RMSE
        ARKP_test_rmse = (ARKP_test_rmse/100)*args.mos_scale    # get correct rmse
        best_ARKP_record = f'PLCC:{ARKP_test_PLCC:.6f},  SRCC:{ARKP_test_SRCC:.6f},  RMSE:{ARKP_test_rmse:.6f}'
        print(f'ARKP  test:  {best_ARKP_record}')
 
        CORA_test_PLCC = stats.mstats.pearsonr(filenum_ARKP_true, filenum_CORA_mos)[0]    # PLCC
        CORA_test_SRCC = stats.mstats.spearmanr(filenum_ARKP_true, filenum_CORA_mos)[0]   # SRCC
        CORA_test_rmse = torch.sqrt(((filenum_ARKP_true - filenum_CORA_mos)**2).mean())   # RMSE
        CORA_test_rmse = (CORA_test_rmse/100)*args.mos_scale    # get correct rmse
        best_CORA_record = f'PLCC:{CORA_test_PLCC:.6f},  SRCC:{CORA_test_SRCC:.6f},  RMSE:{CORA_test_rmse:.6f}'
        print(f'CORA test:  {best_CORA_record}')

        time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        with open(args.results_dir + '/train_CORA_log.txt', 'a+') as txt:
            txt.write(f'\n{time_now}    ARKP Test {epoch:3d}    {best_ARKP_record}')
            txt.write(f'\n{time_now}    CORA Test {epoch:3d}    {best_CORA_record}')

        if CORA_test_PLCC > best_test_plcc:             # Find the best model and save it
            best_test_plcc = CORA_test_PLCC
            torch.save(CORA_model.state_dict(), CORA_model_path)
            CORA_model_score_path = f'{args.results_dir}/{int(best_test_plcc*1000000)}__model_CORA.pth'
            torch.save(CORA_model.state_dict(), CORA_model_score_path)
            print(f'\033[1;35m{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
            print(f'@@@@@@@@@@@@@@@@@@@@@    best_test_plcc: {best_test_plcc:.6f} \033[0m')

            with open(args.results_dir + '/train_CORA_log.txt', 'a+') as txt:
                txt.write(f'  @@@ Best @@@  ')



def test(args):
    # print('Start test...')
    test_data = WPC_SD(args, pattern='test')
    test_loader = DataLoader(test_data, num_workers=4,
                            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    ARKP_model = ARKP_Feature(args).to(device)
    CORA_model = CORA(args).to(device)
    # print(str(model))
    ARKP_model = nn.DataParallel(ARKP_model)
    CORA_model = nn.DataParallel(CORA_model)

    ARKP_model_path = f'./checkpoints/Train_ARKP/model_ARKP.pth'
    CORA_model_path = f'{args.results_dir}/model_CORA.pth'

    try:
        ARKP_model.load_state_dict(torch.load(ARKP_model_path))
        CORA_model.load_state_dict(torch.load(CORA_model_path))
        print('\033[1;35mUse pretrained ARKP_model... \033[0m')
        print('\033[1;35mUse pretrained CORA_model... \033[0m')
    except:
        print(f'\033[1;33mThere\'s no pre_trained CORA_model, Please train first...\033[0m')
        return None

    ARKP_model = ARKP_model.eval()       #  training turn off
    CORA_model = CORA_model.eval()

    test_ply_num = int(len(test_data))
    all_pre_ARKP = np.zeros((test_ply_num, args.patch_num))
    all_pre_CORA = np.zeros((test_ply_num, args.patch_num))
    all_true_CORA = np.zeros((test_ply_num, args.patch_num))
    all_weight = np.zeros((test_ply_num))
    filenum_ARKP_true = [0]*test_ply_num

    for id, (data_b, data_s, mos, filenum, patch_num, patch_corr, pre_p_mos) in tqdm(enumerate(test_loader, 0), 
            total=len(test_loader), smoothing=0.9, desc =f'Just test:', colour = 'green'):

        data_b, data_s = data_b.to(device), data_s.to(device)
        true_mos_cpu = (mos).cpu().numpy()
        mos = mos.to(torch.float64).to(device).squeeze()
        mos = (mos / args.mos_scale)*100   # scale to 0-100
        data_b = data_b.reshape(-1, args.point_num_big, 6)
        data_b = data_b.permute(0, 2, 1)
        data_b = data_b.type(torch.FloatTensor)

        data_s = data_s.reshape(-1, args.point_num_small, 6)
        data_s = data_s.permute(0, 2, 1)
        data_s = data_s.type(torch.FloatTensor)

        batch_size = args.test_batch_size
        with torch.no_grad():    #! 有效减少显存占用
            feature = ARKP_model(data_b, data_s)
            pre_corr = CORA_model(feature)              #*@@@@@@@@@@@@@@@@@@@@@@@  test forward

        pre_corr_max = pre_corr.max(dim=2)[1].cpu().numpy()         # for classfication
        pre_corr_max = pre_corr_max.astype(float)
        for i, bach_corr in enumerate(pre_corr_max):
            for j, corr in enumerate(bach_corr):
                if corr==2.0:
                    pre_corr_max[i][j] = 2
                elif corr==1.0:
                    pre_corr_max[i][j] = 1
                elif corr==0.0: 
                    pre_corr_max[i][j] = 0.1
        for i, num_list in enumerate(filenum):
            file_n = num_list[0]  
            all_pre_ARKP[file_n] = pre_p_mos[i]
            all_pre_CORA[file_n] = pre_corr_max[i]
            all_true_CORA[file_n] = patch_corr[i]
            all_weight[file_n] = pre_corr_max[i].sum()
            filenum_ARKP_true[file_n] = true_mos_cpu[i][0]

    filenum_ARKP_true = torch.tensor(filenum_ARKP_true)           # list2Tensor
    filenum_ARKP_pred = all_pre_ARKP.mean(axis=1)

    filenum_CORA_mos = (all_pre_ARKP * all_pre_CORA).sum(axis=1)
    filenum_CORA_mos = filenum_CORA_mos/all_weight

    ARKP_test_PLCC = stats.mstats.pearsonr(filenum_ARKP_true, filenum_ARKP_pred)[0]    # PLCC
    ARKP_test_SRCC = stats.mstats.spearmanr(filenum_ARKP_true, filenum_ARKP_pred)[0]   # SRCC
    ARKP_test_rmse = torch.sqrt(((filenum_ARKP_true - filenum_ARKP_pred)**2).mean())   # RMSE
    ARKP_test_rmse = (ARKP_test_rmse/100)*args.mos_scale    # get correct rmse
    best_ARKP_record = f'PLCC:{ARKP_test_PLCC:.6f},  SRCC:{ARKP_test_SRCC:.6f},  RMSE:{ARKP_test_rmse:.6f}'
    print(f'mos  test:  {best_ARKP_record}')

    CORA_test_PLCC = stats.mstats.pearsonr(filenum_ARKP_true, filenum_CORA_mos)[0]    # PLCC
    CORA_test_SRCC = stats.mstats.spearmanr(filenum_ARKP_true, filenum_CORA_mos)[0]   # SRCC
    CORA_test_rmse = torch.sqrt(((filenum_ARKP_true - filenum_CORA_mos)**2).mean())   # RMSE
    CORA_test_rmse = (CORA_test_rmse/100)*args.mos_scale    # get correct rmse
    best_CORA_record = f'PLCC:{CORA_test_PLCC:.6f},  SRCC:{CORA_test_SRCC:.6f},  RMSE:{CORA_test_rmse:.6f}'
    print(f'corr test:  {best_CORA_record}')


    filenum_ARKP_pred = torch.tensor(filenum_ARKP_pred)
    filenum_CORA_mos = torch.tensor(filenum_CORA_mos)
    print(f'\033[1;35m filenum_mos_true:{filenum_ARKP_true}')
    print(f'filenum_ARKP_pred:{filenum_ARKP_pred}')
    print(f'filenum_CORA_pred:{filenum_CORA_mos} \033[0m')





if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    

    parser.add_argument('--pre_train', type=bool,  default=False, help='evaluate the model?')
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model?')

    
    parser.add_argument('--data_dir', type=str, default='../data/WPC', metavar='N', help='Where does dataset exist?')
    parser.add_argument('--patch_dir', type=str, default='patch_16', help='Where does patches exist?')
    parser.add_argument('--patch_dir_big', type=str, default='patch_16_big', help='Where to store patch?')
    parser.add_argument('--patch_dir_small', type=str, default='patch_16_small', help='Where to store patch?')
    parser.add_argument('--point_num_big', type=int, default=1024, help='num of points to use')
    parser.add_argument('--point_num_small', type=int, default=8192, help='num of points to use')

    parser.add_argument('--patch_num', type=int, default=16, metavar='patches num', help='How many patchs each PC have?')
    parser.add_argument('--mos_scale', type=int, default=100, metavar='MAX MOS', help='Maximum value of MOS label?')

    args = parser.parse_args()
    args.results_dir = './checkpoints/Train_CORA'
    print(args)
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)

    if args.cuda:
        print(f'Using GPU :{torch.cuda.current_device()} from {torch.cuda.device_count()}devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    if args.eval:
        test(args)       # evaluate begin
        # test(args)       # evaluate begin
        # test(args)       # evaluate begin
    else:
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)
        train(args)      # train begin

