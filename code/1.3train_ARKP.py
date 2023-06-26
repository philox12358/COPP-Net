import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_load_mos import WPC_SD
from model_ARKP import ARKP_Double

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
    mos_model = ARKP_Double(args).to(device)
    # print(str(model))
    mos_model = nn.DataParallel(mos_model)

    if args.use_sgd:
        print("Use SGD...")
        optimizer = optim.SGD(mos_model.parameters(), lr=args.lr, 
                    momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(mos_model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    mse_criterion = nn.MSELoss()
    best_test_plcc = -10000.0
    best_record = 'no info'
    mos_model_path = f'{args.results_dir}/model_ARKP.pth'
    if args.pre_train:
        try:
            mos_model.load_state_dict(torch.load(mos_model_path))
            print('\033[1;35mUSE pretrained model... \033[0m')
        except:
            print(f'There\'s no pre_trained model, training from scrach.')

    begin_time = time.time()
    for epoch in range(args.epochs):
        #?###################
        #? Train
        #?###################
        train_plcc_loss = 0.0
        train_mae_loss = 0.0
        train_total_loss = 0.0
        train_count = 0.0
        mos_model.train()    # training turn on
        train_ply_num = int(len(train_data)/args.patch_num)
        filenum_mos_true = [0]*train_ply_num
        filenum_mos_pred = [0]*train_ply_num

        for id, (data_b, data_s, mos, filenum, patch_num) in tqdm(enumerate(train_loader, 0), 
                total=len(train_loader), smoothing=0.9, desc =f'train epoch: {epoch}', colour = 'blue'):
            data_b, data_s = data_b.to(device), data_s.to(device)
            mos = mos.to(torch.float64).to(device).squeeze()
            mos = (mos / args.mos_scale)*100   # scale to 0-100
            data_b = data_b.permute(0, 2, 1)
            data_b = data_b.type(torch.FloatTensor)
            data_s = data_s.permute(0, 2, 1)
            data_s = data_s.type(torch.FloatTensor)

            batch_size = data_b.size()[0]
            optimizer.zero_grad()
            pre_mos, corr = mos_model(data_b, data_s)              #?@@@@@@@@@@@@@@@@@@@@@@@  train forward
            pre_mos = pre_mos.to(torch.float64).view(batch_size)
            pre_mos_cpu = (pre_mos).detach().cpu().numpy()
            true_mos_cpu = (mos).cpu().numpy()

            loss = mse_criterion(pre_mos, mos)

            loss.backward()
            optimizer.step()
            train_count += batch_size
            train_total_loss += loss.item()
            
            for i in range(batch_size):
                filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
                filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]
        scheduler.step()        
        filenum_mos_true = torch.tensor(filenum_mos_true)               # list2Tensor
        filenum_mos_pred = torch.tensor(filenum_mos_pred)
        filenum_mos_pred = filenum_mos_pred / args.patch_num
        ply_train_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]   # PLCC
        ply_train_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]  # SRCC
        ply_train_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())  # RMSE
        ply_train_rmse = (ply_train_rmse/100)*args.mos_scale    # get correct rmse

        record = f'Train {epoch:3d},  loss:{train_total_loss*1.0/train_count:.4f}, PLCC:{ply_train_PLCC:.4f}, SRCC:{ply_train_SRCC:.4f}, rmse:{ply_train_rmse:.4f}'
        print(record)
 
        time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        with open(args.results_dir + '/train_ARKP_log.txt', 'a+') as txt:
            txt.write(f'\n{time_now}    {record}')



        #*###################
        #* Test
        #*###################
        test_ply_num = int(len(test_data)/args.patch_num)
        test_plcc_loss = 0.0
        test_total_loss = 0.0
        test_count = 0.0
        mos_model.eval()     # training turn off
        filenum_mos_true = [0]*test_ply_num
        filenum_mos_pred = [0]*test_ply_num

        for id, (data_b, data_s, mos, filenum, patch_num) in tqdm(enumerate(test_loader, 0), 
                total=len(test_loader), smoothing=0.9, desc =f'test  epoch：{epoch}', colour = 'green'):
            data_b, data_s = data_b.to(device), data_s.to(device)
            mos = mos.to(torch.float64).to(device).squeeze()
            mos = (mos / args.mos_scale)*100   # scale to 0-100
            data_b = data_b.permute(0, 2, 1)
            data_b = data_b.type(torch.FloatTensor)
            data_s = data_s.permute(0, 2, 1)
            data_s = data_s.type(torch.FloatTensor)
            batch_size = data_b.size()[0]
            pre_mos, corr = mos_model(data_b, data_s)                  #*@@@@@@@@@@@@@@@@@@@@@@@  test forward
            pre_mos = pre_mos.to(torch.float64).view(batch_size)
            pre_mos_cpu = (pre_mos).detach().cpu().numpy()
            true_mos_cpu = (mos).cpu().numpy()

            loss = mse_criterion(pre_mos, mos)

            # preds = logits.max(dim=1)[1]            # for classfication
            test_count += batch_size
            test_total_loss += loss.item()

            for i in range(batch_size):
                filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
                filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]

        filenum_mos_true = torch.tensor(filenum_mos_true)           # list2Tensor
        filenum_mos_pred = torch.tensor(filenum_mos_pred)
        filenum_mos_pred = filenum_mos_pred / args.patch_num
        ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]    # PLCC
        ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]   # SRCC
        ply_test_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())   # RMSE
        ply_test_rmse = (ply_test_rmse/100)*args.mos_scale    # get correct rmse

        record = f'Test  {epoch:3d},  loss:{test_total_loss*1.0/test_count:.4f}, PLCC:{ply_test_PLCC:.4f}, SRCC:{ply_test_SRCC:.4f}, rmse:{ply_test_rmse:.4f}'
        print(record)

        time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        with open(args.results_dir + '/train_ARKP_log.txt', 'a+') as txt:
            txt.write(f'\n{time_now}    {record}')


        filenum_mos_true = (filenum_mos_true/100)*args.mos_scale    # rescale to reality
        filenum_mos_pred = (filenum_mos_pred/100)*args.mos_scale    # rescale to reality
        if ply_test_PLCC > best_test_plcc:             # Find the best model and save it
            best_test_plcc = ply_test_PLCC
            best_test_record = record
            torch.save(mos_model.state_dict(), mos_model_path)
            print(f'\033[1;35mTime now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}', end='')
            print(f'    {best_test_record}')
            print(f'The best model and record have been saved (epoch:{epoch})')
            print(f'filenum_mos_true:{filenum_mos_true}')            # print the mos of ture and pred
            print(f'filenum_mos_pred:{filenum_mos_pred} \033[0m')

            with open(args.results_dir + '/train_ARKP_log.txt', 'a+') as txt:
                txt.write(f'  @@@ Best @@@  ')

        if epoch==100:                               # record score epoch=100
            cost_time = time.time()-begin_time
            print(f'\033[1;35mTime now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
            print(f'Time in 100 epoch:  {cost_time/60:.4f}  minute...')
            print(f'BEST_Test_Record: {best_test_record}')
            print(f'best_test_plcc: {best_test_plcc}')
            print(f'filenum_mos_true:{filenum_mos_true}')            
            print(f'filenum_mos_pred:{filenum_mos_pred} \033[0m')
            



def test(args):
    # print('Start test...')
    test_data = WPC_SD(args, pattern='test')
    test_loader = DataLoader(test_data, num_workers=4,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    mos_model = ARKP_Double(args).to(device)
    mos_model = nn.DataParallel(mos_model)
    model_path = f'./checkpoints/Train_ARKP/model_ARKP.pth'
    if os.path.exists(model_path):
        mos_model.load_state_dict(torch.load(model_path))
        # print('\033[1;35mUSE pretrained model for testing... \033[0m')
    else:
        print(f'\nPlease prepare the model for testing first...in {model_path}')
        return None
    mos_model = mos_model.eval()       #  training turn off

    test_ply_num = int(len(test_data)/args.patch_num)
    test_count = 0.0
    filenum_mos_true = [0]*test_ply_num
    filenum_mos_pred = [0]*test_ply_num
    show_all_mos = torch.zeros([test_ply_num, int(args.patch_num)])
    for id, (data_b, data_s, mos, filenum, patch_num) in tqdm(enumerate(test_loader, 0), 
        total=len(test_loader), smoothing=0.9, desc =f'Just test', colour = 'green'):
        data_b, data_s = data_b.to(device), data_s.to(device)
        mos = mos.to(torch.float64).to(device).squeeze()
        mos = (mos / args.mos_scale)*100   # scale to 0-100
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

    show_all_mos = show_all_mos.permute(1,0)

    filenum_mos_true = torch.tensor(filenum_mos_true)
    filenum_mos_pred = torch.tensor(filenum_mos_pred)
    filenum_mos_pred = filenum_mos_pred / args.patch_num
    ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]   # calculate corelation
    ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())
    ply_test_rmse = (ply_test_rmse/100)*args.mos_scale    # get correct rmse
    
    print(f'\033[1;35mTest (ply) {test_ply_num},    PLCC:{ply_test_PLCC:.4f},  SRCC:{ply_test_SRCC:.4f}', end='')
    print(f',  RMSE:{ply_test_rmse:.4f}\n')

    # print(f'Time now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
    filenum_mos_true = (filenum_mos_true/100)*args.mos_scale    # rescale to reality
    filenum_mos_pred = (filenum_mos_pred/100)*args.mos_scale    # rescale to reality
    print(f'filenum_mos_true:{filenum_mos_true}')
    print(f'filenum_mos_pred:{filenum_mos_pred}\033[0m')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
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
    args.results_dir = './checkpoints/Train_ARKP'
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
    else:
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)
        train(args)      # train begin

