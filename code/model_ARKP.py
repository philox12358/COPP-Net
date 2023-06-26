import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()        # ([batchsize, npoints, neighbor, feature])
        x = x.permute(0, 1, 3, 2)    # ([batchsize, npoints, feature, neighbor])
        x = x.reshape(-1, d, s)      # ([batchsize*npoints, feature, neighbor])
        batch_size, _, N = x.size()  # ([batchsize*npoints, feature, neighbor])
        x1 = F.relu(self.bn1(self.conv1(x)))    # ([batchsize*npoints, feature, neighbor])
        x2 = F.relu(self.bn2(self.conv2(x1)))    # ([batchsize*npoints, feature, neighbor])
        x3 = F.adaptive_max_pool1d(x2, 1)        # ([batchsize*npoints, feature, 1 ])
        x4 = x3.view(batch_size, -1)             # ([batchsize*npoints, feature])
        x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
        return x_res                             # ([batchsize, feature, npoints])


class ARKP_Net(nn.Module):
    def __init__(self, point_num):
        super(ARKP_Net, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(point_num/256), bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_fuse1 = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):    # 32, 6, 1024
        xyz = x[:,0:3,:].permute(0, 2, 1)                  # get xyz axis
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))                # [32, 64, 8192]
        # B, D, N
        x_skip = F.relu(self.bn2(self.conv2(x)))           # [32, 1024, 256]

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, neighbor=32, xyz=xyz, feature=x)
        feature_0 = self.gather_local_0(new_feature)       # [B, 128, 512] <= [B, 512, 32, 128]

        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, neighbor=32, xyz=new_xyz, feature=feature_0)
        feature_1 = self.gather_local_1(new_feature)       # [B, 256, 256] <= [B, 256, 32, 256]

        feature_1 = torch.cat((feature_1, x_skip), dim=1)
        res = self.conv_fuse1(feature_1)

        return res      # 32, 1


class ARKP_Double(nn.Module):
    def __init__(self, args, final_channels=1):
        super(ARKP_Double, self).__init__()

        self.Net_b = ARKP_Net(point_num=args.point_num_big)
        self.Net_s = ARKP_Net(point_num=args.point_num_small)
        self.CBR = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear_mos = nn.Linear(256, 1)        # 评价MOS
        self.linear_CORR = nn.Linear(256, 4)       # 预留CORR

    def forward(self, x_b, x_s):    # 32, 6, 1024
        batch_size = x_b.shape[0]
        x_b = self.Net_b(x_b)
        x_s = self.Net_s(x_s)
        x = torch.cat((x_b, x_s), dim=1)    # [32, 1024, 256]
        x = self.CBR(x)               # [32, 512, 256]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)    # [32, 512]

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        mos = self.linear_mos(x)
        corr = self.linear_CORR(x)

        return mos, corr      # 32, 1



class ARKP_Feature(nn.Module):
    def __init__(self, args, final_channels=1):
        super(ARKP_Feature, self).__init__()

        self.Net_b = ARKP_Net(point_num=args.point_num_big)
        self.Net_s = ARKP_Net(point_num=args.point_num_small)
        self.CBR = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear_mos = nn.Linear(256, 1)        # 评价MOS
        self.linear_CORR = nn.Linear(256, 4)       # 预留CORR

    def forward(self, x_b, x_s):    # 32, 6, 1024
        batch_size = x_b.shape[0]
        x_b = self.Net_b(x_b)
        x_s = self.Net_s(x_s)
        x = torch.cat((x_b, x_s), dim=1)    # [32, 1024, 256]
        x = self.CBR(x)               # [32, 512, 256]

        return x      # 32, 1