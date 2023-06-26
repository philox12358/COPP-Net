import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def info(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def square_distance(xyz, center_xyz):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm:
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = center_xyz.shape
    _, M, _ = xyz.shape
    dist = -2 * torch.matmul(center_xyz, xyz.permute(0, 2, 1))   # Matrix multiplication
    dist += torch.sum(center_xyz ** 2, -1).view(B, N, 1)
    dist += torch.sum(xyz ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, center_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = center_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(xyz, center_xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(neighbor, xyz, center_xyz):
    """ KNN algorithm
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(xyz, center_xyz)
    _, group_idx = torch.topk(sqrdists, neighbor, dim = -1, largest=False, sorted=False)
    return group_idx

def furthest_point_sample(xyz, npoint):
    """ FPS algorithm
    batch_size: 
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape     # B=32，N=1024，C=6
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     # 32*npoint，init
    distance = torch.ones(B, N).to(device) * 1e10                       # 32*1024，Distance matrix initialization
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)   # 24 random points, B number 0~N
    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # 0~31 increasing sequence
    for i in range(npoint):                                             # select npoint point
        centroids[:, i] = farthest                                      # B*npoint, put it in
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)        # Find the farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]                           # Find the farthest point
    return centroids    # reture index of farthest point (1-1024)

def sample_and_group(npoint, radius, neighbor, xyz, feature):
    """ SG layer
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    feature = feature.permute(0, 2, 1)                             # [32, 1024, 64]，
    B, N, C = xyz.shape        # 
    S = npoint 
    
    xyz = xyz.contiguous()   # deep copy
    # fps_idx = furthest_point_sample(xyz, npoint).long()   # USE Uniform sampling here

    noise = torch.rand(B, N, device=xyz.device)             # noise in [0, 1]     USE random sampling here
    ids_shuffle = torch.argsort(noise, dim=1)               # sorted index
    ids_keep = ids_shuffle[:, :S]                           # Select top
    fps_idx = torch.arange(N, dtype=torch.long).to(xyz.device).unsqueeze(0).repeat(B,1)
    fps_idx = torch.gather(fps_idx, dim=1, index=ids_keep)    # random select some points

    center_xyz = index_points(xyz, fps_idx)                 # index point
    center_feature = index_points(feature, fps_idx)         # index point's feature



    idx = knn_point(neighbor, xyz, center_xyz)              # 32 neighbors found
    grouped_xyz = index_points(xyz, idx)                    # [B, npoint, nsample, C] 
    grouped_xyz_norm = grouped_xyz - center_xyz.view(B, S, 1, C)
    grouped_feature = index_points(feature, idx)           
    grouped_feature_center = grouped_feature - center_feature.view(B, S, 1, -1)     # centerize group feature
    res_points = torch.cat([grouped_feature_center, center_feature.view(B, S, 1, -1).repeat(1, 1, neighbor, 1)], dim=-1)
    return center_xyz, res_points