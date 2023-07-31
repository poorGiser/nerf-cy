'''
Author: cy 2449471714@qq.com
Date: 2023-07-28 10:28:02
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-30 23:14:37
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\data_utils\volumn_render.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn.functional as F
def volumn_render(points,splits,sample_num,view_dirs):#将点的体密度和颜色映射为图像颜色    
    points_alpha = points[:,:,-1]
    points_rgb = points[:,:,:3]
    
    if len(splits.shape) > 1:
        dists = splits[...,1:] - splits[...,:-1]
        dists = torch.concat((dists,torch.tensor([1e10],dtype=torch.float32,device=torch.device('cuda:0')).expand(dists.shape[0],1)),dim=-1).reshape(-1,sample_num)
    else:
        # print(points.isnan().any())
        dists = splits[1:] - splits[:-1]
        dists = torch.concat((dists,torch.tensor([1e10],dtype=torch.float32,device=torch.device('cuda:0'))),dim=-1).reshape(1,sample_num)
    # dists = torch.repeat_interleave(dists,repeats=points.shape[0],dim=0)
    norm_views = torch.norm(torch.repeat_interleave(view_dirs[:,None,:],dim=1,repeats=sample_num),dim=-1)
    dists = dists * norm_views
    
    alphan = 1 - torch.exp(-F.relu(points_alpha) * dists)
    
    Tn = torch.concat((torch.ones(size=(points.shape[0],1),device=torch.device('cuda:0')),1 - alphan + 1e-10),dim=-1)
    Tn = torch.cumprod(Tn,dim=1)[:,:-1]

    weights = (Tn * alphan)[...,None]
    
    rgb_map = torch.sum((weights * points_rgb),dim=-2)
    
    return rgb_map,weights
    
    
    
    
    