'''
Author: cy 2449471714@qq.com
Date: 2023-07-27 11:23:53
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-28 17:43:48
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\data_utils\get_rays.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
#获取采样像素点对应的光线起点和方向
def get_rays(select_indexs,K,transform_mat,H,W):
    i = torch.arange(W).reshape(1,-1).repeat_interleave(H,dim=0)#像素的u坐标，即列
    j = torch.arange(H).reshape(-1,1).repeat_interleave(W,dim=1)#像素的v坐标，即行
    uv_mat = torch.stack((i,j,torch.ones_like(i)),dim=-1)
    uv_mat_flat = uv_mat.reshape(H * W,3)
    select_uv = uv_mat_flat[select_indexs,:]#挑选出被选择出的像素点位置
    u0 = K[0,2]
    v0 = K[1,2]
    focal = K[0,0]
    #1 将uv坐标转换到camera坐标系,camera坐标系->(y轴反转，z轴反转)->ndc坐标系(pixel_num * 3)
    select_uv = torch.stack(((select_uv[...,0] - u0) / focal,-(select_uv[...,1] - v0) / focal,-(select_uv[...,2])),dim=-1)
    #2 将ndc坐标转换到世界坐标系
    select_cartesian = transform_mat[:3,:3] @ select_uv.T
    select_cartesian = select_cartesian.T
    select_dirs = select_cartesian / torch.norm(select_cartesian,p=2,dim=-1,keepdim=True).float()
    
    #获取原点
    select_o = torch.repeat_interleave(transform_mat[:3,-1].reshape(1,-1),dim=0,repeats=select_cartesian.shape[0])
    # select_o = select_o / torch.norm(select_o,p=2,dim=1,keepdim=True)
    return select_cartesian,select_o,select_dirs
    
    