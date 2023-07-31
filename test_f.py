'''
Author: cy 2449471714@qq.com
Date: 2023-07-25 19:33:25
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-30 19:42:52
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import math
H,W = 800,800
import numpy as np
from model.nerf import Nerf
import torch.nn.functional as F

# # i = torch.arange(W).reshape(1,-1).repeat_interleave(H,dim=0)#像素的u坐标，即列
# # j = torch.arange(H).reshape(-1,1).repeat_interleave(W,dim=1)#像素的v坐标，即行
# # uv_mat = torch.stack((i,j,torch.ones_like(i)),dim=-1)
# # uv_mat_flat = uv_mat.reshape(H * W,3)
# # select_indexs = torch.randint(10000,size=(1000,)).long()
# # print(uv_mat_flat[select_indexs,:].shape)
# x = torch.rand(4,4)
# x.shape[0]

#测试坐标转换程序
# test_transformer = torch.tensor([
#                 [
#                     -0.9999021887779236,
#                     0.004192245192825794,
#                     -0.013345719315111637,
#                     -0.05379832163453102
#                 ],
#                 [
#                     -0.013988681137561798,
#                     -0.2996590733528137,
#                     0.95394366979599,
#                     3.845470428466797
#                 ],
#                 [
#                     -4.656612873077393e-10,
#                     0.9540371894836426,
#                     0.29968830943107605,
#                     1.2080823183059692
#                 ],
#                 [
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0
#                 ]
#             ])
# camera_angle = 0.6911112070083618
# focal = (W/2)/math.tan(camera_angle/2)

# K = torch.tensor(torch.tensor([
#                     [focal,0.0,W // 2],
#                         [0.0,focal,H // 2],
#                         [0.0,0.0,1.0]
#                         ],dtype=torch.float32))

# select_uv = torch.tensor([[0,0,1],[0,1,1]])
# u0 = K[0,2]
# v0 = K[1,2]
# focal = K[0,0]
# #1 将uv坐标转换到camera坐标系,camera坐标系->(y轴反转，z轴反转)->ndc坐标系(pixel_num * 3)
# select_uv = torch.stack(((select_uv[...,0] - u0) / focal,-(select_uv[...,1] - v0) / focal,-(select_uv[...,2])),dim=-1)
# #2 将ndc坐标转换到世界坐标系
# select_cartesian = test_transformer[:3,:3] @ select_uv.T
# select_cartesian = select_cartesian.T
# print(select_cartesian[1,:])

# i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
# i = i.t()
# j = j.t()
# dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
# rays_d = torch.sum(dirs[..., np.newaxis, :] * test_transformer[:3, :3], -1)
# print(rays_d[1,0,:])


#测试插值点
# N_samples = 64
# near = 2
# far = 6
# t_vals = torch.linspace(0., 1., steps=N_samples)
# z_vals = near * (1. - t_vals) + far * (t_vals)  # 插值采样
# split_point = torch.linspace(near,far,N_samples)
# print(z_vals)
# print(split_point)

# a = object()
# b = a
# print(id(a))
# print(id(b))


#检查网络结构
# nerf = Nerf(8,256,4,63,27)
# x = torch.rand(size=(1024,64,90))

# print(nerf(x).shape)

# a = torch.tensor([-0.5,0.3,0.8])
# print(torch.sum(F.softmax(a ** 2)))

# a = torch.tensor([-1.7643, -1.1342, -0.0679]).float()
# print(a.dtype)

# a = [1,2,3]
# print(a[-1])


# a = torch.rand((1024,64))
# a[:,0] = 0.
# a[:,-1] = 1.

# a,_ = torch.sort(a)

# b = torch.linspace(0,1,128)
# b = b.expand(1024,128)
# b = b.contiguous()
             
# test1 = torch.searchsorted(a,b,right=True)
# test2 = torch.searchsorted(a,b)


# print(test1[0,:])
# print(test2[0,:])

a = torch.tensor([0.0,0.1,0.4,0.9,1.0])
b = torch.linspace(0,1,6)
print(b)

test = torch.searchsorted(a,b,right=True)
print(test)
