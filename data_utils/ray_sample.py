'''
Author: cy 2449471714@qq.com
Date: 2023-07-27 16:44:14
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-30 22:02:28
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\data_utils\ray_sample.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
#TODO:减少参数
def ray_sample(rays_dir,rays_o,sample_num,near,far,split_point):
    if len(list(split_point.shape)) < 2:
        points = torch.stack([rays_o + split * rays_dir for split in split_point],dim=-2)
    else:
        points = rays_o[:,None,:] + split_point[:,:,None] + rays_dir[:,None,:]
    #验证
    # print(points.shape)
    # print(torch.sqrt(torch.sum((points[:,sample_num - 1,:sw23 - points[:,0,:]) ** 2,dim=-1)))
    return points