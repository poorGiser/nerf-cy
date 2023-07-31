'''
Author: cy 2449471714@qq.com
Date: 2023-07-27 17:28:20
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-27 17:48:06
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\data_utils\position_encoding.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
def position_encoding(L,points):
    position_points = torch.concat([torch.sin(2 ** i * points) if j== 0 else torch.cos(2 ** i * points)  for i in range(L) for j in range(2)],dim=-1)
    #加上原始坐标
    position_points = torch.concat([points,position_points],dim=-1)
    # print(points[0,0,:])
    # print(position_points[0,0,:])
    return position_points