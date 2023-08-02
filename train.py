'''
Author: cy 2449471714@qq.com
Date: 2023-07-25 19:37:01
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-08-01 23:02:32
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from train_opts import get_arg
from data_utils.read_data import read_data
from data_utils.get_k import get_k
from data_utils.get_rays import get_rays
from data_utils.ray_sample import ray_sample
from data_utils.position_encoding import position_encoding
from model.nerf import Nerf
import torch
import torch.nn as nn
from tqdm import trange
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset.Ray_Dataset import Ray_Dataset
from data_utils.volumn_render import volumn_render
import torch.nn.functional as F
from model.nerf_model import NeRF
from loss.mseloss import mesloss
# torch.autograd.set_detect_anomaly(True)
import os
import imageio

from data_utils.fine_sample import fine_sample

from itertools import chain

#获取命令行参数
arguments = get_arg().parse_args()

data_dir = arguments.data#数据目录
data_type = arguments.data_type#数据目录
# device = torch.device('cuda:0') if arguments.device == 'gpu' else torch.device('cpu')
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#1 读取数据:返回字典
data = read_data(data_dir,data_type)
train_num = len(data['train']['img_list'])
test_num = len(data['test']['img_list'])
print(f"训练数据样本数：{train_num}")
print(f"验证数据样本数：{len(data['val']['img_list'])}")
print(f"测试数据样本数：{test_num}")

#boundingbox的near和far：便于采样
near = data['near']
far = data['far']

#2 计算相机内参矩阵(像素坐标系->相机坐标系)
H,W = data['train']['img_list'][0].shape[0],data['train']['img_list'][0].shape[1]
K = get_k(W,data['train']['camera_angle_x'],*(W/2,H/2))

#3 构建网络
coord_L = arguments.coord_L#position-encoding
dir_L = arguments.dir_L#position-encoding
net_depth = arguments.net_depth
net_width = arguments.net_width
skip = arguments.skip
nerf = Nerf(net_depth,net_width,skip,3 * coord_L * 2 + 3,3 * dir_L * 2 + 3)
# nerf = NeRF(net_depth,net_width,input_ch=3 * coord_L * 2 + 3,input_ch_views=3 * dir_L * 2 + 3)
nerf.to(device)

#4 训练流程
epoch = arguments.epoch#训练步数
lr = arguments.lr#学习率
pixel_num = arguments.pixel_num#采样光线数目
sample_num = arguments.sample_num#在每条光线上采样点的数目

# optimer = torch.optim.SGD(params=nerf.parameters(),lr=lr,weight_decay=1e-4)
optimer = torch.optim.Adam(params=nerf.parameters(), lr=lr, betas=(0.9, 0.999))

#定义损失函数
mse_loss = nn.MSELoss()


indexs = np.arange(H * W)
#判断是否需要中心区域采样
center_train_epoch = arguments.center_train_epoch
if center_train_epoch > 0:#如果需要
    center_indexs = []
    center_H = arguments.center_H
    center_W = arguments.center_W

    start_x = int(W // 2 - center_W // 2)
    start_y = int(H // 2 - center_H // 2)
    
    for i in range(start_y,start_y + center_H):
        for j in range(start_x,start_x + center_W):
            center_indexs.append(i * W + j)
    center_indexs = np.asarray(center_indexs)
            
#是否需要精细模型和精细采样
importance_sample_num = arguments.importance_sample_num
if importance_sample_num > 0:
    nerf_fine = Nerf(net_depth,net_width,skip,3 * coord_L * 2 + 3,3 * dir_L * 2 + 3)#创建精细网络
    nerf_fine.to(device)
    optimer = torch.optim.Adam(params=list(nerf.parameters())+list(nerf_fine.parameters()), lr=lr, betas=(0.9, 0.999))

#TODO:恢复epoch
epoch = 1000
for e in range(epoch):
    #随机选择一张图片
    img_index = torch.randint(0,train_num,size=(1,)).item()
    img = data['train']['img_list'][img_index]
    # 显示图片
    # cv2.imshow('imshow',img.numpy())
    # cv2.waitKey(0)
    
    if center_train_epoch > 0 and e < center_train_epoch:
        # print("中心区域采样")
        all_indexs = center_indexs
    else:
        all_indexs = indexs
    
    select_pixel_index = torch.tensor(np.random.choice(all_indexs,size=(pixel_num,),replace=False)).long()#随机选择pixel_num个像素点
    img_flat = img.reshape(H * W,3)#将图片展平
    target_img = img_flat[select_pixel_index,...].float()#要预测的像素点
    
    #获得每条光线的原点(摄像机位置)和方向向量
    select_cartesian,select_o,select_dirs = get_rays(select_pixel_index,K,data['train']['transform_list'][img_index],H,W)
    select_cartesian,select_o,select_dirs = select_cartesian.to(device),select_o.to(device),select_dirs.to(device)
    #沿着每条光线获得采样点
    #计算间隔点
    split_point = torch.linspace(near,far,sample_num).to(device)
    points = ray_sample(select_cartesian,select_o,sample_num,near,far,split_point)
    
    #每个采样点的方向向量
    point_dir = torch.repeat_interleave(select_dirs[:,None,:],dim=1,repeats=sample_num)
    # print(point_dir[0,:,:])
    
    #对采样点的坐标和方向进行position-encoding
    pos_enc_points = position_encoding(coord_L,points)
    pos_enc_point_dirs = position_encoding(dir_L,point_dir)
    
    coord_feature_num = pos_enc_points.shape[2]
    view_feature_num = pos_enc_point_dirs.shape[2]
    
    # print('坐标点坐标position-encoding后的维度:',coord_feature_num)
    # print('视线方向position-encoding后的维度:',view_feature_num)
    
    #concat形成训练数据
    train_points = torch.concat([pos_enc_points,pos_enc_point_dirs],dim=-1)
    
    #构建数据迭代器
    batch_size = 32
    train_dl = DataLoader(Ray_Dataset(train_points,target_img,select_cartesian,select_o,select_dirs),batch_size=batch_size,shuffle=True)
    
    for i,(X,Y,View,o,Dir) in enumerate(train_dl):
        X = X.to(device)
        Y = Y.to(device)
        
        predict_Y = nerf(X.reshape(-1,train_points.shape[-1]))
        predict_Y = predict_Y.reshape(batch_size,sample_num,-1)
        
        rgb_map,weights = volumn_render(predict_Y,split_point,sample_num,View)
        
        if importance_sample_num > 0:
            fine_zs = fine_sample(weights.squeeze()[:,1:-1],0.5*(split_point[1:] + split_point[:-1]),importance_sample_num)
            # print(fine_zs.isnan().any())
            fine_zs = fine_zs.detach()
            z_vals, _ = torch.sort(torch.cat([split_point.expand(batch_size,-1), fine_zs], -1), -1)
            fine_points = ray_sample(View,o,sample_num + importance_sample_num,near,far,z_vals)
                        
            pos_enc_fine_points = position_encoding(coord_L,fine_points)
            Dir = Dir[:,None,:].expand(-1,sample_num + importance_sample_num,-1)
            pos_enc__fine_point_dirs = position_encoding(dir_L,Dir)
            
            fine_X = torch.concat([pos_enc_fine_points,pos_enc__fine_point_dirs],dim=-1)
            
            predict_fine_Y = nerf_fine(fine_X.reshape(-1,train_points.shape[-1]))
            predict_fine_Y = predict_fine_Y.reshape(batch_size,sample_num + importance_sample_num,-1)
            
            rgb_map_fine,weights2 = volumn_render(predict_fine_Y,z_vals,sample_num + importance_sample_num,View)
            
            rgb_map_fine = torch.sigmoid(rgb_map_fine)
        #将rgb颜色值转换到0-1
        rgb_map = torch.sigmoid(rgb_map)
        
        # 计算损失函数
        Y = Y / 255.0
        loss = mse_loss(rgb_map,Y)      
        
        # print(rgb_map_fine.isnan().any())
        
        if importance_sample_num > 0:
            loss += mse_loss(rgb_map_fine,Y)      
            
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        

    print(f"train loss:epoch {e}:{loss.item()}")
    # 学习率衰减
    decay_rate = 0.1
    decay_steps = epoch
    new_lrate = lr * (decay_rate ** (e / decay_steps))
    # print(new_lrate)
    for param_group in optimer.param_groups:
        param_group['lr'] = new_lrate
    
#保存模型
weights_path = arguments.weights_path

if not os.path.exists(weights_path):
    os.mkdir(weights_path)

torch.save(nerf.state_dict(),weights_path + '/blender.pt')

#进行测试
weights_path = arguments.weights_path
test_path = arguments.test_path

if not os.path.exists(test_path):
    os.mkdir(test_path)  

#转换rgb图片
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
with torch.no_grad(): 
    for j in range(10):
        c2w = data['test']["transform_list"][j]
        indexs = torch.arange(H * W)
        #每次取1000条光线
        outputs = []
        chunk = 1000  
        for i in range(0,H * W,chunk):
            select_pixel_index = indexs[i:i + chunk]
            select_cartesian,select_o,select_dirs = get_rays(select_pixel_index,K,c2w,H,W)
            select_cartesian,select_o,select_dirs = select_cartesian.to(device),select_o.to(device),select_dirs.to(device)
            
            points = ray_sample(select_cartesian,select_o,sample_num,near,far,split_point)
            point_dir = torch.repeat_interleave(select_dirs[:,None,:],dim=1,repeats=sample_num)
            
            pos_enc_points = position_encoding(coord_L,points)
            pos_enc_point_dirs = position_encoding(dir_L,point_dir)
            
            train_points = torch.concat([pos_enc_points,pos_enc_point_dirs],dim=-1)
            predict_Y = nerf(train_points.reshape(-1,train_points.shape[-1]))
            predict_Y = predict_Y.reshape(chunk,sample_num,-1)
            
            rgb_map,weights = volumn_render(predict_Y,split_point,sample_num,select_cartesian)
            outputs.append(rgb_map.cpu())
        outputs = torch.concat(outputs,dim=0).reshape(H,W,3).numpy()
        rgb8 = to8b(outputs)
        filename = os.path.join(test_path, f'{j}.png'.format(i))
        imageio.imwrite(filename, rgb8)
        