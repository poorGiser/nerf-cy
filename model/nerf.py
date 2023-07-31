'''
Author: cy 2449471714@qq.com
Date: 2023-07-27 20:40:16
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-28 21:21:37
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\model\nerf.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#定义nerf的模型结构
import torch
from torch import nn
from torch.nn import functional as F

class Nerf(nn.Module):
    def __init__(self,net_depth,net_width,skip,coord_feature_num,view_feature_num):
        super().__init__()
        #定义特征提取部分
        self.net_depth = net_depth
        self.skip = skip
        self.coord_feature_num = coord_feature_num
        self.view_feature_num = view_feature_num
        #定义特征提取网络，加入了残差结构
        self.feature_extract_layers = nn.ModuleList()
        for i in range(net_depth):
            if i == 0:
                self.feature_extract_layers.append(nn.Linear(coord_feature_num,net_width))
            elif i==skip + 1:
                self.feature_extract_layers.append(nn.Linear(net_width + coord_feature_num,net_width))
            else:
                self.feature_extract_layers.append(nn.Linear(net_width,net_width))
        #体密度预测层，体密度与视角无关
        self.alpha_layer = nn.Linear(net_width,1)
        #加一层特征提取层，用来提取颜色特征
        self.feature_layer = nn.Linear(net_width,net_width)
        self.color_layer1 = nn.Linear(net_width + view_feature_num,net_width // 2)
        self.color_layer2 = nn.Linear(net_width // 2,3)
    def forward(self,x):
        #分离x得到位置特征和视角特征
        coord_feature = x[:,:self.coord_feature_num]
        view_feature = x[:,self.coord_feature_num:]
                
        h = coord_feature
        
        #经过特征提取网络
        for i in range(self.net_depth):
            h = self.feature_extract_layers[i](h)
            h = F.relu(h)
            if i == self.skip:
                h = torch.cat([coord_feature, h], -1)
        
        #预测体密度
        points_alpha = self.alpha_layer(h)
        
        #预测颜色
        feature = self.feature_layer(h)
        
        # h = F.relu(h)
        
        h = torch.concat((feature,view_feature),dim=-1)
        
        h = F.relu(self.color_layer1(h))
        
        points_rgb = self.color_layer2(h)
        
        #拼接返回
        return torch.concat((points_rgb,points_alpha),-1)

        