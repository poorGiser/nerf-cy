'''
Author: cy 2449471714@qq.com
Date: 2023-07-25 19:16:43
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-31 11:53:22
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\train_opts.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
#获取命令行参数
def get_arg():
    parse = argparse.ArgumentParser("args in nerf-cy porject!")
    parse.add_argument("--data",type=str,default='./data/nerf_synthetic/lego',help='训练数据目录')
    parse.add_argument("--data_type",type=str,default='synthesize',help='训练数据目录')
    parse.add_argument("--epoch",type=int,default=100000,help='训练步数')
    parse.add_argument('--lr',type=float,default=8e-6,help='学习率')
    parse.add_argument('--pixel_num',type=int,default=1024,help='采样像素数目')
    parse.add_argument('--sample_num',type=int,default=64,help='在每条光线上均匀采样点的数目')
    
    #position-encoding
    parse.add_argument('--coord_L',type=int,default=10,help='对坐标进行position-encoding的次数')
    parse.add_argument('--dir_L',type=int,default=4,help='对视线方向进行position-encoding的次数')
    
    parse.add_argument('--net_depth',type=int,default=8,help='特征提取网络的深度')
    parse.add_argument('--net_width',type=int,default=256,help='特征提取网络的宽度')
    parse.add_argument('--skip',type=int,default=4,help='残差结构的出现位置')
    parse.add_argument('--device',type=str,default='gpu',help='gpu')
    parse.add_argument('--weights_path',type=str,default='./weights',help='权重文件保存地址')
    parse.add_argument('--test_path',type=str,default='./test_results',help='测试结果的保存路径')
    
    #优化内容
    #1 中心区域采样
    parse.add_argument('--center_train_epoch',type=int,default=500,help='在中心区域训练的轮次')
    parse.add_argument('--center_H',type=int,default=200,help='中心区域的高度')
    parse.add_argument('--center_W',type=int,default=200,help='中心区域的宽度')
    
    #2 分层采样
    parse.add_argument('--importance_sample_num',type=int,default=128,help='精细采样点的数目,如果为0则不需要精细采样')
    return parse