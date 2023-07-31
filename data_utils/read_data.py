'''
Author: cy 2449471714@qq.com
Date: 2023-07-25 19:49:14
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-07-28 15:55:37
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\data_utils\read_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import os
import torch
import imageio
import cv2
import time

#读取训练数据和测试数据并返回
def read_data(data_dir,data_type):
    time0 = time.time()
    data = None
    if data_type == 'synthesize':
        def read_json_img(mode):#mode：train\val\test
            img_dir = os.path.join(data_dir,mode)
            with open(os.path.join(data_dir,f"transforms_{mode}.json")) as f:
                json_content = json.load(f)
                json_data = {}
                json_data['camera_angle_x'] = json_content['camera_angle_x']
                json_data['rotation'] = json_content['frames'][0]['rotation']
                transform_list = []
                img_list = []
                for frame in json_content['frames']:
                    #存变换矩阵
                    transform_list.append(torch.tensor(frame['transform_matrix']))
                    #存图片数据
                    img_path = os.path.join(img_dir,frame['file_path'].split('/')[-1] + '.png')
                    img_data = imageio.imread(img_path)
                    img_data = cv2.resize(img_data, (img_data.shape[0]//2,img_data.shape[1]//2),interpolation=cv2.INTER_AREA)
                    
                    # cv2.imshow('imshow',img_data)
                    # cv2.waitKey()
                    
                    img_data = torch.tensor(img_data)[:,:,:3]
                    
                    img_list.append(img_data)
                json_data['transform_list'] = transform_list
                json_data['img_list'] = img_list
            return json_data
        modes = ['train','val','test']
        data = {mode:read_json_img(mode) for mode in modes}
        time1 = time.time()
        data['near'] = 2
        data['far'] = 6
        print(f"读取数据耗时{(time1 - time0)/1000}s")
    return data
        
                
                
        
                