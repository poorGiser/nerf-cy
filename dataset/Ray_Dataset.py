import torch
from torch.utils.data import Dataset

class Ray_Dataset(Dataset):
    def __init__(self,x,y,views,os,dirs):
        self.train_x = x
        self.train_y = y
        self.views = views
        self.os = os
        self.dirs = dirs
    def __getitem__(self,index):
        return self.train_x[index,:,:],self.train_y[index,:],self.views[index,:],self.os[index,:],self.dirs[index,:]
    def __len__(self):
        return self.train_x.shape[0]
    
