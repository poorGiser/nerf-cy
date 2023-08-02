'''
Author: cy 2449471714@qq.com
Date: 2023-07-30 20:50:24
LastEditors: cy 2449471714@qq.com
LastEditTime: 2023-08-01 22:30:34
FilePath: \read-nerf-pytorchd:\Code\summerLearn\nerf复现\nerf-cy\data_utils\fine_sample.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
def fine_sample(weights,mids,n_importance):#weights:[bs,62]
    rays_num = weights.shape[0]
    # sample_num = weights.shape[1]
    
    mids = mids.expand(rays_num,-1)
    weights = weights + 1e-5  # prevent nans
    weights = weights / torch.sum(weights,dim=-1,keepdim=True)
    cdf = torch.cumsum(weights,dim=-1)
    cdf = torch.concat((torch.tensor(0.0,dtype=torch.float32,).cuda().expand(rays_num,1),cdf),dim=-1)#(rays_num,63)
    
    # print(weights.isnan().any())
    
    
    u = torch.linspace(0,1,n_importance).cuda()
    u = u.expand(rays_num,n_importance)
    
    u = u.contiguous()
    
    index = torch.searchsorted(cdf,u,right=True)#(rays_num,n_importance)
    below = torch.max(torch.zeros_like(index - 1),index - 1)
    above = torch.min(torch.ones_like(index) * (cdf.shape[-1] - 1),index)
    
    #将索引转变成具体的值
    below_cdf = torch.gather(cdf,dim=-1,index=below)
    above_cdf = torch.gather(cdf,dim=-1,index=above)
    
    below_z = torch.gather(mids,dim=-1,index=below)
    above_z = torch.gather(mids,dim=-1,index=above)
    
    cdf_ = above_cdf - below_cdf
    
    cdf_ = torch.where(cdf_ < 1e-5,torch.ones_like(cdf_),cdf_)
    
    t = (u - below_cdf) / cdf_
    
    samples = below_z + t * (above_z - below_z)
    return samples
    
    