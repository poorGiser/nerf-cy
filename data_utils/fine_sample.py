import torch
def fine_sample(weights,mids,n_importance):#weights:[bs,62]
    rays_num = weights.shape[0]
    # sample_num = weights.shape[1]
    
    mids = mids.expand(rays_num,-1)
    weights = weights + 1e-5  # prevent nans
    weights = weights / torch.sum(weights,dim=-1,keepdim=True)
    cdf = torch.cumsum(weights,dim=-1)
    cdf = torch.concat((torch.tensor(0.0,dtype=torch.float32,).cuda().expand(rays_num,1),weights),dim=-1)#(rays_num,63)
    
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
    
    