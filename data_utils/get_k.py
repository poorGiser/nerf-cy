#返回内参矩阵
import torch
import math
def get_k(W,camera_angle,*center): 
    """_summary_

    Args:
        H (int): 成像图像高度
        camera_angle (float): 相机视角
        center (tuple): _description_
    Returns:
        torch.tensor: _description_
    """    
    focal = (W/2)/math.tan(camera_angle/2)
    return torch.tensor([
                        [focal,0.0,center[0]],
                         [0.0,focal,center[1]],
                         [0.0,0.0,1.0]
                         ],dtype=torch.float32)
    