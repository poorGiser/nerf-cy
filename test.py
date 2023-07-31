from model.nerf import Nerf
import torch

nerf  = Nerf(8,256,4,63,27)

nerf.load_state_dict(torch.load('./weights/blender.pt'))