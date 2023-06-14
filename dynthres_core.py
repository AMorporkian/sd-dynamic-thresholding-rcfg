import torch
class RescaleCFG:
    def __init__(self, phi):
        self.phi = phi

    def dynthresh(self, cond, uncond, cfgScale, weights):
        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)
        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        return multiplier * x_rescaled + (1.0 - self.phi) * x_cfg
        
