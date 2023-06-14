import enum

import torch, math

######################### DynThresh Core #########################

class DynThreshScalingStartpoint(enum.IntEnum):
    ZERO = 0
    MEAN = 1


class DynThreshVariabilityMeasure(enum.IntEnum):
    STD = 0
    AD = 1

def rescale_cfg(cond, uncond, cond_scale, multiplier=0.7):

    return x_final
class DynThresh:
    def __init__(self, phi):
        self.phi = phi

    def dynthresh(self, cond, uncond, cfgScale, weights):
        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)
    
        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - self.phi) * x_cfg

