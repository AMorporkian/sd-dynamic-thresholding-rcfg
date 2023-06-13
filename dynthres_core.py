import enum

import torch, math

######################### DynThresh Core #########################

class DynThreshScalingStartpoint(enum.IntEnum):
    ZERO = 0
    MEAN = 1


class DynThreshVariabilityMeasure(enum.IntEnum):
    STD = 0
    AD = 1

class DynThresh:
    def __init__(self, mimic_scale, separate_feature_channels, scaling_startpoint,variability_measure,interpolate_phi,threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, power_val, experiment_mode, maxSteps):
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.maxSteps = maxSteps
        self.cfg_scale_min = cfg_scale_min
        self.mimic_scale_min = mimic_scale_min
        self.experiment_mode = experiment_mode
        self.power_val = power_val

        self.sep_feat_channels = separate_feature_channels
        self.scaling_startpoint = scaling_startpoint
        self.variability_measure = variability_measure
        self.interpolate_phi = interpolate_phi

    def interpretScale(self, scale, mode, min):
        scale -= min
        max = self.maxSteps - 1
        if mode == "Constant":
            pass
        elif mode == "Linear Down":
            scale *= 1.0 - (self.step / max)
        elif mode == "Half Cosine Down":
            scale *= math.cos((self.step / max))
        elif mode == "Cosine Down":
            scale *= math.cos((self.step / max) * 1.5707)
        elif mode == "Linear Up":
            scale *= self.step / max
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos((self.step / max))
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos((self.step / max) * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(self.step / max, self.power_val)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(self.step / max, self.power_val)
        scale += min
        return scale
    
    def rescale_cfg(cond, uncond, cond_scale):
            x_cfg = uncond + cond_scale * (cond - uncond)
            ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
            ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

            x_rescaled = x_cfg * (ro_pos / ro_cfg)
            x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

            return x_final

    def dynthresh(self, cond, uncond, cfgScale, weights):
        return rescale_cfg(cond, uncond, cfgScale, self.interpolate_phi)
