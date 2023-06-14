##################
# Stable Diffusion Dynamic Thresholding (CFG Scale Fix)
#
# Author: Alex 'mcmonkey' Goodwin
# GitHub URL: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
# Created: 2022/01/26
# Last updated: 2023/01/30
#
# For usage help, view the README.md file in the extension root, or via the GitHub page.
#
##################

import gradio as gr
import torch, traceback
import dynthres_core
from dynthres_core import DynThreshScalingStartpoint, DynThreshVariabilityMeasure
from modules import scripts, script_callbacks, sd_samplers, sd_samplers_compvis, sd_samplers_kdiffusion, sd_samplers_common
try:
    import dynthres_unipc
except Exception as e:
    print(f"\n\n======\nError! UniPC sampler support failed to load! Is your WebUI up to date?\n(Error: {e})\n======")

######################### Data values #########################
VALID_MODES = ["Constant", "Linear Down", "Cosine Down", "Half Cosine Down", "Linear Up", "Cosine Up", "Half Cosine Up", "Power Up", "Power Down"]

######################### Script class entrypoint #########################
class Script(scripts.Script):

    def title(self):
        return "Rescale CFG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enabled = gr.Checkbox(value=False, label="Enable Rescale CFG")
        interpolate_phi = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Interpolate Phi",value=0.7)
        enabled.change(
            fn=lambda x: {"visible": x, "__type__": "update"},
            inputs=[enabled],
            outputs=[accordion],
            show_progress = False)
        self.infotext_fields = (
            (enabled, lambda d: gr.Checkbox.update(value="Enable Rescale CFG" in d)),
            (interpolate_phi, "Interpolate Phi"),
        return [enabled, interpolate_phi]

    last_id = 0

    def process_batch(self, p, enabled, phi, prompts, seeds, subseeds):
        enabled = getattr(p, 'rescale_enabled', enabled)
        if not enabled:
            return
        orig_sampler_name = p.sampler_name
        phi = getattr(p, 'rescale_phi', phi)
        p.extra_generation_params["RescaleCFG Enabled"] = True
        p.extra_generation_params["Interpolate Phi"] = phi
        
        Script.last_id += 1
        fixed_sampler_name = f"{orig_sampler_name}_rescalecfg{Script.last_id}"

        sampler = sd_samplers.all_samplers_map[orig_sampler_name]
        rescaleData = dynthres_core.RescaleCFG(interpolate_phi)
        if orig_sampler_name == "UniPC":
            def uniPCConstructor(model):
                return CustomVanillaSDSampler(dynthres_unipc.CustomUniPCSampler, model, dtData)
            newSampler = sd_samplers_common.SamplerData(fixed_sampler_name, uniPCConstructor, sampler.aliases, sampler.options)
        else:
            def newConstructor(model):
                result = sampler.constructor(model)
                cfg = CustomCFGDenoiser(result.model_wrap_cfg.inner_model, dtData)
                result.model_wrap_cfg = cfg
                return result
            newSampler = sd_samplers_common.SamplerData(fixed_sampler_name, newConstructor, sampler.aliases, sampler.options)
        # Apply for usage
        p.orig_sampler_name = orig_sampler_name
        p.sampler_name = fixed_sampler_name
        p.fixed_sampler_name = fixed_sampler_name
        sd_samplers.all_samplers_map[fixed_sampler_name] = newSampler
        if p.sampler is not None:
            p.sampler = sd_samplers.create_sampler(fixed_sampler_name, p.sd_model)

    def postprocess_batch(self, p, enabled, phi, batch_number, images):
        if not enabled or not hasattr(p, 'orig_sampler_name'):
            return
        p.sampler_name = p.orig_sampler_name
        del sd_samplers.all_samplers_map[p.fixed_sampler_name]
        del p.orig_sampler_name
        del p.fixed_sampler_name

######################### CompVis Implementation logic #########################

class CustomVanillaSDSampler(sd_samplers_compvis.VanillaStableDiffusionSampler):
    def __init__(self, constructor, sd_model, dtData):
        super().__init__(constructor, sd_model)
        self.sampler.main_class = dtData

######################### K-Diffusion Implementation logic #########################

class CustomCFGDenoiser(sd_samplers_kdiffusion.CFGDenoiser):
    def __init__(self, model, dtData):
        super().__init__(model)
        self.main_class = dtData

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        # conds_list shape is (batch, cond, 2)
        weights = torch.tensor(conds_list, device=uncond.device).select(2, 1)
        weights = weights.reshape(*weights.shape, 1, 1, 1)
        self.main_class.step = self.step
        return self.main_class.dynthresh(x_out[:-uncond.shape[0]], denoised_uncond, cond_scale, weights)

######################### XYZ Plot Script Support logic #########################

def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
    def apply_field_enum(field, EnumClass):
        def fun(p, x, xs):
            enum_x = EnumClass[x]
            setattr(p, field, enum_x)

        return fun

    extra_axis_options = [
        xyz_grid.AxisOption("[RescaleCFG] Phi", float, xyz_grid.apply_field("dynthres_interpolate_phi")),
    ]
    if not any("[RescaleCFG]" in x.label for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(extra_axis_options)

def callbackBeforeUi():
    try:
        make_axis_options()
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to add support for X/Y/Z Plot Script because: {e}")

script_callbacks.on_before_ui(callbackBeforeUi)
