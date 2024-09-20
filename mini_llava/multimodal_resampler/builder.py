import torch 

from .masked_drop import MaskedDrop 

class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forwad(self, x, *args, **kwargs):
        return x 
    
    @property 
    def config(self):
        return {"mm_resampler_type": None}
    
def build_vision_resampler(model_args, delay_load=False, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type == "masked_drop":
        return MaskedDrop(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()
    else:
        raise ValueError(f"Unknown resampler type: {resampler_type}")