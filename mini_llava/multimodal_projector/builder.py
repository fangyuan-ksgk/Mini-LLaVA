import torch 
import torch.nn as nn 
import re 

from .pooler_projector import PoolerProjector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *arg, **kwargs):
        return x 
    
    @property 
    def config(self):
        return {"mm_projector_type": "identity"}
    

# # This one is not required for MVP ver. implementation
# class SimpleResBlock(nn.Module): # This residual connection (with an extra projection) is another component in the multimodal projector (why not put them together?)
#     def __init__(self, channels):
#         super().__init__()
#         self.pre_norm = nn.LayerNorm(channels)
        
#         self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

#     def forward(self, x):
#         x = self.pre_norm(x)
#         return x + self.proj(x)
    

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear") # getattr works for dataclass

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    if projector_type == "identity":
        return IdentityMap()
    
    raise ValueError(f"Unknown projector type: {projector_type}")