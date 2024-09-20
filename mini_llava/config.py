from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    mm_vision_tower: str = field(default="openai/clip-vit-base-patch32")    
    mm_vision_select_feature: str = field(default="patch")
    mm_vision_select_layer: int = field(default=-1)
    mm_hidden_size: int = field(default=768) # this one match the pre-trained vision encoder
    
    use_im_start_end: bool = field(default=False)
    use_im_patch_token: bool = field(default=True)
    
    delay_load: bool = field(default=True)
    
    mm_resampler_type: Optional[str] = field(default=None)
    
    hidden_size: int = field(default=768) # This one should match with the language model


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: int = field(default=1)
    frames_upbound: int = field(default=0)
    add_time_instruction: bool = field(default=False)
    force_sample: bool = field(default=False)
    default_fps: int = field(default=10)  # Added based on usage in LazySupervisedDataset