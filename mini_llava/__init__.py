from llava_llama import LlavaLlamaForCausalLM
from dataprocess import generate_text, LazyProcessor, LazySupervisedDataset, prepare_docci_data, DataCollatorForSupervisedDataset
from config import DataArguments
from trainer import train_mini_llava

# Default data_args used for testing
data_args = DataArguments(
    data_path = "data/mock.json",
    image_folder = "data/",
    video_folder = "data/",
    video_fps = 1,
    frames_upbound = 0,
    add_time_instruction = False,
    force_sample = False,
    default_fps = 10
)

# Default generation Config
generation_config = {
    "max_new_tokens": 100,  # Adjust as needed
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.9,
}