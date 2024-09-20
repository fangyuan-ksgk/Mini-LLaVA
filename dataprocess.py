import transformers 
from typing import Dict, Sequence, List
import copy, torch, json, os, av
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
from torch.utils.data import Dataset 
import numpy as np
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence


# Question: It's important to note the flow of special token <image>, IMAGE_TOKEN_INDEX, as well as the DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# -- Before the tokenizer is updated with these special tokens (IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN) we should only include place-holder IMAGE_TOKEN_INDEX input the input_ids
# -- Then, we should expand it to tokenizer.encode(DEFAULT_IM_START_TOKEN) + IMAGE_EMBEDDINGs + tokenizer.encode(DEFAULT_IM_END_TOKEN) with updated tokenizer
# -- In this data processing pipeline, we did not update the tokenizer, therefore DEFAULT_IM_START_TOKEN & DEFAULT_IM_END_TOKEN should NOT BE INCLUDED !


def preprocess_inference_inputs(
    conversations,
    tokenizer: transformers.PreTrainedTokenizer,
    default_system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    """ 
    Prepare input_ids for inference with chat-template. 
    - convert <image> to IMAGE_TOKEN_INDEX and make sure targets are correct
    """
    # One thing:
    # 1. Chat Template application with <assistant_start> token at the end 
    # I don't need to repeat <image> for videos here -- input_embeds should take care of this
    
    tokenizer = copy.deepcopy(tokenizer) # deepcopy to avoid modification of tokenzier (the '<image>' is a placeholder, not included in actual input_ids)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    
    temp_completion = "#####"
    
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": default_system_message}] + conversations + [{"role": "assistant", "content": temp_completion}], tokenize=False)
    prompt = prompt.split(temp_completion)[0]
    
    input_ids = tokenizer.encode(prompt)
    input_ids = [IMAGE_TOKEN_INDEX if token == image_token_index else token for token in input_ids]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return input_ids 
    
    


def preprocess_llama3(
    conversations,
    frame_counts: List[int],
    tokenizer: transformers.PreTrainedTokenizer,
    default_system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    """ 
    Note that truncation is not done here, it could remove the EOS token, so it's probably more ideal to do it before this function.
    Prepare input_ids & targets for the model after applying chat template. 
    - convert <image> to IMAGE_TOKEN_INDEX and make sure targets are correct
    """
    
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer) # deepcopy to avoid modification of tokenzier (the '<image>' is a placeholder, not included in actual input_ids)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]
    

    input_ids, targets = [], []
    
    
    system_message = default_system_message
    for conv in conversations:
        role = conv.get("role") or conv.get("from")
        if role == "system":
            system_message = conv.get("content") or conv.get("value")
            break
        
    
    input_id, target = [], []
    
    input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}]) # Begin with system message
    target += [IGNORE_INDEX] * len(input_id) # mask out tokens with IGNORE_INDEX
    
    for conv in conversations:
        try: 
            role = conv["role"]
            content = conv["content"]
        except: 
            role = conv["from"]
            content = conv["value"]
            
        # content = content.replace("<image>", "<|start_image_id|> <image> <|end_image_id|>")
        
        chunks = content.split("<image>")
        new_content = chunks[0]  # Start with the first chunk (before any <image> token)
        for i, chunk in enumerate(chunks[1:], 1):  # Start from the second chunk
            if frame_counts: # Direct Place for Telling Video apart from Image
                new_content += "<image>" * frame_counts.pop(0) # Results: <|start_image_id|> <image> <image> <image> <|end_image_id|> (not really?)
            new_content += chunk
        content = new_content
        
        role = roles.get(role, role) # map towards "user" and "assistant"
        
        conv = [{"role" : role, "content" : content}]
        
        if role == "user":
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id 
            target += [IGNORE_INDEX] * len(encode_id)
        elif role == "assistant":
            mask_seq, target_seq = tokenizer.apply_chat_template(conv, tokenize=False).split(content)
            target_seq = content + target_seq
            mask_tokens = tokenizer.encode(mask_seq)[1:] # remove BOS token
            target_tokens = tokenizer.encode(target_seq)
            
            input_id += mask_tokens + target_tokens
            target += [IGNORE_INDEX] * len(mask_tokens) + target_tokens
        else:
            continue # skip over 'system' message
                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id # not sure why this is needed
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        
        input_ids.append(input_id)
        targets.append(target)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )
    
    
def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
                
    # Subsample frames according to desired fps (assuming desired fps is smaller than actual fps)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    fps = total_frame_num / video_time
    sampling_interval = round(fps / data_args.video_fps)
    frame_idx = list(range(0, total_frame_num, sampling_interval))

    if data_args.frames_upbound > 0: # additional subsampling based on interpolation
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

    num_frames_to_sample = len(frame_idx)
    frames = [video_frames[i] for i in frame_idx]
    video = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    frame_time = [frame.time for frame in frames]
    
    return video, video_time, frame_time, num_frames_to_sample

    
def modality_map(mode: int, idx: int, num_frames: int = 1):
    """ 
    One-hot scaled encoder for image & video with ids
    mode: 0 for image, 1 for video
    idx: the first image, the first video within the data_dict etc.
    num_frames: number of frames to repeat the tensor
    """
    idx = int(idx)
    base_tensor = torch.tensor([idx if mode == 0 else 0, idx if mode == 1 else 0])
    if num_frames > 1:
        return base_tensor.repeat(num_frames, 1)
    else:
        return base_tensor.unsqueeze(0)




class LazyProcessor: # For inference with VLM
    
    def __init__(self, data_args, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data = {}
        self.data_args = data_args
                
    def query(self, question: str, media_paths: List[str], id: str = "test"):
        if id not in self.data:
            self.data[id] = {"conversations": [], "media": []}
        
        if "<image>" not in question:
            question = "<image>" * len(media_paths) + " " + question

        self.data[id]["conversations"].append({"role": "user", "content": question})
        
        for media_path in media_paths:
            if media_path.lower().endswith((".jpg", ".png", ".jpeg")):
                self.data[id]["media"].append({"image": media_path})
            elif media_path.lower().endswith((".mp4", ".avi", ".mov")):
                self.data[id]["media"].append({"video": media_path})
            else:
                raise ValueError(f"Unsupported media type: {media_path}")
        
    def process_data(self, device: str = "cuda"):
        dataset = LazySupervisedDataset(self.data_args, self.tokenizer, self.image_processor, self.data)
        
        data_with_media = []
        data_without_media = []
        for _, data in self.data.items():
            images = []
            frame_counts = []
            image_count, video_count = 0, 0
            if "media" in data:
                for media_file in data["media"]:
                    if "image" in media_file:
                        image = dataset.process_image(media_file["image"])
                        image_count += 1
                        modality = modality_map(0, image_count)
                        images.append((image, modality))
                        frame_counts.append(image.shape[0])
                    elif "video" in media_file:
                        video, _, _, _ = dataset.process_video(media_file["video"])
                        processor = self.image_processor 
                        video_frames = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                        video_count += 1
                        modality = modality_map(1, video_count, video_frames.shape[0])
                        images.append((video_frames, modality))
                        frame_counts.append(video_frames.shape[0])
                        
                    
            
            input_ids = preprocess_inference_inputs(data['conversations'], self.tokenizer).to(device)
            modalities = [t[1].to(device) for t in images]
            images = [t[0].to(device) for t in images]
            if images:
                data_with_media.append({"input_ids": input_ids, "images": images, "modalities": modalities})
            else:
                data_without_media.append({"input_ids": input_ids})
                
        if data_with_media:
            input_ids = [d["input_ids"][: self.tokenizer.model_max_length] for d in data_with_media]
            labels = [d["input_ids"][: self.tokenizer.model_max_length] for d in data_with_media]
            
            images = [torch.cat(d["images"], dim=0) for d in data_with_media]
            modalities = [torch.cat(d["modalities"], dim=0) for d in data_with_media]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) # Dummy Value useless for generation

            max_img_len = max(img.size(0) for img in images)
            
            images = torch.stack([torch.nn.functional.pad(img, (0, 0, 0, 0, 0, max_img_len - img.size(0))) for img in images])
            modalities = torch.stack([torch.nn.functional.pad(mod, (0, 0, 0, max_img_len - mod.size(0))) for mod in modalities])

            batch_with_media = dict(input_ids=input_ids, 
                        labels=labels,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                        images=images,
                        modalities=modalities)
        else:
            batch_with_media = None
            
        if data_without_media:
            input_ids = [d["input_ids"][: self.tokenizer.model_max_length] for d in data_without_media]
            labels = [d["input_ids"][: self.tokenizer.model_max_length] for d in data_without_media]
            
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) # Dummy Value useless for generation
            
            batch_without_media = dict(input_ids=input_ids, 
                                       labels=labels,
                                       attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        else:
            batch_without_media = None
            
        return batch_with_media, batch_without_media


    def get_response(self, llava_model, tokenizer):
        data_w_media, data_w_text = self.process_data()
        generate_texts = []
        if data_w_media: 
            output = llava_model.generate(**data_w_media)
            for out in output:
                generate_texts.append(tokenizer.decode(out.tolist(), skip_special_tokens=True))
        if data_w_text:
            output = llava_model.generate(**data_w_text)
            for out in output:
                generate_texts.append(tokenizer.decode(out.tolist(), skip_special_tokens=True))
        return generate_texts


class LazySupervisedDataset(Dataset):
    
    def __init__(self, data_args, tokenizer, image_processor, data=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args
        if data is None:
            self.data = self.load_data(data_args.data_path)
        else:
            self.data = data

    def load_data(self, data_path):
        """ 
        Assumption here is that under data_path, a json file is provided, with input_ids interleaving text, image and video (with DEFAULT_IMAGE_TOKEN I suppose), 
        although I don't see a 'DEFAULT_VIDEO_TOKEN' here ? Perhaps we should include one ? or is it not used in inference ? It's just a interleaved text & image 
        without text, so a sequence of images, so technically video should be handled as a sequence of DEFAULT_IMAGE_TOKEN ..... who / where is it processed?
        """
        with open(data_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def process_text(self, text):
        return self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]

    def process_video(self, video_path):
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist!")
        
        if os.path.isdir(video_path): # Path is a directory saving frames separately
            frame_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            frame_files.sort()
            
            num_frames_to_sample = self.data_args.frames_upbound if self.data_args.force_sample else 10
            total_frames = len(frame_files)
            sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
            
            video = []
            for idx in sampled_indices:
                frame_path = frame_files[idx]
                try:
                    with Image.open(frame_path) as img:
                        frame = img.convert("RGB")
                        video.append(frame)
                except IOError:
                    print(f"Failed to read frame at path: {frame_path}")
            
            avg_fps = self.data_args.default_fps  # Use a default FPS or get from data_args
            video_time = total_frames / avg_fps
            frame_time = [i/avg_fps for i in sampled_indices]
        else:
            video, video_time, frame_time, num_frames_to_sample = process_video_with_pyav(video_path, self.data_args)
    
        return video, video_time, frame_time, num_frames_to_sample
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """ 
        Interleaved text, image, video
        modality: [idx, 0] for idx-th image, [0, idx] for idx-th video
        modality has same length as number of frames, to identify image and video respectively
        """
        source = self.data[i]
        conversations = copy.deepcopy(source["conversations"])
        images = []
        frame_counts = []
        image_count, video_count = 0, 0
        if "media" in source:
            for media_file in source["media"]:
                if "image" in media_file:
                    image = self.process_image(os.path.join(self.data_args.image_folder, media_file["image"]))
                    image_count += 1
                    modality = modality_map(0, image_count)
                    images.append((image, modality))
                    frame_counts.append(image.shape[0])
                elif "video" in media_file:
                    video, _, _, _ = self.process_video(os.path.join(self.data_args.video_folder, media_file["video"]))
                    processor = self.image_processor 
                    video_frames = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                    video_count += 1
                    modality = modality_map(1, video_count, video_frames.shape[0])
                    images.append((video_frames, modality))
                    frame_counts.append(video_frames.shape[0])

        # Process the conversations and create the data dictionary
        data_dict = preprocess_llama3(conversations, frame_counts, self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # Add image or video data if present | Unite video & image
        
        if images: # I think it's important to include them anyway ... ? 
            data_dict["image"] = images

        data_dict["id"] = source.get("id", i)

        return data_dict
    
    
def to_cuda(batch):
    if not torch.cuda.is_available():
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.cuda()
    elif isinstance(batch, dict):
        return {key: to_cuda(value) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [to_cuda(item) for item in batch]
    else:
        return batch
        
@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        # Obviously I want to pad on the right (during inference, I would like the model to begin by generating meaningful contents and not padding tokens ...)
        input_ids = pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """ 
        Padding: 
        input_ids: tokenizer.pad_token_id 
        images: torch.zeros 
        modalities: torch.zeros 
        lables: IGNORE_INDEX 
        """        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        
        assert self.tokenizer.pad_token_id is not None, "Pad token id is not set!" # LLaVA uses pad_token_id = 0, which means '!' and claims good performance
            
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        image_tuples = [instance["image"] for instance in instances]
        
        images = [torch.cat([im[0] for im in im_list], dim=0) for im_list in image_tuples]
        modalities = [torch.cat([im[1] for im in im_list], dim=0) for im_list in image_tuples]
        
        # print("Modalities shape: ", modalities[0].shape)
        
        max_img_len = max(img.size(0) for img in images)
    
        images = torch.stack([torch.nn.functional.pad(img, (0, 0, 0, 0, 0, max_img_len - img.size(0))) for img in images])
        modalities = torch.stack([torch.nn.functional.pad(mod, (0, 0, 0, max_img_len - mod.size(0))) for mod in modalities])
        
        batch = dict(input_ids=input_ids, 
                     labels=labels.long() if labels.dtype == torch.int32 else labels, 
                     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                     images=images,
                     modalities=modalities)
        
        return to_cuda(batch)
    
    
from config import DataArguments 
from tqdm import tqdm as tqdm
import datasets
from datasets import load_dataset
    
    
def prepare_docci_data(output_json_path, image_folder="data/docci"):
    
    docci_dataset = load_dataset("google/docci", trust_remote_code=True) # load docci dataset

    data_json = []
    os.makedirs(image_folder, exist_ok=True)
    
    for idx, item in tqdm(enumerate(docci_dataset['train']), desc="Processing", total=len(docci_dataset['train'])):
        description = item['description']
        image = item['image']
        
        # Store the image and record the image path
        img_filename = f"docci_{idx}.jpg"
        img_path = os.path.join(image_folder, img_filename)
        
        # Save the image
        image.save(img_path)
        
        # Create the compatible json structure
        data_item = {
            "id": f"docci_{idx}",
            "media": [
                {"image": img_filename}
            ],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nCan you describe this image?"
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        
        data_json.append(data_item)
    
    # Save the converted data to a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(data_json, f, indent=2)
        
    data_args = DataArguments(
        data_path = output_json_path,
        image_folder = image_folder + "/",
        video_folder = image_folder + "/",
        video_fps = 1,
        frames_upbound = 0,
        add_time_instruction = False,
        force_sample = False,
        default_fps = 10
    )
    
    return data_args


def generate_text(prompt, model, tokenizer, device, generation_config):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        **generation_config
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text