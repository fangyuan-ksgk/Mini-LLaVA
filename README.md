<!-- <div style="display: flex; align-items: center; margin-bottom: 20px;"> -->
<div align="center">
  <img src="data/title.png" width="300" alt="llava">
</div>
<div align="center">
  <img src="data/mini-llava-visual.gif" width="800" alt="Mini-LLaVA Demo">
  <p><em>Mini-LLaVA handles text, image and video inputs.</em></p>
</div>


Mini-LlaVA is a minimal implementation of the LLaVA model, designed to help us learn how to unlock multimodal capabilities of Large Language Models (LLMs) using a single GPU. This project extends the original LLaVA concept by enabling interleaved processing of multiple images, videos, and text inputs respecting their order of appearance.

## :new: Updates
- [09/2024] [Minimal Implementation] Tutorial in Mini_LLaVA.ipynb showing how a pre-trained adaptor could helps Llama3.1 to see.


## Features
- Minimal implementation of LLaVA
- Interleaved processing of multiple modalities of any number, obeying order of their inputs:
  - Images
  - Videos
  - Text

## TODO 
- [ ] Fine-tune on language decoder
- [ ] Audio modality
- [ ] Retrieval modality
- [ ] Benchmark inference test


## Environment Set-up
```shell
run set.sh
```


