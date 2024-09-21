<!-- <div style="display: flex; align-items: center; margin-bottom: 20px;"> -->
<div align="center">
  <img src="data/title.png" width="300" alt="llava">
</div>
<div align="center">
  <img src="data/mini-llava-visual.gif" width="800" alt="Mini-LLaVA Demo">
  <p><em>Mini-LLaVA handles text, image and video inputs.</em></p>
</div>

Welcome to Mini-LLaVA – a minimal and seamless implementation of the LLaVA model, specifically crafted to help you unlock the true multimodal potential of a Large Language Model (based on Llama-3.1) with just a single GPU.

This project goes above and beyond the original by introducing powerful support for interleaved processing of multiple input types—including images, videos, and text—all respecting their order of appearance. Whether you're handling complex visual-textual correlations or want seamless transitions between media formats, Mini-LLaVA has you covered with minimal code and maximum flexibility.

🚦 TL;DR: Mini-LLaVA is the simplest and smartest way to convert a language model (LLaMA 3.1) into a multimodal powerhouse capable of handling text, images, and even videos—all on a single GPU! 


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


