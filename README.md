<!-- <div style="display: flex; align-items: center; margin-bottom: 20px;"> -->
<div align="center">
  <img src="data/title.png" width="300" alt="llava">
</div>
<div align="center">
  <img src="data/mini-llava-visual.gif" width="800" alt="Mini-LLaVA Demo">
  <p><em>Mini-LLaVA handles text, image and video inputs.</em></p>
</div>

This minimal yet powerful codebase transforms LLaMA 3.1 into a vision-language model, giving it the ability to "see" and process visual information. 📷💻

I’ve learned a ton from the LLaVA project, and while it’s incredibly powerful, I felt there was room for a simplified implementation that anyone can use, modify, and extend easily.

Here’s what Mini-LLaVA does:

🛠️ Minimal Structure: A clean and lightweight codebase, designed for easy understanding and quick experimentation.
🎥 Enhanced Multimodal Input: Supports interleaving multiple images and video inputs, tackling a wider range of vision-language tasks.
🐱 Pretrained Vision Projector: With some minor tuning, Mini-LLaVA can already recognize images (like a cat, for instance! 🐱) out of the box!



## 🔥 Updates
- [09/2024] [Minimal Implementation] [Tutorial in Mini_LLaVA.ipynb](Mini_LLaVA.ipynb) showing how a pre-trained adaptor could helps Llama3.1 to see.


## 💡 Features
- Minimal Code Structure: Transform a language model (Llama 3.1) into a powerful vision-language model with minimal, easy-to-understand code.
- Simplified Implementation: Our code is significantly simpler than the original LLaVA implementation, making it easier to dive into and build upon.
- Extended Functionality: We've added support for interleaved processing of images, videos, and text, giving you more flexibility and power.

## 🚧 TODO 
- [ ] Fine-tune on language decoder
- [ ] Audio modality
- [ ] Retrieval modality
- [ ] Benchmark inference test


## Environment Set-up
```shell
run set.sh
```


