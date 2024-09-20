pip install --upgrade transformers trl huggingface_hub datasets accelerate bitsandbytes peft vllm deepspeed
MAX_JOBS=4 pip install flash-attn -U --no-build-isolation --force-reinstall
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install av