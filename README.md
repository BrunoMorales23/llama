Reminder: All content from Ollama is located in AppData Dir
LLM used 'Llama 3.2' | Embed used 'mxbai-embed-large'
Ollama v0.9.3 for this test

GPU: XFX Radeon RX 6500 XT 4gb
CPU: AMD Ryzen 5 5500
RAM: 16GB generic

(deprecated, not valid on AMD)
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CUDA NOT AVAIABLE FOR AMD GPU. IF GPU IS REALLY REQUIRED, THEN CHECK FOR 'Google Colab' FOR FURTHER INFORMATION

AMD has an alternative with CUDA: ROCm (Radeon Open Compute).
But ROCm is not compatible with Windows (neither with all AMD GPUs).