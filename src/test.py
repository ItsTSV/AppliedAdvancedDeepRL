import torch
import torchvision
import torchaudio

# Verify autogen of requirements.txt via pipreqs
print(f"Torch version {torch.__version__}")
print(f"Torchvision version {torchvision.__version__}")
print(f"Torchaudio version {torchaudio.__version__}")
