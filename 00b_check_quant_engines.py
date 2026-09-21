import torch
print("supported engines on this build:", torch.backends.quantized.supported_engines)
print("torch version:", torch.__version__)

import platform
print(platform.machine(), platform.system(), platform.release())