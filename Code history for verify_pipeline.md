Code history for verify\_pipeline



# 1

import time

import torch

from use\_croma import PretrainedCROMA



device = "cpu"



print("1. Loading MARIDA dataset...")

dataset = torch.load("MARIDA\_preprocessed.pt", map\_location=device, weights\_only=False)



\# Inspect tensor shapes

print(f"   Radar tensor shape:   {dataset\['radar'].shape}")

print(f"   Optical tensor shape: {dataset\['optical'].shape}\\n")



print("2. Loading CROMA Baseline Model...")

model = PretrainedCROMA(

&#x20;   pretrained\_path="CROMA\_base.pt",

&#x20;   size="base",

&#x20;   modality="both",

&#x20;   image\_resolution=120

)

model.to(device)

model.eval()

print("   Model loaded successfully!\\n")



\# 3. Grab a test batch of 4 satellite images

sar\_batch = dataset\['radar']\[:4].float().to(device)

opt\_batch = dataset\['optical']\[:4].float().to(device)



print("3. Running Forward Pass \& Measuring Baseline Latency...")

start\_time = time.perf\_counter()



with torch.no\_grad():

&#x20;   # Pass SAR + Optical images through CROMA to extract embeddings

&#x20;   joint\_embeddings, optical\_embeddings, sar\_embeddings = model(

&#x20;       SAR\_images=sar\_batch, 

&#x20;       optical\_images=opt\_batch

&#x20;   )



end\_time = time.perf\_counter()

latency\_ms = (end\_time - start\_time) \* 1000



print("--- BASELINE SUCCESS ---")

print(f"Joint Feature Output Shape: {joint\_embeddings.shape}")

print(f"Inference Latency for 4 images: {latency\_ms:.2f} ms")

print(f"Average Latency per image:     {latency\_ms / 4:.2f} ms")



output



# 2

import os

import glob

import time

import torch

import numpy as np

import rasterio  # Ensure rasterio or tifffile is installed: pip install rasterio

from use\_croma import PretrainedCROMA



device = "cpu"



\# ==========================================

\# 1. Official CROMA Normalization

\# ==========================================

def normalize(x, use\_8\_bit=False):

&#x20;   """

&#x20;   Official normalization logic provided in CROMA README.

&#x20;   Clips imagery to \[mean - 2\*std, mean + 2\*std] per channel.

&#x20;   """

&#x20;   x = x.float()

&#x20;   imgs = \[]

&#x20;   for channel in range(x.shape\[1]):

&#x20;       mean\_val = x\[:, channel, :, :].mean()

&#x20;       std\_val = x\[:, channel, :, :].std()

&#x20;       min\_value = mean\_val - 2 \* std\_val

&#x20;       max\_value = mean\_val + 2 \* std\_val



&#x20;       if use\_8\_bit:

&#x20;           img = (x\[:, channel, :, :] - min\_value) / (max\_value - min\_value + 1e-8) \* 255.0

&#x20;           img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)

&#x20;       else:

&#x20;           img = (x\[:, channel, :, :] - min\_value) / (max\_value - min\_value + 1e-8)

&#x20;           img = torch.clip(img, 0, 1).unsqueeze(dim=1)

&#x20;       imgs.append(img)



&#x20;   return torch.cat(imgs, dim=1)



\# ==========================================

\# 2. Helper to Load a Batch from benv1\_14k

\# ==========================================

def load\_ben14k\_batch(dataset\_dir="benv1\_14k", batch\_size=4):

&#x20;   """

&#x20;   Loads a batch of spatially aligned S1 (SAR) and S2 (Optical) patches 

&#x20;   from the local benv1\_14k directory.

&#x20;   """

&#x20;   # Locate s1 and s2 directories (Updated to match lowercase folders)

&#x20;   s1\_dirs = sorted(glob.glob(os.path.join(dataset\_dir, "s1", "\*")))

&#x20;   s2\_dirs = sorted(glob.glob(os.path.join(dataset\_dir, "s2", "\*")))



&#x20;   if not s1\_dirs or not s2\_dirs:

&#x20;       raise ValueError(f"Could not find subdirectories inside {dataset\_dir}/s1 or {dataset\_dir}/s2. Check your paths!")



&#x20;   sar\_tensors = \[]

&#x20;   opt\_tensors = \[]



&#x20;   for i in range(min(batch\_size, len(s1\_dirs))):

&#x20;       s1\_path = s1\_dirs\[i]

&#x20;       s2\_path = s2\_dirs\[i]



&#x20;       # Load S1 (2 Channels)

&#x20;       if os.path.isfile(s1\_path) and s1\_path.endswith('.npy'):

&#x20;           s1\_arr = np.load(s1\_path)  # Expected shape: (2, 120, 120)

&#x20;       else:

&#x20;           # If folder containing band TIFs

&#x20;           tif\_files = sorted(glob.glob(os.path.join(s1\_path, "\*.tif")))

&#x20;           if len(tif\_files) < 2:

&#x20;               raise FileNotFoundError(f"Not enough .tif files in {s1\_path}")

&#x20;           bands = \[rasterio.open(f).read(1) for f in tif\_files\[:2]]

&#x20;           s1\_arr = np.stack(bands, axis=0)



&#x20;       # Load S2 (12 Channels)

&#x20;       if os.path.isfile(s2\_path) and s2\_path.endswith('.npy'):

&#x20;           s2\_arr = np.load(s2\_path)  # Expected shape: (12, 120, 120)

&#x20;       else:

&#x20;           tif\_files = sorted(glob.glob(os.path.join(s2\_path, "\*.tif")))

&#x20;           # Drop B10 (cirrus) to keep exactly 12 bands

&#x20;           bands = \[rasterio.open(f).read(1) for f in tif\_files if "B10" not in f]\[:12]

&#x20;           if len(bands) != 12:

&#x20;               raise ValueError(f"Expected 12 bands in {s2\_path}, found {len(bands)}")

&#x20;           s2\_arr = np.stack(bands, axis=0)



&#x20;       sar\_tensors.append(torch.from\_numpy(s1\_arr).float())

&#x20;       opt\_tensors.append(torch.from\_numpy(s2\_arr).float())



&#x20;   sar\_batch = torch.stack(sar\_tensors)  # Shape: (B, 2, 120, 120)

&#x20;   opt\_batch = torch.stack(opt\_tensors)  # Shape: (B, 12, 120, 120)



&#x20;   return sar\_batch, opt\_batch





\# ==========================================

\# 3. Execution Pipeline

\# ==========================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   print("1. Loading benv1\_14k sample batch...")

&#x20;   try:

&#x20;       # Assuming the script is run from the 'code' directory 

&#x20;       # and 'benv1\_14k' is inside 'code'

&#x20;       sar\_raw, opt\_raw = load\_ben14k\_batch("benv1\_14k", batch\_size=4)

&#x20;       print(f"   Raw SAR shape:     {sar\_raw.shape}")

&#x20;       print(f"   Raw Optical shape: {opt\_raw.shape}\\n")

&#x20;   except Exception as e:

&#x20;       print(f"Error loading files directly from folder: {e}")

&#x20;       print("Exiting pipeline. Please check the error above.")

&#x20;       exit(1)



&#x20;   print("2. Applying Official CROMA Normalization...")

&#x20;   sar\_batch = normalize(sar\_raw, use\_8\_bit=False).to(device)  # (4, 2, 120, 120)

&#x20;   opt\_batch = normalize(opt\_raw, use\_8\_bit=False).to(device)  # (4, 12, 120, 120)

&#x20;   print("   Normalization complete!\\n")



&#x20;   print("3. Loading CROMA Baseline Model...")

&#x20;   model = PretrainedCROMA(

&#x20;       pretrained\_path="CROMA\_base.pt",

&#x20;       size="base",

&#x20;       modality="both",

&#x20;       image\_resolution=120

&#x20;   )

&#x20;   model.to(device)

&#x20;   model.eval()

&#x20;   print("   Model loaded successfully!\\n")



&#x20;   print("4. Running Forward Pass \& Measuring Latency...")

&#x20;   start\_time = time.perf\_counter()



&#x20;   with torch.no\_grad():

&#x20;       outputs = model(SAR\_images=sar\_batch, optical\_images=opt\_batch)



&#x20;   end\_time = time.perf\_counter()

&#x20;   latency\_ms = (end\_time - start\_time) \* 1000



&#x20;   print("\\n--- BASELINE SUCCESS ---")

&#x20;   print(f"Joint Feature Output Shape ('joint\_GAP'):      {outputs\['joint\_GAP'].shape}")

&#x20;   print(f"Joint Patch Encoding Shape ('joint\_encodings'): {outputs\['joint\_encodings'].shape}")

&#x20;   print(f"Inference Latency for 4 images:                {latency\_ms:.2f} ms")

&#x20;   print(f"Average Latency per image:                    {latency\_ms / 4:.2f} ms")



OUTPUT:

1\. Loading benv1\_14k sample batch...

C:\\The FOLDER\\LALLA'S  PROJECT - ACTUALITY\\Research Honours\\code\\verify\_pipeline.py:67: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.

As an alternative, you can create a new view using np.reshape (with copy=False if needed).

&#x20; bands = \[rasterio.open(f).read(1) for f in tif\_files\[:2]]

C:\\The FOLDER\\LALLA'S  PROJECT - ACTUALITY\\Research Honours\\code\\verify\_pipeline.py:76: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.

As an alternative, you can create a new view using np.reshape (with copy=False if needed).

&#x20; bands = \[rasterio.open(f).read(1) for f in tif\_files if "B10" not in f]\[:12]

Error loading files directly from folder: all input arrays must have the same shape

Exiting pipeline. Please check the error above.



# 3

import os

import glob

import time

import torch

import numpy as np

import rasterio

from rasterio.enums import Resampling

from use\_croma import PretrainedCROMA



device = "cpu"



\# ==========================================

\# 1. Official CROMA Normalization

\# ==========================================

def normalize(x, use\_8\_bit=False):

&#x20;   x = x.float()

&#x20;   imgs = \[]

&#x20;   for channel in range(x.shape\[1]):

&#x20;       mean\_val = x\[:, channel, :, :].mean()

&#x20;       std\_val = x\[:, channel, :, :].std()

&#x20;       min\_value = mean\_val - 2 \* std\_val

&#x20;       max\_value = mean\_val + 2 \* std\_val



&#x20;       if use\_8\_bit:

&#x20;           img = (x\[:, channel, :, :] - min\_value) / (max\_value - min\_value + 1e-8) \* 255.0

&#x20;           img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)

&#x20;       else:

&#x20;           img = (x\[:, channel, :, :] - min\_value) / (max\_value - min\_value + 1e-8)

&#x20;           img = torch.clip(img, 0, 1).unsqueeze(dim=1)

&#x20;       imgs.append(img)



&#x20;   return torch.cat(imgs, dim=1)



\# ==========================================

\# 2. Helper to Load a Batch from benv1\_14k

\# ==========================================

def read\_and\_resize(filepath, target\_shape=(120, 120)):

&#x20;   """Reads a tif file and forces it to a target shape (120x120)."""

&#x20;   with rasterio.open(filepath) as dataset:

&#x20;       data = dataset.read(

&#x20;           1,

&#x20;           out\_shape=target\_shape,

&#x20;           resampling=Resampling.bilinear

&#x20;       )

&#x20;   return data



def load\_ben14k\_batch(dataset\_dir="benv1\_14k", batch\_size=4):

&#x20;   """

&#x20;   Loads a batch of spatially aligned S1 (SAR) and S2 (Optical) patches 

&#x20;   from the local benv1\_14k directory.

&#x20;   """

&#x20;   s1\_dirs = sorted(glob.glob(os.path.join(dataset\_dir, "s1", "\*")))

&#x20;   s2\_dirs = sorted(glob.glob(os.path.join(dataset\_dir, "s2", "\*")))



&#x20;   if not s1\_dirs or not s2\_dirs:

&#x20;       raise ValueError(f"Could not find subdirectories inside {dataset\_dir}/s1 or {dataset\_dir}/s2. Check your paths!")



&#x20;   sar\_tensors = \[]

&#x20;   opt\_tensors = \[]



&#x20;   for i in range(min(batch\_size, len(s1\_dirs))):

&#x20;       s1\_path = s1\_dirs\[i]

&#x20;       s2\_path = s2\_dirs\[i]



&#x20;       # Load S1 (2 Channels)

&#x20;       if os.path.isfile(s1\_path) and s1\_path.endswith('.npy'):

&#x20;           s1\_arr = np.load(s1\_path)

&#x20;       else:

&#x20;           tif\_files = sorted(glob.glob(os.path.join(s1\_path, "\*.tif")))

&#x20;           if len(tif\_files) < 2:

&#x20;               raise FileNotFoundError(f"Not enough .tif files in {s1\_path}")

&#x20;           # Resize all SAR bands to 120x120

&#x20;           bands = \[read\_and\_resize(f) for f in tif\_files\[:2]]

&#x20;           s1\_arr = np.stack(bands, axis=0)



&#x20;       # Load S2 (12 Channels)

&#x20;       if os.path.isfile(s2\_path) and s2\_path.endswith('.npy'):

&#x20;           s2\_arr = np.load(s2\_path)

&#x20;       else:

&#x20;           tif\_files = sorted(glob.glob(os.path.join(s2\_path, "\*.tif")))

&#x20;           # Drop B10 and resize all optical bands to 120x120

&#x20;           bands = \[read\_and\_resize(f) for f in tif\_files if "B10" not in f]\[:12]

&#x20;           if len(bands) != 12:

&#x20;               raise ValueError(f"Expected 12 bands in {s2\_path}, found {len(bands)}")

&#x20;           s2\_arr = np.stack(bands, axis=0)



&#x20;       sar\_tensors.append(torch.from\_numpy(s1\_arr).float())

&#x20;       opt\_tensors.append(torch.from\_numpy(s2\_arr).float())



&#x20;   sar\_batch = torch.stack(sar\_tensors)

&#x20;   opt\_batch = torch.stack(opt\_tensors)



&#x20;   return sar\_batch, opt\_batch





\# ==========================================

\# 3. Execution Pipeline

\# ==========================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   print("1. Loading benv1\_14k sample batch...")

&#x20;   try:

&#x20;       sar\_raw, opt\_raw = load\_ben14k\_batch("benv1\_14k", batch\_size=4)

&#x20;       print(f"   Raw SAR shape:     {sar\_raw.shape}")

&#x20;       print(f"   Raw Optical shape: {opt\_raw.shape}\\n")

&#x20;   except Exception as e:

&#x20;       print(f"Error loading files directly from folder: {e}")

&#x20;       print("Exiting pipeline. Please check the error above.")

&#x20;       exit(1)



&#x20;   print("2. Applying Official CROMA Normalization...")

&#x20;   sar\_batch = normalize(sar\_raw, use\_8\_bit=False).to(device)

&#x20;   opt\_batch = normalize(opt\_raw, use\_8\_bit=False).to(device)

&#x20;   print("   Normalization complete!\\n")



&#x20;   print("3. Loading CROMA Baseline Model...")

&#x20;   model = PretrainedCROMA(

&#x20;       pretrained\_path="CROMA\_base.pt",

&#x20;       size="base",

&#x20;       modality="both",

&#x20;       image\_resolution=120

&#x20;   )

&#x20;   model.to(device)

&#x20;   model.eval()

&#x20;   print("   Model loaded successfully!\\n")



&#x20;   print("4. Running Forward Pass \& Measuring Latency...")

&#x20;   start\_time = time.perf\_counter()



&#x20;   with torch.no\_grad():

&#x20;       outputs = model(SAR\_images=sar\_batch, optical\_images=opt\_batch)



&#x20;   end\_time = time.perf\_counter()

&#x20;   latency\_ms = (end\_time - start\_time) \* 1000



&#x20;   print("\\n--- BASELINE SUCCESS ---")

&#x20;   print(f"Joint Feature Output Shape ('joint\_GAP'):      {outputs\['joint\_GAP'].shape}")

&#x20;   print(f"Joint Patch Encoding Shape ('joint\_encodings'): {outputs\['joint\_encodings'].shape}")

&#x20;   print(f"Inference Latency for 4 images:                {latency\_ms:.2f} ms")

&#x20;   print(f"Average Latency per image:                    {latency\_ms / 4:.2f} ms")



OUTPUT:

1\. Loading benv1\_14k sample batch...

C:\\The FOLDER\\LALLA'S  PROJECT - ACTUALITY\\Research Honours\\code\\verify\_pipeline.py:40: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.

As an alternative, you can create a new view using np.reshape (with copy=False if needed).

&#x20; data = dataset.read(

&#x20;  Raw SAR shape:     torch.Size(\[4, 2, 120, 120])

&#x20;  Raw Optical shape: torch.Size(\[4, 12, 120, 120])



2\. Applying Official CROMA Normalization...

&#x20;  Normalization complete!



3\. Loading CROMA Baseline Model...

Initializing SAR encoder

Initializing optical encoder

Initializing joint SAR-optical encoder

&#x20;  Model loaded successfully!



4\. Running Forward Pass \& Measuring Latency...



\--- BASELINE SUCCESS ---

Joint Feature Output Shape ('joint\_GAP'):      torch.Size(\[4, 768])

Joint Patch Encoding Shape ('joint\_encodings'): torch.Size(\[4, 225, 768])

Inference Latency for 4 images:                1818.10 ms

Average Latency per image:                    454.53 ms



# 4



