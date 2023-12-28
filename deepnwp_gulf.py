import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np
import sys

import torch



from swin import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import xarray as xr



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print("Is CUDA available:", cuda_available)

# If CUDA is available, you can also check the number of GPUs available and the current active GPU
if cuda_available:
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Current active CUDA Device:", torch.cuda.current_device())
    print("Name of current CUDA Device:", torch.cuda.get_device_name(torch.cuda.current_device()))


path = '/apollo/deepnwp/nc/'
ds = xr.open_dataset(path+'deepnwp_2017_04_22_12.nc')
#ds



import numpy as np
import pandas as pd
import xarray as xr
import os
import subprocess
import warnings
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

# [Include the CubeSphereConv2D, CubeSpherePadding2D, and CubeSphereModel class definitions here]

# # Swin Transformer for Gulf Coast

# In[41]:

#path = ''
input_file = path+'deepnwp_2017_04_22_12.nc'
ds = xr.open_dataset(input_file)
#ds.sel(lat=slice(37,16)).sel(lon=slice(360-110,360-76))


from tqdm import tqdm

data_prep = True # False
if data_prep:
    tas_gfs_ = []
    pr_gfs_  = []
    pr_gpcp_ = []


    input_file = path+'deepnwp_2017_04_22_12.nc'

    import glob
    # Define the pattern to match the filenames
    pattern = path+'deepnwp_????_??_??_??.nc'

    # Use glob.glob to find all files matching the pattern
    filenames = glob.glob(pattern)

    # Loop over the found files
    for filename in tqdm(filenames):
        #print(filename)
        input_file = filename

        ds_remap = xr.open_dataset(input_file).sel(lat=slice(41,12)).sel(lon=slice(360-107,360-78))

        tas_gfs_.append(ds_remap.tas_gfs.values)
        pr_gfs_.append(ds_remap.pr_gfs.values)
        pr_gpcp_.append(ds_remap.pr_gpcp.values)

    tas_gfs_input = np.array(tas_gfs_)
    pr_gfs_input = np.array(pr_gfs_)
    pr_gpcp_input = np.array(pr_gpcp_)    
    print(tas_gfs_input.shape, pr_gfs_input.shape, pr_gpcp_input.shape)


# In[176]:


# Concatenate arrays along the second dimension (dimension index 1)
#print(pr_gfs_max, tas_gfs_max, pr_gpcp_max)

pr_gfs_max = np.max(pr_gfs_input)
tas_gfs_max = np.max(tas_gfs_input)
pr_gpcp_max = np.max(pr_gpcp_input)

print(pr_gfs_max, tas_gfs_max, pr_gpcp_max)

tas_gfs_input_norm = tas_gfs_input / tas_gfs_max
pr_gfs_input_norm = pr_gfs_input / pr_gfs_max
pr_gpcp_input_norm = pr_gpcp_input / pr_gpcp_max

concatenated_array = np.concatenate((tas_gfs_input_norm, pr_gfs_input_norm), axis=1)

# Convert to PyTorch tensor
input_tensor = torch.tensor(concatenated_array,  dtype=torch.float32)
output_tensor = torch.tensor(pr_gpcp_input_norm, dtype=torch.float32)


# In[177]:


print(input_tensor.shape, output_tensor.shape)


# In[178]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split

class deepnwpDataset(Dataset):
    def __init__(self, inputs, outputs):
        # Assuming inputs and outputs are numpy arrays or PyTorch tensors of the same shape
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]
        return input_sample, output_sample
    
dataset = deepnwpDataset(input_tensor, output_tensor)

# Define the sizes for train, validation, and test sets
total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% for training
val_size = int(0.15 * total_size)   # 15% for validation
test_size = total_size - train_size - val_size  # Remaining for test

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# You can then create DataLoader instances for each set if needed
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Use the DataLoader in your training loop
for inputs, outputs in train_loader:
    print(inputs.shape, outputs.shape)
    height = inputs.shape[2]
    width = inputs.shape[3]
    break

#sys.exit()
# In[179]:



upscale = 1

window_size = 5

#height = 30 #(1024 // upscale // window_size + 1) * window_size

#width = 30 #(720 // upscale // window_size + 1) * window_size

device = 'cuda'

model = SwinIR(upscale=1, img_size=(height, width), in_chans=6,
               window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
               embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
for inputs, outputs in train_loader:
#    print(inputs.shape, outputs.shape)
    height = inputs.shape[2]
    width = inputs.shape[3]
    model_output = model(inputs.to(device))
    print('input shape = ', inputs.to(device).shape,'model output shape = ', model_output.shape, 'actual shape = ', outputs.to(device).shape)
    break


#sys.exit()
# Initialize the model
#model = CubeSphereModel(channels, base_filter_number, cso, integration_steps, skip_connections, kernel_size, strides, padding, dilation_rate)

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim


# Define the loss function (MSE for regression tasks)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation data loaders
# train_loader and val_loader should be defined

epochs = 2000  # Adjust as needed
best_val_loss = float('inf')  # Initialize best validation loss


# Training loop
for epoch in tqdm(range(epochs)):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Compute validation loss
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No gradient computation in validation phase
        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()

    # Print training and validation loss
    print(f'Epoch {epoch+1} - Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')

 # Check if the current model is the best

    if val_loss < best_val_loss:
        best_val_loss = val_loss  # Update the best validation loss
        torch.save(model.state_dict(), f'best_model_epoch_{epoch+1}.pth')
        print(f'Best model saved at epoch {epoch+1} with validation loss {val_loss}')

#    # Save the model every 500 epochs
#    if (epoch + 1) % 500 == 0:
#        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
#        print(f'Model saved at epoch {epoch+1}')

print('Finished Training')


# In[ ]:





# In[ ]:




