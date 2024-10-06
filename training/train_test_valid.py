import os
from os.path import join, relpath, isfile, abspath
import random
import glob
import time

from math import sqrt, floor
import numpy as np
from imageio import imread
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd

import sys
import os
from os.path import join, isdir, isfile, splitext
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.mydatasets import ImageLaSOT, ImageLaSOT_val, ImageLaSOT_test
import training.optim as optim
from torch.utils.data import DataLoader
from training.models import BaselineEmbeddingNet, SiameseNet
import training.losses as losses
from training.labels import create_BCELogit_loss_label as BCELoss



def center_error(output, upscale_factor=4):
    b = output.shape[1]  # 8
    s = output.shape[-1]  # 33
    print('b', b)
    print('s', s)
    
    out_flat = output.view(b, -1)
    max_idx = torch.argmax(out_flat, dim=1)
    estim_center = torch.stack([max_idx // s, max_idx % s], dim=1)
    
    center = torch.tensor([s / 2, s / 2], device=output.device)
    
    dist = torch.norm(estim_center.float() - center, dim=1)
    c_error = dist.mean()
    
    c_error = c_error * upscale_factor
    
    return c_error

# Load the model
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "training/BaselinePretrained.pth"

# Instantiate the model architecture
embedding_net_instance = BaselineEmbeddingNet()
model = SiameseNet(embedding_net=embedding_net_instance)

# Load the model state dictionary with map_location to CPU
checkpoint=torch.load(model_path, map_location=device,weights_only=True)

model_state_dict = checkpoint['state_dict']
model.load_state_dict(model_state_dict)
model.to(device)
optimizer = optim.get_Adam(model.parameters(), lr=0.02)
imageLaSOT = ImageLaSOT('home/')

train_dataloader = DataLoader(imageLaSOT, batch_size=8, shuffle = True)


train_loss_per_epoch=[]

for epoch in range(10):
    model.train()
    running_loss = 0.0
    counter = 0
    prev_time = time.time()
    for i, data in enumerate(train_dataloader):
        #if i >= 6:  # Stop after the first 4 batches
        #    break
        # print(i)
    # Every data instance is an input + label pair
        ref_frame_tensor = data['ref_frame']
        srch_frame_tensor = data['srch_frame']
        label = data['label']
        label=label.to(device)
        ref_frame_tensor=ref_frame_tensor.to(device)
        srch_frame_tensor=srch_frame_tensor.to(device)
        # print(f"Batch {i + 1}")
        # print(f"ref_frame_tensor shape: {ref_frame_tensor.shape}")
        # print(f"srch_frame_tensor shape: {srch_frame_tensor.shape}")
        # print(f"label shape: {label.shape}")

        #model.train()
    
        optimizer.zero_grad()
        output = model(ref_frame_tensor,srch_frame_tensor)
        output=output.permute(1,0,2,3)
        center_error(output=output)
        loss = losses.BCELogit_Loss(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        counter += 1
        
        if counter % 100 == 0:
            print("batch counter:", counter)
            curr_time = time.time()
            print("batch time:", curr_time - prev_time)
            prev_time = curr_time

            print(center_error(output, data['srch_cx'].to(device), data['srch_cy'].to(device), data['srch_x'].to(device), data['srch_y'].to(device),data['width_cropped'].to(device), data['height_cropped'].to(device)))
    
    train_loss_per_epoch.append(running_loss)
    print(f'Epoch [{epoch+1}/{10}], Loss: {running_loss}')
    model_path = "model" + "_" + str(epoch) + ".pth"
    torch.save(model.state_dict(), model_path)


plt.figure(figsize=(10, 6))
plt.plot(train_loss_per_epoch, marker='o', linestyle='-', color='b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss through Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.clf()