import torch
import torchvision.transforms as transforms
import math
from torchvision.io import read_image as read_image
from torchvision.io import ImageReadMode as ImageReadMode
import torch.nn.functional as F

def crop_and_resize(image_path, x, y, w, h, output_size):
    # Calculate the original crop size
    image = read_image(path = image_path, mode = ImageReadMode.RGB).float()
    crop_size = 0.125 * math.sqrt((5 * w + h) * (w + 5 * h))
    
    # Calculate the new crop size based on the desired output size
    new_crop_size = crop_size * output_size / 127
    
    # Coordinates of the center of the bounding box
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    
    # Calculate the crop boundaries
    half_crop_size = new_crop_size / 2
    left = cx - half_crop_size
    right = cx + half_crop_size
    top = cy - half_crop_size
    bottom = cy + half_crop_size

    # Get image dimensions
    img_height, img_width = image.shape[1], image.shape[2]

    # Calculate padding if necessary
    pad_left = int(max(0, -left))
    pad_right = int(max(0, right - img_width))
    pad_top = int(max(0, -top))
    pad_bottom = int(max(0, bottom - img_height))

    # Add padding to the image if required
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        mean_pixel = image.mean(dim=(1, 2), keepdim=True)
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        # Create padding tensor
        image = image.unsqueeze(0)  # Add batch dimension
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        image = image.squeeze(0)  # Remove batch dimension

        # Apply mean pixel values to padding regions
        image[:, :pad_top, :] = mean_pixel
        image[:, -pad_bottom:, :] = mean_pixel
        image[:, :, :pad_left] = mean_pixel
        image[:, :, -pad_right:] = mean_pixel

    # Adjust the coordinates after padding
    left = max(0, left)
    right = left + new_crop_size
    top = max(0, top)
    bottom = top + new_crop_size

    # Crop the image
    image_cropped = image[:, int(top):int(bottom), int(left):int(right)]

    # Resize the cropped image to the desired output size
    resize_transform = transforms.Resize((output_size, output_size))
    image_resized = resize_transform(image_cropped)

    return image_resized

