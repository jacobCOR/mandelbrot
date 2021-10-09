# By Jacob and Al

from PIL import Image 
import numpy as np
import colorsys 
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
  
# setting the width of the output image as 1024 
WIDTH = 1024
device = 'cuda'
  
# a function to return a tuple of colors 
# as integer value of rgb 
def rgb_conv(i): 
    color = 255 * np.array(colorsys.hsv_to_rgb(i / 255.0, 1.0, 0.5)) 
    return tuple(color.astype(int)) 
  
# function defining a mandelbrot 
def mandelbrot(c0, img, iters=1000): 

    # Initialize c
    c = torch.zeros((c0.shape[0], c0.shape[1]), device=device)

    # tracker to remember which pixels have already been modified
    tracker = torch.zeros((c0.shape[0], c0.shape[1]), device=device)

    pixels = torch.tensor(np.array(img), device=device, dtype=torch.int64)

    for i in range(1, iters): 

        mask = torch.logical_and(torch.abs(c) > 2, tracker == 0)

        pixels[mask] = torch.tensor(rgb_conv(i),device=device, dtype=torch.int64)

        # Update tracker
        tracker[mask] = -1

        # Avoid overflow by setting c and c0 to 0 for pixels we've already set
        c[mask] = 0
        c0[mask] = 0

        c = c * c + c0 


    # Set remaining untouched pixels to black
    pixels[tracker == 0] = torch.tensor((0,0,0), device=device, dtype=torch.int64) 

    return pixels
  
# creating the new image in RGB mode 
img = Image.new('RGB', (WIDTH, int(WIDTH / 2))) 

# # Generate list of indices formed into complex numbers of the for x + yi
zoom = 1
x = torch.tensor(range(img.size[0]), device='cuda')

x = ((x - (0.75 * WIDTH)) / (WIDTH / 4) / zoom)

y = torch.tensor(range(img.size[1]), device='cuda')
y = ((y - (WIDTH / 4)) / (WIDTH / 4) / zoom)
y = y * 1j
c0 = (x.reshape(-1, 1) + y).T
print(c0)

pixels = mandelbrot(c0, img, 1000)
pixels = pixels.cpu().numpy().astype(np.uint8)
imageio.imsave('mandeltest.jpg', pixels)

# i = 0
# images = []
# zoom = 1
# while i < 2000:
#     print(i)
#     x = torch.tensor(range(img.size[0]), device='cuda')
#     x = ((x - (0.75 * WIDTH))  / (WIDTH / 4) / zoom) + 0.395
#     y = torch.tensor(range(img.size[1]), device='cuda')
#     y = ((y - (WIDTH / 4)) / (WIDTH / 4) / zoom) + 0.6045
#     y = y * 1j
#     c0 = (x.reshape(-1, 1) + y).T
#     pixels = mandelbrot(c0, img, 1000)
#     pixels = pixels.cpu().numpy().astype(np.uint8)
#     images.append(pixels)
#     i += 1
#     zoom += i/512
# imageio.mimsave('mandel.gif', images, duration = 0.1)