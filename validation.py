import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pytorch_ssim

from model import ResNet18Unet

checkpoint = 'unet-2/net_trained.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = ResNet18Unet().to(device)

net.load_state_dict(torch.load(checkpoint, map_location=device)["params"])
net.eval()

preprocess = transforms.Compose([
    transforms.ToTensor()
])

# Set font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['text.usetex'] = True


fringe_fn = os.listdir("dataset/Fringe_colors/")
stress_fn = os.listdir("dataset/Stress_maps/")

# Set up figure
fig, axes = plt.subplots(3, len(fringe_fn), figsize=(6.8, 3.5), dpi=300)

ssims = []

for i, filename in enumerate(fringe_fn):
    fringe_colour = Image.open("dataset/Fringe_colors/" + filename)
    axes[0, i].imshow(fringe_colour)
    axes[0, i].axis('off')

    stress_map = Image.open("dataset/Stress_maps/" + stress_fn[i])
    axes[1, i].imshow(stress_map, cmap='jet')
    axes[1, i].axis('off')

    # Produce stress map prediction with model
    img_tensor = preprocess(fringe_colour)
    prediction = net(img_tensor.unsqueeze(0).to(device))*255
    prediction_img = prediction.long().squeeze(0)
    axes[2, i].imshow(prediction_img.data.cpu().numpy().squeeze(0), cmap='jet')
    axes[2, i].axis('off')

    # Calculate SSIM
    ssim = pytorch_ssim.ssim(prediction/255,preprocess(stress_map).unsqueeze(0).to(device)).data.item()
    ssims.append(ssim)

# Labels for the first column
labels = ['Isochromatic \n Image', 'Reference', 'Prediction']
titles = ['Dragon', 'Ring', 'Ring', 'Ring', 'Ring', 'Disc']

# Add horizontal y labels to the first column
for j, label in enumerate(labels):
    ax = axes[j, 0]
    ax.axis('on')
    ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=12)
    ax.set_xticks([])
    ax.set_yticks([])

# Add titles to first row and ssim to last
for i, ssim in enumerate(ssims):
    axes[0, i].set_title(titles[i])

    ax_bottom = axes[2, i]
    ax_bottom.axis('on')
    ax_bottom.set_xlabel(format(ssim, '.4f'), labelpad=5)
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])

fig.text(0.092, 0.067, 'SSIM', ha='left', va='center')

plt.tight_layout()
plt.savefig('results/validation.png', bbox_inches='tight')
plt.show()
