import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from torchvision import transforms
from PIL import Image
import numpy as np

from model import ResNet18Unet

checkpoint = 'unet-2/net_trained.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = ResNet18Unet().to(device)

net.load_state_dict(torch.load(checkpoint, map_location=device)["params"])
net.eval()

preprocess = transforms.Compose([
    transforms.ToTensor()
])

# Set up figure using gridspec
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05])

# Set font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

filenames = ['100.bmp', '200.bmp', '300.bmp', '400.bmp']

axes = [plt.subplot(gs[0, i]) for i in range(len(filenames))] + [plt.subplot(gs[1, i]) for i in range(len(filenames))] 
cax = plt.subplot(gs[1, -1])

axes[0].set_ylabel('Isochromatic Image', rotation=0, labelpad=20, fontsize=12, ha='right')
axes[len(filenames)].set_ylabel('Predicted Stress Map', rotation=0, labelpad=20, fontsize=12, ha='right')

for i, filename in enumerate(filenames):
    axes[i].set_title(f'{int(filename[:-4]) // 100}.00')

fig.text(0.01, 0.928, 'Compressive Force (kN)', ha='left', va='center', fontsize=12)

for i, filename in enumerate(filenames):
    # Load img
    img = Image.open('resources/ring/' + filename)

    # Preprocess img
    img_tensor = preprocess(img)

    # Predict stress map
    predict = net(img_tensor.unsqueeze(0).to(device)) * 255
    predict_img = predict.long().squeeze(0)
    print(np.max(predict_img.data.cpu().numpy().squeeze(0)))
    axes[i].imshow(img)

    # Replace below with maximum stress for set of images
    max_stress = 151
    res = axes[i + 4].imshow(predict_img.data.cpu().numpy().squeeze(0) / max_stress, cmap='gray', vmin=0, vmax=1)

    axes[i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axes[i + 4].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

# Add colorbar
cbar = plt.colorbar(res, cax=cax, shrink=0.7)
cbar.set_label('Relative Stress')

plt.tight_layout()
plt.savefig('results/ring/ring_ml_comparison.png', bbox_inches='tight')
plt.show()
