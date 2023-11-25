import torch
from torch import nn, optim
<<<<<<< HEAD
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
=======
>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import pytorch_ssim

from model import ResNet18Unet
print(torch.version.cuda)
time_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder="dataset/Fringe_colors"
target_folder="dataset/Stress_maps"
<<<<<<< HEAD

# epoch_lr=[(10,0.0001,1),(10,0.00001,5)]
epoch_lr=[(10,0.0001,1)]

=======

epoch_lr=[(10,0.0001,1),(2,0.00001,5)]
>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f
batch_size = 128

checkpoint = 'unet-2/net.pth'
model_checkpoint = 'unet-2/net19.pth'


fringe_files_list = ['Img_' + str(i) +'.bmp' for i in range(1,100001,10)]
target_files_list = ['Target_' + str(i) +'.bmp' for i in range(1,100001,10)]
<<<<<<< HEAD
fringe_files_list1 = ['Img_' + str(i) +'.bmp' for i in range(3,100001,500)]
target_files_list1 = ['Target_' + str(i) +'.bmp' for i in range(3,100001,500)]
=======
fringe_files_list1 = ['Img_' + str(i) +'.bmp' for i in range(3,100001,50)]
target_files_list1 = ['Target_' + str(i) +'.bmp' for i in range(3,100001,50)]

>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f
fringe_files=[os.path.join(data_folder,i) for i in fringe_files_list]
target_files=[os.path.join(target_folder,i) for i in target_files_list]
fringe_files1=[os.path.join(data_folder,i) for i in fringe_files_list1]
target_files1=[os.path.join(target_folder,i) for i in target_files_list1]

train_fringe_files=fringe_files
train_target_files=target_files
test_fringe_files=fringe_files1
test_target_files=target_files1

<<<<<<< HEAD
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )
])
=======
>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f

def default_loader(path):
    img_pil = Image.open(path)
    # img_pil = img_pil.resize((224,224))
    img_tensor = transforms.ToTensor()(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = train_fringe_files
        self.target = train_target_files
        self.loader = loader

    def __getitem__(self, index):
        fn1 = self.images[index]
        img = self.loader(fn1)
        fn2 = self.target[index]
        target = self.loader(fn2)
        return img,target

    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = test_fringe_files
        self.target = test_target_files
        self.loader = loader

    def __getitem__(self, index):
        fn1 = self.images[index]
        img = self.loader(fn1)
        fn2 = self.target[index]
        target = self.loader(fn2)
        return img, target

    def __len__(self):
        return len(self.images)

def train():
    net=ResNet18Unet().to(device)
    # load model params
    net.load_state_dict(torch.load(model_checkpoint)["params"])
    print('successful')

    for params in net.parameters():
        nn.init.normal_(params, mean=0, std=0.01)

    train_data = trainset()
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = testset()
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
<<<<<<< HEAD

    print('Training set size: ', len(trainloader))
    print('-------------')
    print('Testing set size: ', len(testloader))
    print('-------------')

    #loss function
=======
    
    # Mean Squared Error Loss
>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f
    criteron = nn.MSELoss()
    best_accuracy = 0

    for n, (num_epochs, lr, ld) in enumerate(epoch_lr):
        print(f'Epoch set: {n}')
        print(f'num_epochs: {num_epochs}')
        
        optimizer = optim.Adam(
            net.parameters(), lr=lr, weight_decay=0,
        )
        for epoch in range(num_epochs):
<<<<<<< HEAD
            print(f'Epoch: {epoch}')
            # if n == 0:
            #     ld = 1
            # else:
            #     ld = 1 + epoch*0.2
            net.train()
            epoch_loss = 0.0

            for i, (img, target) in enumerate(trainloader):
                out = net(img.to(device))
                print(i)
                # print(out.shape)
                # print(target.shape)
                ssim_loss = 1 - pytorch_ssim.ssim(out, target.to(device).float())
                out = out.squeeze(1)
                target = target.to(device).float().squeeze(1)
                # if n == 1 or n == 0:
                #     physics_loss = 0.0
                #     for j in range(out.shape[0]):
                #         batch_t = out[j]
                #         print(len(batch_t))
                #         for p in range(epoch+1,batch_t.shape[0]-1,10):
                #             for q in range(epoch+1,batch_t.shape[1]-1,10):
                #                 batch_pq_ave = (batch_t[p-1][q]+batch_t[p+1][q]+batch_t[p][q-1]+batch_t[p][q+1])/4
                #                 physics_loss += (batch_t[p][q]-batch_pq_ave)*(batch_t[p][q]-batch_pq_ave)
                #     loss = ssim_loss + 100 * criteron(out, target) + 0.0001 * physics_loss
                #     # print(loss, ssim_loss, physics_loss, criteron(out, target))

                # I removed ld * as we use ld = 1
                loss = ssim_loss + criteron(out, target)
                # loss = 1 - pytorch_ssim.ssim(out,target.to(device).float()) + ld * criteron(out.squeeze(1),target.to(device).float().squeeze(1))
                # print(out.squeeze(1))
                # print(target.to(device).float().squeeze(1))
=======
            print('-'*100)
            print(f'Epoch: {epoch}')
            

            # Set lambda for loss calculation
            if n == 0:
                ld = 1
            else:
                ld = 1 + epoch*0.2

            # Set net to training mode
            net.train()

            # Iterate through the training set, using tqdm for progress bar
            for i, (img, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
                # print(f'Epoch: {epoch}, Batch: {i}')

                # Forward Pass
                out = net(img.to(device))

                # Compute loss
                loss = 1 - pytorch_ssim.ssim(out,target.to(device).float()) + ld * criteron(out.squeeze(1),target.to(device).float().squeeze(1))

                # Backwards pass
>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f
                optimizer.zero_grad()
                loss.backward()

                # Update model parameters
                optimizer.step()

            # Validation
            with torch.no_grad():
                net.eval()
                test_accuracy = 0.0
                batch = 0
                for i, (img, target) in enumerate(testloader):
                    out = net(img.to(device))
                    loss = pytorch_ssim.ssim(out, target.to(device).float())
                    batch += 1
                    test_accuracy += loss.item()
                # print loss
                print("test_accuracy:{}".format(test_accuracy / batch))
                time_end = time.time()
                print('totally cost', time_end - time_start)

            if test_accuracy / batch > best_accuracy:
                best_accuracy = test_accuracy / batch
                torch.save(
                    {"params":net.state_dict(), "accuracy":test_accuracy}, checkpoint
                )
                print("model save")

if __name__ == "__main__":
    train()
