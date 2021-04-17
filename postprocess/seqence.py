import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys, torch
import torch.nn as nn
import numpy as np
sys.path.append('..')
# from loader.loader import RerferDataset
from model.siam_unet import SeqUnet
from utils.evaluate import prob2msk
from skimage.io import imread, imsave
# pt_list = []

def msk2one(out):
    return out[:, 0]*1 + out[:, 1]*2 + out[:, 0]*3 + out[:, 1]*4

class TestDataset(Dataset):
    def __init__(self):
        self.root = '../data/raw'
        # print(os.listdir('%s'%self.root))

        self.imgs = {str(i):[] for i in range(1, 17)}
        # print(self.imgs)
        for i in os.listdir('%s'%self.root):
            time, position = i[:-4].split('_')
            # print(time, position)
            self.imgs[position].append(i)
        # print(self.imgs)

        
        self.keys = list(self.imgs.keys())
        self.len = len(self.keys)
        print('len of set is: ', self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        imgs = []
        for t in range(20):
            img_path = '%s/%s'%(self.root, self.imgs[self.keys[idx]][t])
            # print(img_path)
            imgs.append(imread(img_path)[None])
        imgs = np.concatenate(imgs, axis=0)[:, None]
        return imgs/255

if __name__ == '__main__':
    device = 'cuda:6'
    test_set = TestDataset()
    # print(test_set.__getitem__(7).shape)
    x = torch.tensor(test_set.__getitem__(9)).to(device, dtype=torch.float)

    model = torch.load('../pt/aug.pt', map_location=device)
    pred = model(x)
    one = msk2one(prob2msk(pred).cpu().detach().numpy())

    print(x.shape, pred.shape)

    x = x.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    fig, axes = plt.subplots(2, 10)
    ax = axes.flatten()
    for i in range(10):
        ax[i].imshow(x[i*2, 0])
        ax[10+i].imshow(one[i*2])

    for i in range(20):
        ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()