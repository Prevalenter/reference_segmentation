from skimage.io import imread, imsave
from torch.utils.data import Dataset, DataLoader
import os, sys
import numpy as np
import torch

sys.path.append('..')
from utils import para_io
os.environ['KMP_DUPLICATE_LIB_OK']='True'
root = '../data/'

class RerferDataset(Dataset):
    def __init__(self,  yaml_name, data_type='train', root='Drosophila_256_False', stake=''):
        self.root = '../data/%s/%s'%(root, stake)
        yaml_path = '../data/%s/%s.yaml'%(root, yaml_name)
        print(yaml_path)
        train_positions = para_io.para_load(yaml_path)[data_type]
        self.image_sum =\
             [i for i in os.listdir('%sraw'%self.root) if int(i[:-4].split('_')[1]) in train_positions]

        # filter the first fram
        self.refer_frams = {}
        self.image_list = []
        for idx, i in enumerate(self.image_sum):
            seq_idx, positon = i[:-4].split('_')
            # print(idx, i, positon, seq_idx)
            if seq_idx == '00':
                self.refer_frams[positon] = i
            else: self.image_list.append(i)

        self.types = ['raw', 'membranes', 'mitochondria', 'synapses', 'cytoplasm']
        self.len = len(self.image_list)
        print('len of set is: ', self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        _, positon = self.image_list[idx][:-4].split('_')

        imgs = []
        for t in self.types:
            img_path = '%s/%s/%s'%(self.root, t, self.image_list[idx])
            # print(img_path)
            imgs.append(imread(img_path)[None])
        imgs = np.concatenate(imgs, axis=0)

        imgs_refer = []
        for t in self.types:
            img_path = '%s/%s/%s'%(self.root, t, self.refer_frams[positon])
            # print(img_path)
            imgs_refer.append(imread(img_path)[None])
        imgs_refer = np.concatenate(imgs_refer, axis=0)        

        return imgs/255, imgs_refer/255

class SplitedDataset(Dataset):
    def __init__(self,  yaml_name, data_type='train', root='Drosophila_256_False', stake=''):
        self.root = '../data/%s/%s'%(root, stake)
        yaml_path = '../data/%s/%s.yaml'%(root, yaml_name)
        print(yaml_path)
        train_positions = para_io.para_load(yaml_path)[data_type]
        print(train_positions)
        self.image_list =\
             [i for i in os.listdir('%sraw'%self.root) if int(i[:-4].split('_')[1]) in train_positions]

        self.types = ['raw', 'membranes', 'mitochondria', 'synapses', 'cytoplasm']
        self.len = len(self.image_list)
        print('len of set is: ', self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        imgs = []
        for t in self.types:
            img_path = '%s/%s/%s'%(self.root, t, self.image_list[idx])
            # print(img_path)
            imgs.append(imread(img_path)[None])
        imgs = np.concatenate(imgs, axis=0)
        return imgs/255

class DrosophilaDataset(Dataset):
    def __init__(self, root='Drosophila_256_False', stake='stack1/'):
        self.root = '../data/%s/%s'%(root, stake)
        print(self.root)
        self.image_list = os.listdir('%sraw'%self.root)

        self.types = ['raw', 'membranes', 'mitochondria', 'synapses', 'cytoplasm']
        self.len = len(self.image_list)
        print('len of set is: ', self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        imgs = []
        print(self.image_list[idx])
        for t in self.types:
            img_path = '%s/%s/%s'%(self.root, t, self.image_list[idx])
            # print(img_path)
            imgs.append(imread(img_path)[None])
        imgs = np.concatenate(imgs, axis=0)
        return imgs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_set = RerferDataset(root='Drosophila_256_False', 
                data_type='valid', stake='', yaml_name='para0')
    train_loader = DataLoader(
        train_set,
        batch_size=3,
        shuffle=True,
        num_workers=4)

    for item in train_loader:
        print(item[0].shape, item[1].shape)



