
from skimage.io import imread, imsave
from torch.utils.data import Dataset, DataLoader
import os, sys
import numpy as np
import torch
from skimage.transform import rotate
sys.path.append('..')
# from loader.loader import KflodDataset

def aug(img):
    imgs = [img, img[:, ::-1]]
    imgs_out = []

    for i in range(len(imgs)):
        imgs_out.append(imgs[i])
        for angele in [90,180,270]:
            rotated = rotate(imgs[i], angele)*255
            # print(rotated.shape)
            imgs_out.append(rotated)
    # imgs = np.concatenate(imgs, axis=0)
    # print(imgs.shape)      
    return imgs_out

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    source_root = '../data/Drosophila_256_False/'
    target_root = '../data/Drosophila_augmentation/'

    for fold in os.listdir(source_root):

        if fold.split('.')[-1] == 'yaml': continue
        source_fold = '%s%s'%(source_root, fold)
        target_fold = '%s%s'%(target_root, fold)

        if not os.path.exists(target_fold):
            os.makedirs(target_fold)

        # print(target_root)
        for f in os.listdir(source_fold):
            img_path = '%s/%s'%(source_fold, f)
            img = imread(img_path)

            # ax[0].imshow(img)
            imgs = aug(img)

            for i in range(len(imgs)):
                img_save_path = '%s/%s_%d.png'%(target_fold, f.split('.')[0], i)
                print(img_save_path)
                imsave(img_save_path, imgs[i].astype('uint8'))
            print(img_path)

            # fig, axes = plt.subplots(2, 4)
            # ax = axes.flatten()
            # for i in range(len(imgs)):
            #     ax[i].imshow(imgs[i])
            # for i in range(8): ax[i].set_axis_off()
            # fig.tight_layout()
            # plt.show()
            # break