from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte, util
if __name__ == '__main__':
    root = '../data/Drosophila/stack1/'
    classs = ['membranes', 'mitochondria', 'synapses']

    img_paths = os.listdir('%sraw'%(root))

    # fig, axes = plt.subplots(2, 2)
    # ax = axes.flatten()

    # img_path = img_paths[0]
    for img_path in img_paths:
        imgs = []
        for idx, c in enumerate(classs):
            path = '%s%s/%s'%(root, c, img_path)
            print(idx, path)
            imgs.append((imread(path)>0))
            # ax[idx].imshow(imgs[idx])

        cytoplasm = np.ones(imgs[0].shape)
        for i in imgs:
            cytoplasm -= i
        cytoplasm = cytoplasm

        imsave('%s%s/%s'%(root, 'cytoplasm', img_path), img_as_ubyte(cytoplasm>0),bits=1)
    # imsave('bw_skimage.png',util.img_as_uint(check),plugin='pil',optimize=True,bits=1)

    # ax[3].imshow(cytoplasm>0)
    # for i in range(4):
    #     ax[i].set_axis_off()
    # plt.show()
    # plt.imshow()
