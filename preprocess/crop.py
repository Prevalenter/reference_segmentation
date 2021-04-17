import os
from skimage.io import imread, imsave
import sys
from skimage import img_as_ubyte, util
sys.path.append('..')
# from loader.loader import DrosophilaDataset
if __name__ == '__main__':
    original_size = 1024
    out_size = 256
    overlape = False

    # datas = DrosophilaDataset()

    result_path = '../data/Drosophila_{}_{}/'.format(out_size, overlape)

    types = ['raw', 'membranes', 'mitochondria', 'synapses', 'cytoplasm']

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        for t in types:
            path = '%s%s'%(result_path, t)
            # print()
            os.makedirs(path)

    source_root = '../data/Drosophila/stack1/'
    image_names = os.listdir('%sraw'%(source_root))
    print(image_names)


    num_patch = original_size//out_size
    print(num_patch)

    for img_idx, image_name in enumerate(image_names):
        for t in types:
            img_path = '%s%s/%s'%(source_root, t, image_name)
            img = imread(img_path)
            save_path = '%s%s/%d%d'%(result_path, t, img_idx//10, img_idx%10)
            print(save_path, img.shape)
            idx = 0
            for i in range(num_patch):
                for j in range(num_patch):
                    idx += 1
                    patch = img.copy()[i*out_size: (i+1)*out_size, j*out_size: (j+1)*out_size]
                    # print(i, j, patch.shape)
                    if t=='raw':
                        imsave('%s_%d.png'%(save_path, idx), patch)
                    else:
                        imsave('%s_%d.png'%(save_path, idx), img_as_ubyte(patch>0),bits=1)


    # for img_idx in range(datas.len):
    #     for t in datas.types:
    #         # print(datas.__getitem__(i).keys(), t, i)
    #         img = datas.__getitem__(img_idx)[t]
    #         save_path = '%s%s/%d%d'%(result_path, t, img_idx//10, img_idx%10)
    #         print(save_path)
    #         idx = 0
    #         for i in range(num_patch):
    #             for j in range(num_patch):
    #                 idx += 1
    #                 patch = img.copy()[i*out_size: (i+1)*out_size, j*out_size: (j+1)*out_size]
    #                 # print(i, j, patch.shape)
    #                 imsave('%s_%d.png'%(save_path, idx), patch)