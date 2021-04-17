import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys, torch
import torch.nn as nn
sys.path.append('..')
from loader.loader import RerferDataset
from model.siam_unet import SeqUnet
from utils.evaluate import prob2msk
# pt_list = []

def msk2one(out):
    return out[:, 0]*1 + out[:, 1]*2 + out[:, 0]*3 + out[:, 1]*4

if __name__ == '__main__':

    test_set = RerferDataset(yaml_name='para3', data_type='test', root='Drosophila_256_False', stake='')
    test_loader = DataLoader(
        test_set,
        batch_size=4,
        shuffle=True,
        num_workers=1)  
    # device = 'cuda:0'
    device = 'cuda:6'
    model1 = torch.load('../pt/refer.pt', map_location=device)
    model2 = torch.load('../pt/aug.pt', map_location=device)
    model3= torch.load('../pt/attention.pt', map_location=device)
    # model1 = SeqUnet(merge_mode='conca', in_ch=1, out_ch=4).to(device)

    for item in test_loader:
        x = item[0][:,[0]].to(device, dtype=torch.float)
        y = item[0][:,1:].to(device, dtype=torch.float)

        x_ref = item[1].to(device, dtype=torch.float)
        out1 = model1(x, x_ref)
        out2 = model2(x)
        out3 = model3(x)

        one1 = msk2one(prob2msk(out1).cpu().detach().numpy())
        one2 = msk2one(prob2msk(out2).cpu().detach().numpy())
        one3 = msk2one(prob2msk(out3).cpu().detach().numpy())
        label = msk2one(prob2msk(y).cpu().detach().numpy())


        print(x.shape, y.shape, out1.shape)
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        out1 = out1.cpu().detach().numpy()
        out2 = out2.cpu().detach().numpy()

        fig, axes = plt.subplots(4, 5)
        ax = axes.flatten()

        for i in range(4):
            ax[i*5].imshow(x[i, 0])
            ax[i*5+1].imshow(label[i])

            ax[i*5+2].imshow(one1[i])

            ax[i*5+3].imshow(one3[i])
            ax[i*5+4].imshow(one2[i])

        for i in range(20):
            ax[i].set_axis_off()

        fig.tight_layout()
        plt.show()

        # break