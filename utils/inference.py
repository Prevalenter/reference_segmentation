import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')

from model.unet import UNet

def predict_one(model, item, device, is_train, pred_type='unet'):
    def pred():
        if pred_type == 'unet':
            x = item[:,[0]].to(device, dtype=torch.float)
            y = item[:,1:].to(device, dtype=torch.float)
            # print()
            pred = model(x)
        elif pred_type=='unet_refer':
            x = item[0][:,[0]].to(device, dtype=torch.float)
            y = item[0][:,1:].to(device, dtype=torch.float)

            x_ref = item[1].to(device, dtype=torch.float)
            pred = model(x, x_ref)  
        elif pred_type=='vedio':   
            # item = item.to(device, dtype=torch.float)
            x = item[:, :, [0]].to(device, dtype=torch.float)
            y = item[:, :, 1:].to(device, dtype=torch.float)
            x_ref = item[:, 0].to(device, dtype=torch.float)

            pre_mask = torch.zeros(y[:, 0].shape).to(device, dtype=torch.float)

            pred_sum = []
            for i in range(20):
                x_in = torch.cat((x[:, i], pre_mask), axis=1)
                pred = model(x_in, x_ref)
                pred_sum.append(pred[:, None])
                #update pre_msk
                pre_mask = pred
            pred = torch.cat(pred_sum, axis=1)
        return x, y, pred

    if is_train: 
        model.train()
        return pred()

    else:
        model.eval()
        with torch.no_grad():
            return pred()

def predict_one_with_ref(model, item, device, is_train):
    def pred():
        x = item[0][:,[0]].to(device, dtype=torch.float)
        y = item[0][:,1:].to(device, dtype=torch.float)

        x_ref = item[1].to(device, dtype=torch.float)
        pred = model(x, x_ref)         
        return x, y, pred

    if is_train: 
        model.train()
        return pred()

    else:
        model.eval()
        with torch.no_grad():
            return pred()

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from loader.loader import SplitedDataset, VedioDataset
    import matplotlib.pyplot as plt
    from utils.evaluate import prob2msk
    from model.siam_unet import VedioUnet

    device = 'cuda:0'

    model = VedioUnet('conca', 5, 4).to(device, dtype=torch.float)

    train_set = VedioDataset(root='Drosophila_256_False', data_type='train', stake='', yaml_name='para0')
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True, 
        num_workers=1)  

    for item in train_loader:
        x, y, pred = predict_one(model, item, device, False, pred_type='vedio')
        print(x.shape, y.shape, pred.shape)

    # for item in train_loader:
    #     # print(item.shape)
    #     item = item.to(device, dtype=torch.float)
    #     x = item[:, :, [0]]
    #     y = item[:, :, 1:]

    #     x_ref = item[:, 0]
    #     pre_mask = torch.zeros(y[:, 0].shape).to(device, dtype=torch.float)
    #     print(x.shape, y.shape, x_ref.shape, pre_mask.shape)

    #     pred_sum = []
    #     for i in range(20):
    #         x_in = torch.cat((x[:, i], pre_mask), axis=1)
    #     # with torch.no_grad():
    #         pred = model(x_in, x_ref)
    #         pred_sum.append(pred[:, None])
    #         #update pre_msk
    #         pre_mask = pred

    #     pred_sum = torch.cat(pred_sum, axis=1)
    #     print(pred_sum.shape)





    #     ref_imgs = torch.zeros((1, 4, 256, 256)).to(device, dtype=torch.float)
    #     for i in range(20):
    #         x = item[:, :, i].to(device, dtype=torch.float)
    #         y = item[0][:,1:].to(device, dtype=torch.float)

    #         pred = model(x, ref_imgs)
    #         print('pred', pred.shape)

    # train_set = SplitedDataset(root='Drosophila_256_False', stake='', yaml_name='para1')
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=1,
    #     shuffle=True, 
    #     num_workers=1)

    # for item in train_loader:
    #     x, y, pred = predict_one(model, item, device, True, pred_type='unet')

    #     plt.clf()
    #     fig, axes = plt.subplots(2, 4)
    #     ax = axes.flatten()
    #     pred_msk = prob2msk(pred)

    #     x = x.cpu().detach().numpy()[0]
    #     pred_msk = pred_msk.cpu().detach().numpy()[0]

    #     print(x.shape, pred_msk.shape)
    #     ax[0].imshow(x[0])
    #     for i in range(1,4):
    #         ax[i].imshow(pred_msk[i-1])
    #     plt.show()

    #     break