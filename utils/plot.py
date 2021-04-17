import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
def plot_loss(loss_record, save_path):
    plt.clf()
    fig, axes = plt.subplots(1, 1)
    plt.plot(loss_record)
    # plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def plot_metric(train_rmse_list, valid_rmse_list, save_path):
    plt.clf()
    fig, axes = plt.subplots(1, 1)
    plt.plot(train_rmse_list, label = 'train metric')
    plt.plot(valid_rmse_list, label = 'valid metric')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.title('metric')
    plt.legend()
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def plot_show(imgs, path):
    plt.clf()
    fig, axes = plt.subplots(2, 3)
    ax = axes.flatten()

    for i in range(5):
        ax[i].imshow(imgs[i])
        ax[i].set_axis_off()
    fig.tight_layout()
    # plt.show()
    print(path)
    plt.savefig(path)
    plt.close('all')
    # del plt, axes
    # plt.show()

def plot_channel_loss(channel_loss, save_path, weight):
    plt.clf()
    labels = ['membr', 'mitoc', 'synap']
    loss = np.array(channel_loss)
    # print(loss.shape)
    for i in range(loss.shape[1]):
        plt.plot(loss[-1000:, i], label=labels[i])
    plt.legend()
    plt.grid(True)
    plt.title('Weight: %s'%str(weight))
    plt.savefig(save_path)
    plt.close('all')
    
if __name__ == '__main__':
    import sys
    import torch
    sys.path.append('..')
    from loader.loader import SplitedDataset
    from torch.utils.data import Dataset, DataLoader
    from utils.inference import predict_one
    # from utils.plot import 

    device = 'cuda:0'
    model = torch.load('../result/test/checkpoint/best.pt')

    train_set = SplitedDataset(root='Drosophila_256_False', stake='')
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True, 
        num_workers=1)

    for item in train_loader:
        x, y, pred = predict_one(model, item, device, False, pred_type='unet')
        print(x.shape, pred.shape)
        gt = np.concatenate([
            x[0].cpu().detach().numpy(),
            y[0].cpu().detach().numpy()
            ])

        prd = np.concatenate([
            x[0].cpu().detach().numpy(),
            pred[0].cpu().detach().numpy()>0.5
            ])
        plot_show(prd, path='prd.png')
        plot_show(gt, path='gt.png')
        break