from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import torch
sys.path.append('..')

from utils.inference import predict_one
from utils import metric

def prob2msk(y):
    _, y_max = y.max(1)
    y_max = y_max[:, None]
    # print(y_max.shape)
    msks = torch.cat(((y_max==0), (y_max==1), (y_max==2), (y_max==3)), 1)*1
    # print(msks.shape)
    return msks

def evaluate(dataloader, model, metric_fun, device, threshold=0.5, eva_type='unet'):
    m = [[],[],[],[]]
    for item in dataloader:
        # if is_seq:
        x, y, pred = predict_one(model, item, device, False, pred_type=eva_type)
        # else:
        #     x, y, pred = predict_one(model, item, device, False, pred_type='unet')

        y = (y>threshold)*1
        # pred = pred>threshold
        pred = prob2msk(pred)


        for i in range(4):
            pred_in = pred[:,i]
            y_in =  y[:,i]
            m[i].append(metric_fun(pred_in, y_in).mean().data.item())
    m = np.array(m)
    return m.mean(axis=1)


if __name__ == '__main__':
    import torch
    from loader.loader import SplitedDataset
    import matplotlib.pyplot as plt

    device = 'cuda:0'
    model = torch.load('../result/test/checkpoint/best.pt')

    train_set = SplitedDataset(root='Drosophila_256_False', stake='', yaml_name='para1')
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True, 
        num_workers=1)

    e = evaluate(train_loader, model, metric.my_iou_pytorch, device='cuda:0')
    # print(prob2msk(e))