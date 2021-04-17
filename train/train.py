import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

import random
import matplotlib.pyplot as plt
sys.path.append('..')
from loader.loader import SplitedDataset, RerferDataset
from model.unet import UNet
from utils.loss import multi_task_loss
from utils.plot import plot_loss, plot_metric, plot_show, plot_channel_loss
from utils.inference import predict_one
from utils.evaluate import evaluate
from utils import metric
from utils.para_io import para_write
from model.siam_unet import SeqUnet

def train(args):
    torch.cuda.empty_cache()
    
    result_path = '../result/%s/'%args.save_path
    print(result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs('%simage'%result_path)
        os.makedirs('%scheckpoint'%result_path)

    print('using model: %s'%args.model)

    if args.model=='unet':
        model = UNet(in_channels=1, out_channels=4)
        loader_methode = SplitedDataset
        pred_type='unet'
    elif args.model=='unet_refer':
        model = SeqUnet(merge_mode='conca', in_ch=1, out_ch=4)
        loader_methode = RerferDataset    
        pred_type='unet_refer'    

    train_set = loader_methode(yaml_name=args.yaml, 
        root=args.dataset, 
        stake='')
    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True, 
        num_workers=args.num_workers)

    valid_set = loader_methode(yaml_name=args.yaml, 
        data_type='valid', 
        root=args.dataset, stake='')
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.eval_batchsize,
        shuffle=True,
        num_workers=args.num_workers) 

    test_set = loader_methode(yaml_name=args.yaml, 
        data_type='test', 
        root=args.dataset, 
        stake='')
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batchsize,
        shuffle=True,
        num_workers=args.num_workers)  

    # device = 'cuda:0'
    device = 'cuda:6' if torch.cuda.device_count()>1 else 'cuda:0'



    if args.multi_cuda:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids = [6, 4, 5])

    model = model.to(device)

    model.train()
    criterion_bse = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_list = []
    channel_loss_list = []
    train_metric_list = []
    valid_metric_list = []
    valid_metric_best = 0

    # imgs = valid_set.__getitem__(20)
    # img_show = torch.tensor(imgs[[0]]).to(device).float()[None, :]
    # print(img_show.shape)
    # plot_show(imgs, path='%simage/test.png'%(result_path))

    for epo in tqdm(range(1, args.epochs+1), ascii=True):
        epo_loss = []
        for idx, item in enumerate(train_loader):
            optimizer.zero_grad()
            # if
            x, y, pred = predict_one(model, item, device, True, pred_type=pred_type)
            # print(x.shape, y.shape, pred.shape)

            if args.loss_type=='mse':
                loss, channel_loss = multi_task_loss(pred, y, args.weight)
            elif args.loss_type=='bse':
                # channel loss is not used
                channel_loss = [0, 0, 0]
                # print(pred, y)
                loss = criterion_bse(pred, y)

            channel_loss_list.append(torch.tensor(channel_loss).cpu().detach().numpy().tolist())
            epo_loss.append(loss.data.item())
            loss.backward()
            optimizer.step()

        epo_loss_mean = np.array(epo_loss).mean()
        loss_list.append(epo_loss_mean)
        plot_loss(loss_list, '%simage/loss_%s.png'%(result_path, args.yaml))
        plot_channel_loss(channel_loss_list, 
                          '%sloss_channel_%s.png'%(result_path, args.yaml),
                           args.weight)

        # channel_loss_list
        np.save('%schannel_loss_list_%s.npy'%(result_path, args.yaml), np.array(channel_loss_list))
        
        # with torch.no_grad():
        #     pred = model(img_show.clone())
            
        #     imgs = np.concatenate([
        #         img_show[0].cpu().detach().numpy(),
        #         pred[0].cpu().detach().numpy()
        #         ])
        #     # print(imgs.shape)
        #     plot_show(imgs, path='%simage/%s_%d.png'%(result_path, args.yaml, epo))
        #loss
        if epo % 5 ==0:
            # torch.save(model, '%scheckpoint/%d.pt'%(result_path, epo))
            train_metric_list.append(evaluate(train_loader, 
                                    model, 
                                    metric.my_iou_pytorch, 
                                    device,
                                    eva_type=pred_type).mean())
                                    
            valid_metric = evaluate(valid_loader, 
                                    model, 
                                    metric.my_iou_pytorch, 
                                    device,
                                    eva_type=pred_type).mean()

            valid_metric_list.append(valid_metric)
            plot_metric(train_metric_list, valid_metric_list, '%smetric_%s.png'%(result_path, args.yaml))
            if valid_metric > valid_metric_best:
                valid_metric_best = valid_metric
                torch.save(model, '%scheckpoint/best_%s.pt'%(result_path, args.yaml))
            np.save('%sloss_%s.npy'%(result_path, args.yaml), np.array(loss_list))

    model = torch.load('../result/%s/checkpoint/best_%s.pt'%(args.save_path, args.yaml))
    return evaluate(test_loader, 
                    model, 
                    metric.my_iou_pytorch, 
                    device,
                    eva_type=pred_type)


if __name__ == '__main__':
    #python convlstm.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batchsize", default=24 if torch.cuda.device_count()>1 else 1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_batchsize", default=256, type=int)
    parser.add_argument("--multi_cuda", default=True, type=bool)
    parser.add_argument("--weight", default=[1, 1, 1, 1], type=list)
    # Drosophila_256_False
    parser.add_argument("--dataset", default='Drosophila_256_False', type=str,
            choices=['Drosophila_256_False'])
    parser.add_argument("--save_path", default='', type=str)
    parser.add_argument("--yaml", default='', type=str)
    #unet_detail my_unet family_unet R2U_Net AttU_Net R2AttU_Net NestedUNet
    parser.add_argument("--model", default='', type=str,
            choices=['unet', 'unet_refer'])
    # mse bse
    parser.add_argument("--loss_type", default='bse', type=str,
            choices=['bse', 'mse'])
    args = parser.parse_args()

    # model_list = [
    # 'my_unet',
    # ]

    # for ex_idx in range(5):

    #     for m in model_list:
    #         # print()
    #         args.model = m
    #         args.save_path = '%s_new_(%d)'%(m, ex_idx)

    #         result = []
    #         for i in range(4):
    #             args.yaml = 'para%d'%i
    #             print(args)
    #             r = train(args)
    #             print(r)
    #             result.append(r)
    #         result = np.array(result)
    #         print(result)
    #         print(result.mean(axis=0))
    #         print(result.mean(axis=1))
    #         print(result.mean())

    #         np.save('%sresult_%s.npy'%('../result/%s/'%args.save_path, args.yaml), np.array(result))

    #         para_write({'mean1':str(result.mean(axis=0)), 
    #                     'mean2':str(result.mean(axis=1)),
    #                     'mean':str(result.mean())},
    #                     '../result/%s/result.yaml'%args.save_path)