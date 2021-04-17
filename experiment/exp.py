import sys
import argparse
import numpy as np
sys.path.append('..')

from train.train import train
from utils.para_io import para_write
def k_flod_wrap(args):
    result = []
    for i in range(4):
        args.yaml = 'para%d'%i
        print(args)
        r = train(args)
        print(r)
        result.append(r)
    result = np.array(result)
    return result

def _exp(args):
    if args.save_path=='':
        args.save_path = args.model

    result = k_flod_wrap(args)

    print(result)
    print(result.mean(axis=0))
    print(result.mean(axis=1))
    print(result.mean())

    np.save('%sresult_%s.npy'%('../result/%s/'%args.save_path, args.yaml), np.array(result))

    para_write({'mean1':str(result.mean(axis=0)), 
                'mean2':str(result.mean(axis=1)),
                'mean':str(result.mean())},
                '../result/%s/result.yaml'%args.save_path)

def _exps(args, exp_num=5):
    for i in range(exp_num):
        args.save_path = '%s(%d)'%(args.model, i)
        _exp(args)

def train_unet(args):
    args.model = 'unet'
    _exp(args)

def train_unet_refer(args):
    args.model = 'unet_refer'
    _exp(args)

def train_unets(args):
    args.model = 'unet'
    _exps(args)

def train_unet_refers(args):
    # unet_refer is more unstable
    # so train more epoch
    args.epochs = 150
    args.model = 'unet_refer'
    _exps(args)

if __name__ == '__main__':
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batchsize", default=24 if torch.cuda.device_count()>1 else 1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_batchsize", default=256, type=int)
    parser.add_argument("--multi_cuda", default=True, type=bool)
    parser.add_argument("--weight", default=[1, 1, 1, 1], type=list)
    parser.add_argument("--dataset", default='Drosophila_256_False', type=str,
            choices=['Drosophila_256_False'])
    parser.add_argument("--save_path", default='', type=str)
    parser.add_argument("--yaml", default='', type=str)
    parser.add_argument("--model", default='', type=str,
            choices=['unet', 'unet_refer'])
    # mse bse
    parser.add_argument("--loss_type", default='bse', type=str,
            choices=['bse', 'mse'])
    args = parser.parse_args()

    # train_unets(args)
    train_unet_refers(args)


    # result = k_flod_wrap(args)

    # print(result)
    # print(result.mean(axis=0))
    # print(result.mean(axis=1))
    # print(result.mean())

    # np.save('%sresult_%s.npy'%('../result/%s/'%args.save_path, args.yaml), np.array(result))

    # para_write({'mean1':str(result.mean(axis=0)), 
    #             'mean2':str(result.mean(axis=1)),
    #             'mean':str(result.mean())},
    #             '../result/%s/result.yaml'%args.save_path)



