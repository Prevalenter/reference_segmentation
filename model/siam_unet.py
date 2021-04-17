from collections import OrderedDict

import torch
import torch.nn as nn

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# merge the feature via add/conca + conv
class MergeConv(nn.Module):
    def __init__(self, in_ch, out_ch, merge_mode):
        #merge_mode : conca or add
        super(MergeConv, self).__init__()
        if merge_mode=='conca': in_ch = 2*in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.merge_mode = merge_mode

    def forward(self, x, x_refer):
        if self.merge_mode=='conca': out = torch.cat((x, x_refer), dim=1)
        elif self.merge_mode=='add': out = x + x_refer
        return self.conv(out)

class SeqUnet(nn.Module):
    def __init__(self, merge_mode, in_ch=3, out_ch=1, in_ch_refer=5):
        super(SeqUnet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_refer = conv_block(in_ch_refer, filters[0])

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Softmax2d()

        if merge_mode in ['add', 'conca']:
            print('using merge mode: %s'%merge_mode)
            self.merge = MergeConv(filters[4], filters[4], merge_mode)

    def forward(self, x, x_refer=None):
        def encoder_prop(x, msk_input):
            # encoder
            if msk_input: e1 = self.conv1_refer(x)
            else: e1 = self.Conv1(x)
            e2 = self.Maxpool1(e1)
            e2 = self.Conv2(e2)
            e3 = self.Maxpool2(e2)
            e3 = self.Conv3(e3)
            e4 = self.Maxpool3(e3)
            e4 = self.Conv4(e4)
            e5 = self.Maxpool4(e4)
            e5 = self.Conv5(e5)
            return e1, e2, e3, e4, e5

        e1, e2, e3, e4, e5 = encoder_prop(x, False)

        # print(e5.shape)
        if x_refer is not None:
            e5_refer = encoder_prop(x_refer, True)[-1]
            e5 = self.merge(e5, e5_refer)
        # print(e5_refer.shape)

        #bottleneck
        d5 = self.Up5(e5)

        # decoder
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)

        out = self.active(out)

        return out


if __name__ == '__main__':
    # model_R2U_Net = R2U_Net(img_ch=1, output_ch=4)
    device = 'cuda:0'

    model_unet = VedioUnet('conca', 5, 4).to(device)
    x = torch.randn((2,5,256,256)).to(device)
    x_ref = torch.randn((2,5,256,256)).to(device)
    y = model_unet(x, x_ref)
    print(x.shape, y.shape)
    print(x.dtype, y.dtype)
    print(y.sum(1))

