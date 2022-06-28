import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.BatchNorm2d(dim),
                       nn.LeakyReLU(0.2),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.BatchNorm2d(dim)
                       ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):

    def __init__(self, in_channel, out_channel, n_res):
        super(Generator, self).__init__()
        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel,64,kernel_size=7,stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResNetBlock(256) for _ in range(n_res)]
        self.res_block = nn.Sequential(
            *res_blocks
        )
        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64,out_channel,kernel_size=7,stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        x = self.decode_block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512,out_channel,kernel_size=4,stride=1,padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        return x