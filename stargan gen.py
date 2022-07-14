import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from models.stargan import Generator, Discriminator
import argparse

def label_img(img, label):
    newimg = []
    for i in range(img.shape[0]):
        newimg.append(img[i])
    newimg.append((np.ones((img.shape[1],img.shape[2]))*label).astype(img.dtype))
    newimg = np.asarray(newimg)
    return newimg


def img4net(img):
    img = np.array([img])
    img = torch.from_numpy(img)
    return img

def convdataset(wavpath):
    datas = []
    wave,rate = librosa.load(wavpath,sr=44100)
    for l in range(int(len(wave)/12750)):
        mel = librosa.feature.melspectrogram(y=wave[(l*12750):((l+1)*12750)], sr=rate, n_mels=256, hop_length=int(12750/255))
        mel = librosa.power_to_db(mel,ref=np.max)
        datas.append(np.array([mel]))
    
    return datas

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_G = Generator(2, 1, 3).to(device)
    net_G.load_state_dict(torch.load(args.checkpointpath+'/net_G.pth',map_location=device))
    net_G.eval()
    net_D = torch.load(args.checkpointpath+'/net_D.pth',map_location=device)
    net_D.eval()

    net_G.eval()
    net_D.eval()

    datas = convdataset(args.wavpath)
    genimg = []
    for i in range(len(datas)):
        genimg.append(net_G(img4net(label_img(datas[i],args.label)).to(device)).to('cpu').detach().numpy().copy()[0])
    newimg = []
    
    for e in range(256):
        newimg.append([])
        for i in range(len(datas)):
            newimg.append([])
            for l in range(256):
                newimg[e].append(genimg[i][e][l])
    np.save('mels', np.asarray(newimg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #epoch, net_G, net_D, optimizerG, optimizerD, criterion_GAN, criterion_cycle, datasets, datapath, checkpoint=True, checkpointpath, n_checkpoint=500
    parser.add_argument('--label', type=int, default=100, help='index number that you want to convert image to')
    parser.add_argument('--wavpath', type=list, help='paths for wav file')
    parser.add_argument('--checkpointpath', type=str, help='path for model')
    
    args = parser.parse_args()
    generate(args)