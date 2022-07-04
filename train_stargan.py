import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import glob
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

def labelcri(size,l1,l2):
    r1 = np.ones(size)*l1
    r2 = np.ones(size)*l2
    r = np.array([[r1,r2]])
    r = torch.from_numpy(r).to(device)
    return r.to(device)
 
def convdataset(filepaths):
    datas = []
    for i in range(len(filepaths)):
        datas.append([])
        for fipath in glob.glob(filepaths[i]+'/*.wav'):
            wave,rate = librosa.load(fipath,sr=44100)
            for l in range(int(len(wave)/12750)):
                mel = librosa.feature.melspectrogram(y=wave[(l*12750):((l+1)*12750)], sr=rate, n_mels=256, hop_length=int(12750/255))
                mel = librosa.power_to_db(mel,ref=np.max)
                datas[i].append(np.array([mel]))
    mins = []
    for i in range(len(datas)):
        mins.append(len(datas[i]))
    minmin = min(mins) -1
    for i in range(len(datas)):
        datas[i] = datas[i][:minmin]
    return datas
  
def train(args):##epoch, datapath, checkpoint=True, checkpointpath, n_checkpoint=500
    """
    datasets : (labels, (batches, x, y, channels))
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr_d = 0.000014
    lr_g = 0.0002
    
    if args.checkpoint:
      net_G = Generator(2, 1, 3).to(device)
      net_G.load_state_dict(torch.load(args.checkpointpath+'/net_G.pth',map_location=device))
      net_G.eval()
      net_D = torch.load(args.checkpointpath+'/net_D.pth',map_location=device)
      net_D.eval()
    else:
      net_G = Generator(2, 1, 3).to(device)
      net_D = Discriminator(1,2).to(device)
      
    optimizerG = torch.optim.Adam(net_G.parameters(),lr=lr_g,betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(net_D.parameters(), lr = lr_d, betas=(0.5, 0.999))

    criterion_GAN = nn.BCELoss()
    criterion_cycle = nn.L1Loss()
    datasets = convdataset(args.datapath)
    print(len(datasets[0]))
    net_G.train()
    net_D.train()
    
    
    items = 0
    loss_train_D_epoch = 0
    loss_train_G_epoch = 0
    loss_train_cycle_epoch = 0
    lossesses = []
    for e in range(args.epoch):
        for i in range(len(datasets[0])):
            iimg = random.randint(1,len(datasets))
            ict = random.randint(1,len(datasets))
            while ict == iimg:
                ict = random.randint(1,len(datasets))

            real = datasets[iimg-1][i]
            fake = net_G(img4net(label_img(real,ict)).to(device))
            recy = net_G(img4net(label_img(fake.to('cpu').detach().numpy().copy()[0],iimg)).to(device))

            optimizerD.zero_grad()
            batch_size = 1
            label = labelcri((30,30),1,iimg)
            output = net_D(img4net(real).to(device))
            output = output.to(torch.float64)
            errD_real = criterion_GAN(output, label)
            errD_real.backward()

            label = labelcri((30,30),0,ict)
            output = net_D(fake.detach())
            output = output.to(torch.float64)
            errD_fake = criterion_GAN(output, label)
            errD_fake.backward()

            loss_train_D_epoch += errD_real.item() + errD_fake.item()

            optimizerD.step()

            
            optimizerG.zero_grad()

            label = labelcri((30,30),1,iimg)
            output = net_D(fake)
            output = output.to(torch.float64)

            errG = criterion_GAN(output, label)

            loss_train_G_epoch += errG.item()


            loss_cycle = criterion_cycle(recy.to(device), img4net(real).to(device))
            loss_train_cycle_epoch += loss_cycle.item()

            errG += loss_cycle
            errG.backward()

            optimizerG.step()
            lossesses.append([loss_train_cycle_epoch,loss_train_D_epoch,loss_train_G_epoch])
            items += 1
            print("\r", 'epoch:'+str(e)+'/'+str(args.epoch)+', item:'+str(i)+'/'+str(len(datasets[0])), end="")
            if items % args.n_checkpoint == (args.n_checkpoint - 1):
                torch.save(net_G.state_dict(), args.checkpointpath+'/net_G.pth')
                torch.save(net_D, args.checkpointpath+'/net_D.pth')
                ndata = np.loadtxt(checkpointpath+"/data.csv",delimiter=',').tolist()
                lossessess = []
                for i23 in ndata:
                    lossessess.append(i23)
                for i23 in lossesses:
                    lossessess.append(i23)
                filesaved = False
                fi = 0
                while not filesaved:
                    if os.path.exists(checkpointpath+"/data.csv"):
                        np.savetxt(checkpointpath+"/data.csv", lossesses, delimiter=',')
                        filesaved = True
                        lossesses = []
                    else:
                        fi +=1
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #epoch, net_G, net_D, optimizerG, optimizerD, criterion_GAN, criterion_cycle, datasets, datapath, checkpoint=True, checkpointpath, n_checkpoint=500
    parser.add_argument('--epoch', type=int, default=100, help='epoch for train')
    parser.add_argument('--datapath', type=str, help='path for dataset folder')
    parser.add_argument('--checkpoint', type=bool, default=True, help='resume train from pretrained model or not')
    parser.add_argument('--checkpointpath', type=str, help='path for checkpoint folder')
    parser.add_argument('--n_checkpoint', type=int, default=500, help='model save frequency per item')
    
    args = parser.parse_args()
    train(args)
