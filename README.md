# ConvertSoundWithStarGAN

Converting audio using StarGAN and WaveRNN

## Train StarGAN
### 0. prepare dataset

prepare .wav file that you want to train with and put in folder like below:
```
<dataset_folder> ──┬── <category1> ─┬─ soundfile1_1.wav
                   │                ├─ soundfile1_2.wav
                   │                ...
                   ├── <category2> ─┬─ soundfile2_1.wav
                   │                ├─ soundfile2_2.wav
                   │                ...
                   ├── <categpry3> ─┬─ soundfile3_1.wav
                   │                ├─ soundfile3_2.wav
                   │                ...
                   ...
```
### 1. download codes
Clone this repository:
```
git clone https://github.com/HaruNari529/Convert-Sound-With-StarGAN.git
cd Convert-Sound-With-StarGAN/
```
Install python packages:
```
pip install torch==1.11.0 numpy==0.8.1 librosa==1.21.6
```
### 2. train StarGAN
```
python train_stargan.py \
    --epoch <enter the epoch in integer> \
    --datapath <enter paths for datasets e.g. /category1 /category2 ...> \
    --checkpoint <resume train or not, type in bool> \
    --checkpointpath <path to save your trained models> \
    --n_checkpoint <model save frequency per items>
```
## Train WaveRNN
### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/

### 1. download codes
Clone this repository:
```
git clone https://github.com/HaruNari529/Convert-Sound-With-StarGAN.git
cd Convert-Sound-With-StarGAN/
```
### 2. Preprocessing
```
python wavernn_preprocess.py \
        --dataset_dir <Path to the dataset dir (Location where the dataset is downloaded)>\
        --out_dir <Path to the output dir (Location where processed dataset will be written)>
```

The preprocessing code currently supports the following datasets:
- LJSpeech (en)

### 3. Training
```
python train_wavernn.py \
     --train_data_dir <Path to the dir containing the data to train the model> \
     --checkpoint_dir <Path to the dir where the training checkpoints will be saved> \
     --resume_checkpoint_path <If specified load checkpoint and resume training from that point>
```
## Convert audio
### 0. prepare models and audio files
prepare .wav file that you want to convert and model that you are using and put in folder like below:
```
<generate folder> ──┬── <wavfolder> ─── soundfile.wav
                    ├── <generatefolder> 
                    └── <checkpointfolder> ─┬─ net_G.pth
                                            ├─ net_D.pth
                                            └─ wavernn.pth
```
### 1. Clone repository
```
git clone https://github.com/HaruNari529/Convert-Sound-With-StarGAN.git
cd Convert-Sound-With-StarGAN/
```
### 2. Generate melspectrogram
```
python generate_stargan.py \
     --label <index number that you want to convert image to in integer> \
     --wavpath <paths for wav file> \
     --checkpointpath <path for whre models saved> \
     --generatepath <path for putting generated image>
```
### 3. generate .wav file
```
python generate.py \
    --checkpoint_path <Path to the checkpoint to use to instantiate the model> \
    --eval_data_dir <Path to the generate folder> \ 
    --out_dir <Path to the generate folder>
```
## Acknowledgements

The code in this repository is based on the code in the following repositories
1. [clovaai/stargan-v2](https://github.com/clovaai/stargan-v2)
2. [anandaswarup/WaveRnn](https://github.com/anandaswarup/waveRNN)
