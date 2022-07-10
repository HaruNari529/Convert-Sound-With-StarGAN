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
