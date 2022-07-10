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
## train WaveRNN
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
