"""Configuration settings"""


class Config:
    dataset = "ljspeech"

    # Audio processing parameters
    sampling_rate = 44100
    max_db = 100
    ref_db = 20

    n_fft = 2048
    win_length = None
    hop_length = int(12750/255)  

    num_mels = 256
    fmin = 50

    num_bits = 10  # Bit depth of the signal

    # Model parameters
    conditioning_rnn_size = 128
    audio_embedding_dim = 256
    rnn_size = 896
    fc_size = 1024

    # Training
    batch_size = 16
    num_steps = 200000
    sample_frames = 24
    learning_rate = 4e-4
    lr_scheduler_step_size = 500
    lr_scheduler_gamma = 0.5
    checkpoint_interval = 500
    num_workers = 8
