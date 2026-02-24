import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
    NYU_DATA_DIR = os.getenv("DATA_NYU_DIR")

    RANDOM_SEED = 46020

    experiment_name = "UNet_FM_conditioned"
    run_name = "SelfLabel"

    data_config = {
            "data_dir": DATA_DIR,
            "batch_size": 16,
            "num_workers": 1,
            "val_split": 0.1,
            "side_pixels": 128
        }
    model_config = {
            "type": "UNet_FM",
            "filters_arr": [128, 256, 512],
            "encoder_filters_arr": [128, 256, 512],
            "encoder_denses_arr": [512, 256, 128],
            "t_emb_size": 512,
            "label_emb_size": 1024,
            "side_pixels": 128,
            "in_channels": 1,
            "in_channels_cond": 3,
            "n_channels_group": 8,
            "attn": True,
            "cross_attn": True,
            "use_residuals": True,
            "cond_type": "concat",
            "encoder_type": "resnet"
        }
    training_config = {
            "lr": 1e-4,
            "epochs": 100,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "scheduler_factor": 0.5,
            "patience": 5,
            "loss": "L1_Grad",
            "weight_type": "quad",
            "backbone_lr_ratio": 0.1
            }

