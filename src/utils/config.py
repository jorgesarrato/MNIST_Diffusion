import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
    NYU_DATA_DIR = os.getenv("DATA_NYU_DIR")
    MLFLOW_DIR = os.getenv("MLFLOW_DIR")

    RANDOM_SEED = 46020

    experiment_name = "UNet_FM_conditioned"
    run_name = "Residual-CrossAttn-PosSinCosEmbed"

    data_config = {
            "data_dir": DATA_DIR,
            "batch_size": 64,
            "num_workers": 2,
            "num_workers_train": 4,
            "val_split": 0.1,
            "side_pixels": 128
        }
    model_config = {
            "type": "UNet_FM",
            "filters_arr": [64, 128, 256, 512],
            "encoder_filters_arr": [64, 128, 256],
            "encoder_denses_arr": [512, 256, 128],
            "t_emb_size": 256,
            "label_emb_size": 512,
            "side_pixels": 128,
            "in_channels": 1,
            "in_channels_cond": 3,
            "n_channels_group": 8,
            "attn": True,
            "cross_attn": True,
            "use_residuals": True,
            "cond_type": "simple",
            "encoder_type": "resnet"
        }
    training_config = {
            "lr": 2e-4,
            "epochs": 500,
            "optimizer": "AdamW",
            "weight_decay": 0.03,
            "scheduler_factor": 0.5,
            "patience": 30,
            "threshold": 0.005,
            "loss": "L1",
            "weight_type": "linear",
            "time_sampling": "logit_normal",
            "backbone_lr_ratio": 0.1,
            "scheduler": "OneCycleLR",
            "pct_start": 0.1,
            "final_div_factor": 10.0,
            "div_factor": 25.0
            }

