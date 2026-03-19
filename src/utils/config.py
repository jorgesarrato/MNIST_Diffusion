import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
    NYU_DATA_DIR = os.getenv("DATA_NYU_DIR")
    SUNRGBD_DATA_DIR = os.getenv("DATA_SUNRGBD_DIR")
    MLFLOW_DIR = os.getenv("MLFLOW_DIR")

    RANDOM_SEED = 42

    experiment_name = "UNet_FM_conditioned"
    run_name = "Residual-CrossAttn-PosSinCosEmbed-Distributed-SUNRGBD"

    data_config = {
            "data_dir": DATA_DIR,
            "batch_size": 8,
            "num_workers": 2,
            "num_workers_train": 2,
            "val_split": 0.1,
            "side_pixels": 140,
            "cache_size": 256,
            "log_depth": True
        }
    model_config = {
            "type": "UNet_FM",
            "filters_arr": [32, 64, 128, 256],
            "encoder_filters_arr": [32, 64, 128, 256],
            "encoder_denses_arr": [],
            "t_emb_size": 512,
            "label_emb_size": 512,
            "side_pixels": 140,
            "in_channels": 1,
            "in_channels_cond": 3,
            "n_channels_group": 8,
            "attn": True,
            "cross_attn": True,
            "use_residuals": True,
            "cond_type": "simple",
            "encoder_type": "vit"
            }
    training_config = {
            "lr": 1e-4,
            "epochs": 300,
            "optimizer": "AdamW",
            "weight_decay": 0.02,
            "scheduler_factor": 0.5,
            "patience": 20,
            "threshold": 0.005,
            "loss": "L1",
            "grad_weight": 0.0,
            "edge_weight": 0.0,
            "si_weight": 0.0,
            "weight_type": "none",
            "time_sampling": "uniform",
            "backbone_lr_ratio": 0.2,
            "scheduler": "OneCycleLR",
            "pct_start": 0.10,
            "final_div_factor": 25.0,
            "div_factor": 25.0,
            "ema_decay": 0.999,
            "cond_drop_prob": 0.10,
            "guidance_scale": [1.0, 1.5, 2.0],
            }

