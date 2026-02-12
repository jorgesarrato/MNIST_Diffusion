import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")

    RANDOM_SEED = 46020

    experiment_name = "UNet_FM_conditioned"
    run_name = "Base_test"

    data_config = {
            "data_dir": DATA_DIR,
            "batch_size": 256,
            "num_workers": 1,
            "val_split": 0.1
             }
    model_config = {
            "type": "UNet_FM_Residuals",
            "filters_arr": [32, 64, 128],
            "t_emb_size": 64,
            "label_emb_size": 32,
            "attn": False
        }
    training_config = {
            "lr": 1e-4,
            "epochs": 50,
            "optimizer": "AdamW",
            "wight_decay": 0.01,
            "scheduler_factor": 0.5,
            "patience": 5,
        }

