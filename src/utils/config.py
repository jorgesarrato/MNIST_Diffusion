import os
from dotenv import load_dotenv
import argparse

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
    NYU_DATA_DIR = os.getenv("DATA_NYU_DIR")
    SUNRGBD_DATA_DIR = os.getenv("DATA_SUNRGBD_DIR")
    MYSUNRGBD_DATA_DIR = os.getenv("DATA_MYSUNRGBD_DIR")
    MLFLOW_DIR = os.getenv("MLFLOW_DIR")

    RANDOM_SEED = 42

    experiment_name = "UNet_FM_conditioned"
    run_name = "DINOv2-attn-crossattn"

    data_config = {
            "data_dir": DATA_DIR,
            "batch_size": 8,
            "num_workers": 2,
            "num_workers_train": 2,
            "val_split": 0.1,
            "side_pixels": 168,
            "cache_size": 256,
            "log_depth": True
        }
    model_config = {
            "type": "UNet_FM",
            "filters_arr": [32, 64, 128, 256],
            "encoder_filters_arr": [32, 64, 128],
            "encoder_denses_arr": [],
            "t_emb_size": 256,
            "label_emb_size": 256,
            "side_pixels": 168,
            "in_channels": 1,
            "in_channels_cond": 3,
            "n_channels_group": 8,
            "attn": True,
            "cross_attn": True,
            "use_residuals": True,
            "cond_type": "simple",
            "encoder_type": "vit",
            "norm_type": "layernorm",
            "pos_embed_type": "sincos"
            }
    training_config = {
            "lr": 3e-4,
            "epochs": 300,
            "optimizer": "AdamW",
            "weight_decay": 0.02,
            "scheduler_factor": 0.5,
            "patience": 10,
            "threshold": 0.005,
            "loss": "L1",
            "grad_weight": 0.0,
            "edge_weight": 0.0,
            "si_weight": 0.0,
            "x1_weight": 0.0,
            "weight_type": "none",
            "time_sampling": "uniform",
            "backbone_lr_ratio": 0.2,
            "scheduler": "OneCycleLR",
            "pct_start": 0.10,
            "final_div_factor": 25.0,
            "div_factor": 25.0,
            "ema_decay": 0.9995,
            "cond_drop_prob": 0.10,
            "guidance_scale": [1.0, 1.5, 2.0],
            }

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Override Config parameters")
    
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--edge_weight", type=float, default=None)
    parser.add_argument("--si_weight", type=float, default=None)
    parser.add_argument("--x1_weight", type=float, default=None)
    parser.add_argument("--grad_weight", type=float, default=None)
    parser.add_argument("--encoder_type", type=str, default=None)

    parser.add_argument("--attn", type=str2bool, default=None)
    parser.add_argument("--cross_attn", type=str2bool, default=None)

    args, _ = parser.parse_known_args()

    if args.run_name is not None:
        Config.run_name = args.run_name

    if args.loss is not None:
        Config.training_config["loss"] = args.loss

    if args.edge_weight is not None:
        Config.training_config["edge_weight"] = args.edge_weight

    if args.si_weight is not None:
        Config.training_config["si_weight"] = args.si_weight

    if args.x1_weight is not None:
        Config.training_config["x1_weight"] = args.x1_weight

    if args.grad_weight is not None:
        Config.training_config["grad_weight"] = args.grad_weight

    if args.attn is not None:
        Config.model_config["attn"] = args.attn

    if args.cross_attn is not None:
        Config.model_config["cross_attn"] = args.cross_attn

    if args.encoder_type is not None:
        Config.model_config["encoder_type"] = args.encoder_type

    return args

 