import torch.optim as optim

def get_optimizer(model, training_config):
    opt_type = training_config.get("optimizer", "AdamW")
    lr = training_config.get("lr", 1e-4)
    weight_decay = training_config.get("weight_decay", 1e-2)
    
    backbone_lr_ratio = training_config.get("backbone_lr_ratio", 0.1)
    
    backbone_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "label_emb" in name:
            backbone_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {
            "params": base_params, 
            "lr": lr
        },
        {
            "params": backbone_params, 
            "lr": lr * backbone_lr_ratio,
            "weight_decay": weight_decay
        }
    ]

    if opt_type == "Adam":
        return optim.Adam(param_groups)
    
    elif opt_type == "AdamW":
        return optim.AdamW(param_groups, weight_decay=weight_decay)
    
    elif opt_type == "SGD":
        momentum = training_config.get("momentum", 0.9)
        return optim.SGD(param_groups, momentum=momentum)
    
    else:
        raise ValueError(f"Optimizer {opt_type} not supported.")