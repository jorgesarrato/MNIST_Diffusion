import torch.optim as optim

def get_optimizer(model, training_config):
    opt_type = training_config.get("optimizer", "Adam")
    lr = training_config.get("lr", 1e-4)
    
    weight_decay = training_config.get("weight_decay", 0)

    if opt_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    
    elif opt_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif opt_type == "SGD":
        momentum = training_config.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    else:
        raise ValueError(f"Optimizer {opt_type} not supported.")