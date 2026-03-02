import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, training_config, steps_per_epoch=None):

    sched_type = training_config.get("scheduler", "OneCycleLR")
    
    if sched_type is None or sched_type.lower() == "none":
        return None
        
    epochs = training_config.get("epochs", 500)
    
    if sched_type == "OneCycleLR":
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR requires 'steps_per_epoch' to be passed to get_scheduler().")
            
        max_lr = training_config.get("max_lr", training_config.get("lr", 2e-4))
        pct_start = training_config.get("pct_start", 0.1)
        div_factor = training_config.get("div_factor", 25.0)
        final_div_factor = training_config.get("final_div_factor", 10.0)
        
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy='cos'
        )
        
    elif sched_type == "ReduceLROnPlateau":
        mode = training_config.get("plateau_mode", "min")
        factor = training_config.get("plateau_factor", 0.5)
        patience = training_config.get("plateau_patience", 5)
        min_lr = training_config.get("min_lr", 1e-6)
        
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
        
    elif sched_type == "CosineAnnealingLR":
        T_max = training_config.get("T_max", epochs)
        eta_min = training_config.get("min_lr", 1e-7)
        
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        
    elif sched_type == "StepLR":
        step_size = training_config.get("step_size", 50)
        gamma = training_config.get("gamma", 0.5)
        
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
    elif sched_type == "CosineAnnealingWarmRestarts":
        T_0 = training_config.get("T_0", 100)
        T_mult = training_config.get("T_mult", 2)
        eta_min = training_config.get("min_lr", 1e-6)
        
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        
    else:
        raise ValueError(f"Scheduler {sched_type} not supported.")