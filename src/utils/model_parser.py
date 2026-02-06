from models import UNet_FM  # Import your model classes here

MODEL_MAP = {
    "UNet_FM": UNet_FM,
}

def get_model(model_config):

    model_type = model_config.get("type")
    
    if model_type not in MODEL_MAP:
        raise ValueError(f"Model type '{model_type}' not found in MODEL_MAP. "
                         f"Available types: {list(MODEL_MAP.keys())}")

    params = model_config.copy()
    params.pop("type")

    model = MODEL_MAP[model_type](**params)
    
    return model