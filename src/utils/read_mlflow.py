import torch
from mlflow.tracking import MlflowClient

def load_artifacts(experiment_name, artifact_prefix="snapshots"):
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    experiment_id = experiment.experiment_id
    print(f"Found Experiment '{experiment_name}' with ID: {experiment_id}")

    runs = client.search_runs(experiment_ids=[experiment_id])
    
    model_snapshots_dict = {}

    for run in runs:
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", run_id)
        
        try:
            artifacts_list = client.list_artifacts(run_id)
        except Exception as e:
            print(f"Could not list artifacts for {run_name}: {e}")
            continue

        snapshot_infos = [
            f for f in artifacts_list 
            if f.path.startswith(artifact_prefix) and f.path.endswith(".pt")
        ]
        
        if not snapshot_infos:
            continue

        snapshot_infos.sort(key=lambda x: int(x.path.split('_')[-1].split('.')[0]))

        run_samples_list = []
        
        for artifact_info in snapshot_infos:
            local_path = client.download_artifacts(run_id, artifact_info.path)
            
            sample_snapshots = torch.load(local_path, map_location='cpu')
            run_samples_list.append(sample_snapshots)
            
        if run_samples_list:
            model_snapshots_dict[run_name] = run_samples_list
            print(f"Loaded {len(run_samples_list)} samples for: {run_name}")

    return model_snapshots_dict
