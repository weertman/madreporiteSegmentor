# configurator.py
import os
from pathlib import Path
import yaml
from datetime import datetime
import subprocess
import sys
from summarizer import summarize_experiment  # Import our summarizer function
from multiprocessing import Process

TRAIN_SCRIPT_PATH = Path("train.py")

base_config = {
    "paths": {
        "dataset": "../dataset/12-15-2024_madreporiteSegmentor",
        "output_dir": "./checkpoints"
    },
    "data": {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "input_shape": (640, 640),
        "random_seed": 42
    },
    "transforms": {
        "EmphasisCrop": {
            "p": 0.25,
            "scale_factor": 15.0,
            "scale_factor_range": (0.25, 1.2),
            "jitter_ratio": 0.2,
            "jitter_ratio_range": (0.8, 1.2)
        },
        "ShiftScaleRotate": {
            "p": 0.25,
            "shift_limit": 0.2,
            "scale_limit": 0.2,
            "rotate_limit": 45
        },
        "Resize": {
            "height": 640,
            "width": 640
        },
        "HorizontalFlip": {"p": 0.25},
        "VerticalFlip": {"p": 0.25},
        "ToGray": {"p": 0.25},
        "HueSaturationValue": {
            "p": 0.25,
            "hue_shift_limit": 20,
            "sat_shift_limit": 30,
            "val_shift_limit": 20
        },
        "RandomBrightnessContrast": {
            "p": 0.25,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2
        },
        "ColorJitter": {
            "p": 0.25,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.2
        },
        "Normalize": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "dataloader": {
        "batch_size": 8,
        "num_workers": 8
    },
    "model": {
        "decoder_type": "unet",
        "backbone": "pvt_v2_b5",
        "num_classes": 1,
        "pretrained": True,
        "freeze_backbone": True,
        "decoder_init_method": "kaiming_normal",
        "decoder_blocks": 5,
        "input_shape": (640, 640)
    },
    "training": {
        "wandb_project": f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-madreporite-segmentation",
        "num_epochs": 100,
        "freeze_backbone_epochs": 0,
        "lr": 1e-4,
        "device": "cuda:1",
        "USE_AUTO_POS_WEIGHT": False,
        "DEFAULT_POS_WEIGHT": 10.0,
        "max_pos_weight": 1000
    },
    "optimizer": {
        "type": "adamw",
        "lr": 1e-4
    },
    "scheduler": {
        "type": "reduce_on_plateau",
        "mode": "min",
        "factor": 0.1,
        "patience": 5
    }
}

# Your experiments list as before
experiments = [
    # Exp 1 (Baseline)
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 2
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 3
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 4
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 5
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 12,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 6
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 12,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 7
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 12,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 8
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 12,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 9
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b4",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 10
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b4",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 11
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 10.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 12
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 10.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 13
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 1.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 14
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 1.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 15
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b4",
        "dataloader.batch_size": 12,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 16
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b4",
        "dataloader.batch_size": 12,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 17
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b4",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 10.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 18
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-5,
        "model.backbone": "pvt_v2_b4",
        "dataloader.batch_size": 16,
        "model.decoder_type": "unet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 10.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (1280, 1280),
        "transforms.Resize.height": 1280,
        "transforms.Resize.width": 1280
    },
    # Exp 19
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "resnet",
        "model.decoder_init_method": "default",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
    # Exp 20
    {
        "training.num_epochs": 100,
        "data.train_ratio": 0.8,
        "data.val_ratio": 0.1,
        "data.test_ratio": 0.1,
        "training.lr": 1e-4,
        "model.backbone": "pvt_v2_b3",
        "dataloader.batch_size": 16,
        "model.decoder_type": "resnet",
        "model.decoder_init_method": "kaiming_normal",
        "optimizer.type": "adamw",
        "scheduler.type": None,
        "training.USE_AUTO_POS_WEIGHT": False,
        "training.DEFAULT_POS_WEIGHT": 100.0,
        "training.freeze_backbone_epochs": 0,
        "data.input_shape": (640, 640),
        "transforms.Resize.height": 640,
        "transforms.Resize.width": 640
    },
]

## set device for each experiment
## cuda:0 for even index, cuda:1 for odd index
for i, experiment in enumerate(experiments):
    experiment["training.device"] = "cuda:0" if i % 2 == 0 else "cuda:1"


def set_nested_value(d, keys, value):
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        set_nested_value(d[keys[0]], keys[1:], value)

def format_lr_scientific(lr):
    lr_sci = f"{lr:e}"  # e.g. "1.000000e-04"
    mantissa, exp = lr_sci.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    if mantissa == '':
        mantissa = '1'
    exp = int(exp)
    return f"{mantissa}e{exp}"

def create_run_name(exp_index, experiment):
    decoder = experiment.get("model.decoder_type", base_config["model"]["decoder_type"])
    backbone = experiment.get("model.backbone", base_config["model"]["backbone"])
    lr = experiment.get("training.lr", base_config["training"]["lr"])
    bs = experiment.get("dataloader.batch_size", base_config["dataloader"]["batch_size"])
    opt = experiment.get("optimizer.type", base_config["optimizer"]["type"])
    sched = experiment.get("scheduler.type", base_config["scheduler"]["type"])
    freeze_ep = experiment.get("training.freeze_backbone_epochs", base_config["training"]["freeze_backbone_epochs"])

    lr_str = format_lr_scientific(lr)
    run_name = f"exp{exp_index:02d}_{decoder}_{backbone}_lr{lr_str}_bs{bs}_{opt}_{sched}_freeze{freeze_ep}"
    run_name = run_name.replace(':', '-').replace('/', '-').replace('\\', '-')
    return run_name

def run_experiments_on_device(device, experiments_list):
    """
    Run a list of experiments sequentially on a given device.
    Each element of experiments_list is a tuple (config_path, run_name).
    """
    for (config_path, run_name) in experiments_list:
        print(f"Starting training for {run_name} on {device}")
        cmd = [
            sys.executable,  # "python" interpreter
            str(TRAIN_SCRIPT_PATH),
            "--config",
            str(config_path)
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Training complete for {run_name} on {device}")
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {run_name} on {device} with return code {e.returncode}.")
            print("Continuing with next experiment.")

if __name__ == "__main__":
    # Define the available GPU devices.
    # For example, if you have 2 A6000 GPUs:
    gpu_devices = ["cuda:0", "cuda:1"]
    # If you want to detect GPUs programmatically (not asked, but just a note),
    # you could query with torch.cuda.device_count() and generate ["cuda:0", "cuda:1", ...].
    # We'll keep it simple and hardcode for now.

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiments_base = Path("../experiments") / timestamp
    experiments_base.mkdir(parents=True, exist_ok=True)

    # Prepare a dictionary to hold experiments by device
    experiments_by_device = {dev: [] for dev in gpu_devices}

    # Assign experiments to devices
    # If an experiment specifies `training.device` already, we respect that.
    # Otherwise, we assign devices in a round-robin manner.
    device_count = len(gpu_devices)
    for i, experiment in enumerate(experiments, start=1):
        experiment_config = yaml.safe_load(yaml.safe_dump(base_config))

        # Apply experiment overrides
        for k, v in experiment.items():
            key_path = k.split('.')
            set_nested_value(experiment_config, key_path, v)

        # Determine device for this experiment
        if "training.device" in experiment:
            # User explicitly set device, use that.
            assigned_device = experiment["training.device"]
            # If assigned_device not in gpu_devices, warn or just add it dynamically.
            # For safety, let's ensure device is known:
            if assigned_device not in gpu_devices:
                print(f"Warning: Experiment {i} requested {assigned_device} not in {gpu_devices}. Using as is.")
                # We either append this device to gpu_devices if we want to handle it dynamically:
                if assigned_device not in experiments_by_device:
                    experiments_by_device[assigned_device] = []
        else:
            # Assign device by round-robin
            assigned_device = gpu_devices[(i - 1) % device_count]
            # Update config with assigned device
            set_nested_value(experiment_config, ["training", "device"], assigned_device)

        run_name = create_run_name(i, experiment)
        run_dir = experiments_base / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Update run_dir in config
        experiment_config["paths"]["run_dir"] = str(run_dir)

        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(experiment_config, f)

        print(f"Created experiment: {run_name}")
        print(f"Config written to: {config_path}")
        print(f"Assigned device: {assigned_device}")

        # Add this experiment to the device's list
        experiments_by_device[assigned_device].append((config_path, run_name))

    # Now we have all experiments assigned to devices.
    # We'll run them concurrently: one process per device.
    processes = []
    for device, device_experiments in experiments_by_device.items():
        if len(device_experiments) == 0:
            # No experiments for this device, skip
            continue
        # Spawn a process that runs all experiments for this device in sequence
        p = Process(target=run_experiments_on_device, args=(device, device_experiments))
        p.start()
        processes.append(p)

    # Wait for all device processes to finish
    for p in processes:
        p.join()

    # After all experiments finish, summarize
    summarize_experiment(experiments_base)
    print("All experiments completed. Summary generated.")
