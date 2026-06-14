from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import glob
import os
from pathlib import Path
import sys
import random
import torch

torch.set_float32_matmul_precision("high")
random.seed(42)
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from segment2d import MultiHeadFCDenseNet, MultiViewDataset, SingleViewBatchSampler
from segment2d.model import Segmenter
import pandas as pd
from hydra import compose, initialize_config_dir

PROJECT_ROOT = src_dir.parent
config_dir = PROJECT_ROOT / "config"
saved_model_dir = PROJECT_ROOT / "saved_models"
csv_path = PROJECT_ROOT / "data" / "csv_files"

with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg_data = compose(config_name="data_config")
    cfg_train = compose(
        config_name="train_config",
        overrides=["LEARNING_RATE=5e-4", "BATCH_SIZE=4", "EPOCHS=2000", "PATIENCE_LR=100", "PATIENCE_ES=500"],
    )


for fold in range(1, 6):
    print(f"training fold {fold} for LGE...")
    # validation set: fold i, training set: remaining folds
    val_files, train_files, num_classes_dict, class_weights_dict = {}, {}, {}, {}

    for key in cfg_data.CMR_MULTI.LGE_MULTI.keys():
        csv_file = pd.read_csv(csv_path / f"LGE_MULTI_{key}_info.csv")
        val_files[key] = csv_file[csv_file["fold"] == fold]["path"].tolist()
        train_files[key] = csv_file[csv_file["fold"] != fold]["path"].tolist()
        num_classes_dict[key] = cfg_data.CMR_MULTI.LGE_MULTI[key].num_classes
        class_weights_dict[key] = cfg_data.CMR_MULTI.LGE_MULTI[key].class_weights

        print(f"Training samples for {key}: {len(train_files[key])} | Validation samples: {len(val_files[key])}")

    model = MultiHeadFCDenseNet(in_channels=cfg_train.INPUT_DIM_MODEL, head_classes=num_classes_dict)

    segmenter = Segmenter(
        model,
        class_weights_dict=class_weights_dict,
        num_classes_dict=num_classes_dict,
        learning_rate=cfg_train.LEARNING_RATE,
        factor_lr=cfg_train.FACTOR_LR,
        patience_lr=cfg_train.PATIENCE_LR,
        size_augmentation=cfg_data.RESIZE_DIM,
    )

    train_dataset = MultiViewDataset(train_files)
    val_dataset = MultiViewDataset(val_files)
    batch_sampler_train = SingleViewBatchSampler(train_dataset, batch_size=cfg_train.BATCH_SIZE)
    batch_sampler_val = SingleViewBatchSampler(val_dataset, batch_size=cfg_train.BATCH_SIZE, shuffle=False)

    # Define data loaders for the training and test data
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=cfg_train.NUM_WORKERS,
        prefetch_factor=cfg_train.PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=batch_sampler_val,
        num_workers=cfg_train.NUM_WORKERS,
        prefetch_factor=cfg_train.PREFETCH_FACTOR,
    )
    # If wandb_logger is True, create a WandbLogger object

    wandb_logger = (
        WandbLogger(
            project="CMR-MULTI",
            name=f"LGE_fold{fold}",
            resume=False,
        )
        if cfg_train.LOG_WANDB
        else False
    )
    save_dir = saved_model_dir / f"LGE_fold{fold}"
    os.makedirs(save_dir, exist_ok=True)
    # Initialize a ModelCheckpoint callback to save the model weights after each epoch
    check_point = ModelCheckpoint(
        save_dir,
        filename="dice_{avg_val_dice:0.4f}",
        monitor="avg_val_dice",
        mode="max",
        save_top_k=cfg_train.SAVE_MODEL_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_last=True,
    )

    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Initialize a EarlyStopping callback to stop training if the validation loss does not improve for a certain number of epochs
    early_stopping = EarlyStopping(
        monitor="avg_val_dice",
        mode="max",
        patience=cfg_train.PATIENCE_ES,
        verbose=True,
        strict=False,
    )

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": cfg_train.ACCELERATOR,
        "devices": cfg_train.DEVICES,
        "benchmark": True,
        "enable_progress_bar": True,
        # "overfit_batches" :5,
        "logger": wandb_logger,
        "callbacks": [check_point, early_stopping, lr_monitor],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 4,
        "max_epochs": cfg_train.EPOCHS,
        "precision": cfg_train.PRECISION,
    }

    checkpoint_paths = glob.glob(os.path.join(saved_model_dir, "CINE_fold1/*.ckpt"))
    checkpoint_paths.sort()
    # If there are checkpoint paths and the load_checkpoint flag is set to True
    if checkpoint_paths and cfg_train.USE_TRANSFER_LEARNING.LGE:
        print("using pretrained model for LGE ...")
        # load model weights from the last checkpoint of CINE fold 2 to initialize the LGE model
        checkpoint = checkpoint_paths[-1]
        print(f"load checkpoint: {checkpoint}")
        # Load the model weights from the selected checkpoint with strict=False to allow missing keys
        # load endcoder and decoder weights, so delete the head weights in the checkpoint state dict before loading
        checkpoint_state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        # Delete the head weights from the checkpoint state dict
        for key in list(checkpoint_state_dict.keys()):
            if "finalHeads" in key:
                del checkpoint_state_dict[key]

        segmenter.load_state_dict(checkpoint_state_dict, strict=False)
    # freeze the encoder weights for the first 50 epochs, then unfreeze them for the remaining epochs
    if cfg_train.FREEZE_MODEL.LGE:
        print("freezing model weights for LGE ...")
        PARAMS_TRAINER_WARMUP = {
            "accelerator": cfg_train.ACCELERATOR,
            "devices": cfg_train.DEVICES,
            "benchmark": True,
            "enable_progress_bar": True,
            # "overfit_batches" :5,
            "logger": False,
            # "callbacks": [check_point, early_stopping, lr_monitor],
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 4,
            "max_epochs": 100,
            "precision": cfg_train.PRECISION,
        }
        segmenter.freeze_model()
        trainer = pl.Trainer(**PARAMS_TRAINER_WARMUP)
        trainer.fit(segmenter, train_loader, val_loader)
        segmenter.unfreeze_model()

    # Train the model using the train_dataset and test_dataset data loaders
    # Initialize a Trainer object with the specified parameters
    trainer = pl.Trainer(**PARAMS_TRAINER)
    trainer.fit(segmenter, train_loader, val_loader)
    if cfg_train.LOG_WANDB:
        import wandb

        wandb.finish()
