import torch
import lightning.pytorch as pl
from segment2d.utils import *
from segment2d.losses import *
import torch.nn.functional as F
from kornia.augmentation import *
from torch.optim import NAdam
import kornia as K
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Segmenter(pl.LightningModule):

    def __init__(
        self,
        model,
        class_weights_dict,
        num_classes_dict,
        learning_rate,
        factor_lr,
        patience_lr,
        size_augmentation=256,
    ):
        super().__init__()
        self.model = model
        # torch 2.3 => compile to make faster
        self.model = torch.compile(self.model)

        self.class_weights_dict = class_weights_dict
        self.num_classes_dict = num_classes_dict
        self.learning_rate = learning_rate
        self.factor_lr = factor_lr
        self.patience_lr = patience_lr
        ################ augmentation ############################
        self.transform = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomGaussianNoise(mean=0.0, std=0.02, p=0.2),
            RandomResizedCrop([size_augmentation, size_augmentation], scale=(0.8, 1.2), ratio=(0.8, 1.2), p=0.5),
            data_keys=["input", "mask"],
        )
        self.test_metric = []
        self.validation_step_outputs = []

    def on_fit_start(self):
        self.training_loss = ActiveFocalContourLossMultiHead(
            self.device, self.class_weights_dict, self.num_classes_dict
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            with torch.no_grad():
                image, mask, view = batch
                # apply augmentation to image and mask
                image, mask = self.transform(image, mask)
                batch = (image, mask, view)
        return batch

    def forward(self, x, view=None):
        # return self.model(self.normalize(x))
        return self.model(x, view=view)

    def training_step(self, batch, batch_idx):
        image, y_true, view = batch
        view = view[0]
        y_pred = self.model(image, view=view)
        loss = self.training_loss(y_true, y_pred, view=view)
        metrics = {"train_loss": loss}
        for i in range(1, self.num_classes_dict[view]):
            dice_class = dice_slice(y_true, y_pred, class_index=i)
            metrics[f"train_dice_{view}_{i}"] = dice_class
        # mean dice of all classes
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, y_true, view = batch
        view = view[0]
        y_pred = self.model(image, view=view)
        # loss = self.training_loss(y_true, y_pred, view=view)
        metrics = {}
        for i in range(1, self.num_classes_dict[view]):
            dice_class = dice_slice(y_true, y_pred, class_index=i)
            metrics[f"val_dice_step_{view}_{i}"] = dice_class
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        # Initialize storage per view/class
        dice_accum = {
            view: {i: [] for i in range(1, num_classes)} for view, num_classes in self.num_classes_dict.items()
        }

        # Aggregate dice from each validation_step output
        for step_output in self.validation_step_outputs:
            for key, value in step_output.items():
                # key format: val_dice_step_{view}_{i}
                parts = key.split("_")
                view = parts[3]
                class_idx = int(parts[4])
                dice_accum[view][class_idx].append(value)

        # Compute mean dice per view/class
        metrics = {}
        for view, class_dict in dice_accum.items():
            for i, values in class_dict.items():
                if values:
                    metrics[f"val_dice_{view}_{i}"] = torch.mean(torch.stack(values))
                else:
                    metrics[f"val_dice_{view}_{i}"] = torch.tensor(0.0, device=self.device)

        # Compute overall mean dice
        all_dice = [v for k, v in metrics.items() if k.startswith("val_dice_")]
        if all_dice:
            metrics["avg_val_dice"] = torch.mean(torch.stack(all_dice))
        else:
            metrics["avg_val_dice"] = torch.tensor(0.0, device=self.device)
        # print(f"Validation metrics: {metrics}")
        # Clear stored step outputs
        self.validation_step_outputs = []

        # Log metrics
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = NAdam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=self.factor_lr, patience=self.patience_lr)

        lr_schedulers = {
            "scheduler": scheduler,
            "monitor": "avg_val_dice",
            "strict": False,
        }

        return [optimizer], lr_schedulers

    def freeze_model(self):
        # freeze the encoder weights, but keep the head weights trainable
        # unfreeze the first conv layer weights to allow the model to adapt to the new input data distribution of LGE modality
        for name, param in self.model.named_parameters():
            if "finalHeads" not in name:
                param.requires_grad = False
            if "firstconv" in name:
                param.requires_grad = True

    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True


def dice_slice(y_true, y_pred, class_index=1, smooth=1e-5):
    output_standard = torch.argmax(y_pred, dim=1, keepdim=True)
    output = torch.where(output_standard == class_index, 1, 0)
    label = torch.where(y_true == class_index, 1, 0)

    intersection = torch.sum(label * output, dim=[1, 2, 3])
    union = torch.sum(label, dim=[1, 2, 3]) + torch.sum(output, dim=[1, 2, 3])
    return torch.mean((2.0 * intersection + smooth) / (union + smooth), dim=0)
