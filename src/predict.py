import glob
import os
from pathlib import Path
import sys
from natsort import natsorted
import torch

from tqdm import tqdm
from hydra import compose, initialize_config_dir

src_dir = Path(__file__).resolve().parents[0]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
from segment2d import MultiHeadFCDenseNet
from segment2d.utils import *

PROJECT_ROOT = src_dir.parent
config_dir = PROJECT_ROOT / "config"
saved_model_dir = PROJECT_ROOT / "saved_models"
save_dir = PROJECT_ROOT / "output"
input_dir = PROJECT_ROOT / "input"
os.makedirs(save_dir, exist_ok=True)
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg_data = compose(config_name="data_config")
    cfg_train = compose(config_name="train_config")

num_classes_dict = {}
for cmr_type in cfg_data.CMR_MULTI:
    num_classes_dict[cmr_type] = {}
    for cmr_view in cfg_data.CMR_MULTI[cmr_type]:
        num_classes_dict[cmr_type][cmr_view] = cfg_data.CMR_MULTI[cmr_type][cmr_view].num_classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dict = {
    "CINE_MULTI": {
        "fold1": "dice_0.8720.ckpt",
        "fold2": "dice_0.8848.ckpt",
        "fold3": "dice_0.8836.ckpt",
        "fold4": "dice_0.8852.ckpt",
        "fold5": "dice_0.8723.ckpt",
    },
    "LGE_MULTI": {
        "fold1": "dice_0.8720.ckpt",
        "fold2": "dice_0.8848.ckpt",
        "fold3": "dice_0.8836.ckpt",
        "fold4": "dice_0.8723.ckpt",
        "fold5": "dice_0.8723.ckpt",
    },
}
for i, cmr_type_path in enumerate(glob.glob(os.path.join(input_dir, "*/*MULTI")), start=1):
    cmr_type = os.path.basename(cmr_type_path)
    task_dir = f"task{i}_{cmr_type.lower().split('_')[0]}"
    os.makedirs(os.path.join(save_dir, task_dir), exist_ok=True)
    for cmr_view_path in glob.glob(os.path.join(cmr_type_path, f"*{cfg_train.PREDICT_FOLDER}")):
        print(f"Processing {cmr_type} - {cmr_view_path}...")
        cmr_view = os.path.basename(cmr_view_path).split("_")[0]
        os.makedirs(os.path.join(save_dir, task_dir, cmr_view), exist_ok=True)
        list_patient = natsorted(glob.glob(os.path.join(cmr_view_path, "image/*")))
        model = MultiHeadFCDenseNet(in_channels=cfg_train.INPUT_DIM_MODEL, head_classes=num_classes_dict[cmr_type])
        model = model.to(device)
        model = torch.compile(model)
        for case_count, image_path in enumerate(tqdm(list_patient), start=1):
            patient_name = os.path.basename(image_path)
            data = preprocess_data_nii(image_path, cmr_type=cmr_type)
            num_class = num_classes_dict[cmr_type][cmr_view]
            prob = np.zeros((data["image"].shape[0], num_class, data["image"].shape[2], data["image"].shape[3]))
            for fold in range(1, 6):
                if fold not in [1,2,3]:
                    continue
                checkpoint = torch.load(
                    saved_model_dir / f"{cmr_type.split('_')[0]}_fold{fold}/{weights_dict[cmr_type][f'fold{fold}']}",
                    map_location=device,
                )
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
                model.load_state_dict(state_dict)
                model = model.eval()
                probability_output = predict_patches(
                    data["image"],
                    model,
                    view=cmr_view,
                    num_classes=num_classes_dict[cmr_type][cmr_view],
                    batch_size=8,
                    device=device,
                )
                # probability_output shape (n, num_classes, dim_resize, dim_resize)
                prob += probability_output
            # calculate the argmax of the average probability
            seg = np.argmax(prob, axis=1).transpose(1, 2, 0)  # shape (dim_resize, dim_resize, n)
            seg = postprocess_multiclass_volume(seg, min_size=200, hole_area=100, smooth_radius=0, keep_largest=False)
            invert_seg = restore_mask(seg, data["restore_info"])
            save_patient_path = os.path.join(save_dir, task_dir, cmr_view, patient_name)
            save_nii(save_patient_path, invert_seg, data["affine"], data["header"])
            # break
    break
