import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from segment2d import crop_resize_image, crop_resize_mask, min_max_normalize
from natsort import natsorted
import csv
from tqdm import tqdm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = src_dir.parent

config_dir = PROJECT_ROOT / "config"
save_dir = PROJECT_ROOT / "data/CMR-MULTI_npz"
csv_dir = PROJECT_ROOT / "data/csv_files"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="data_config")

# loop through all folder in data config
for cmr_type in cfg.CMR_MULTI.keys():
    # for cmr_type in ["LGE_MULTI"]:
    os.makedirs(os.path.join(save_dir, cmr_type), exist_ok=True)
    for cmr_view in cfg.CMR_MULTI[cmr_type].keys():
        os.makedirs(os.path.join(save_dir, cmr_type, cmr_view), exist_ok=True)
        path = PROJECT_ROOT / cfg.CMR_MULTI[cmr_type][cmr_view].path
        list_patient = natsorted(glob.glob(os.path.join(path, "image/*")))
        # shuffle and divide list patient into 5 folds for cross validation
        fold_size = len(list_patient) // 5
        # write fold_patient to 1 csv file
        with open(f"{csv_dir}/{cmr_type}_{cmr_view}_info.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["id_patient", "path", "fold"])
            writer.writeheader()
            for fold in range(1, 6):
                print(f"Processing fold {fold} for {cmr_type} - {cmr_view}...")
                fold_patient = list_patient[(fold - 1) * fold_size : fold * fold_size]
                for image_path in tqdm(fold_patient):
                    id_patient = image_path.split("/")[-1].split(".")[0]
                    anno_path = image_path.replace("image", "anno")
                    slice_count = 0
                    image = nib.load(image_path).get_fdata()
                    mask = nib.load(anno_path).get_fdata()
                    image = min_max_normalize(image, cmr_type=cmr_type)
                    resize_image, restore_info = crop_resize_image(image, cfg.RESIZE_DIM)
                    resize_mask = crop_resize_mask(mask, restore_info)

                    for i in range(resize_image.shape[-1]):
                        if np.sum(resize_mask[:, :, i]) == 0:
                            continue
                        slice_image = resize_image[:, :, i : i + 1]
                        slice_mask = resize_mask[:, :, i]
                        slice_count += 1
                        npz_save_path = os.path.join(save_dir, cmr_type, cmr_view, f"{id_patient}_{slice_count}.npz")
                        np.savez_compressed(npz_save_path, image=slice_image, mask=slice_mask.astype(np.uint8))
                        writer.writerow({"id_patient": id_patient, "path": npz_save_path, "fold": fold})
