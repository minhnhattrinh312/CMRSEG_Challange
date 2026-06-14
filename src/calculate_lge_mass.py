import numpy as np
import nibabel as nib
from pathlib import Path
import os
import json

# Medical parameter definitions
# LGE related parameters
LGE_LABEL3_ID = 3  # Label 3 in the LGE mask
LGE_TISSUE_DENSITY = 1.05  # LGE tissue density, assuming the same as myocardium density


def calculate_label3_mass(lge_sa_mask_path):
    """
    Calculate the mass of label 3 in the LGE SA mask file.

    Args:
        lge_sa_mask_path: Path to the LGE segmentation mask file (str)

    Returns:
        float: Mass of label 3 (grams), None if failed
    """
    try:
        # Load LGE segmentation data
        lge_img = nib.load(lge_sa_mask_path)
        lge_data = np.round(lge_img.get_fdata()).astype(np.int16)

        # Get spacing information
        spacing = lge_img.header.get_zooms()[:3]  # (x, y, z)

        # Calculate total pixels of label 3
        label3_pixels = np.sum(lge_data == LGE_LABEL3_ID)
        label3_volume_ml = label3_pixels * spacing[0] * spacing[1] * spacing[2] / 1000.0  # Convert to ml
        label3_mass_g = label3_volume_ml * LGE_TISSUE_DENSITY  # Calculate mass (g)

        return label3_mass_g

    except Exception:
        return None


def calculate_lge_sa_metrics(lge_sa_mask_path):
    """
    Calculate mass of label 3 in the LGE SA mask file.

    Args:
        lge_sa_mask_path: Path to the LGE segmentation mask file (str)

    Returns:
        float: LGE SA label 3 mass (grams), None if failed
    """
    try:
        result = {}
        result["LGE_SA_Label3_Mass"] = calculate_label3_mass(lge_sa_mask_path)
        result["Scar_Quality"] = result["LGE_SA_Label3_Mass"]
        return result
    except Exception:
        return None


def process_nii_folder(input_dir):
    """
    Batch process all .nii.gz files in a directory and calculate scar quality for each file.

    Args:
        input_dir: Directory containing .nii.gz files

    Returns:
        pd.DataFrame: Calculation results for each file
    """
    input_path = Path(input_dir)
    nii_files = sorted(input_path.glob("*.nii.gz"))

    rows = []
    results = {}
    for nii_file in nii_files:
        metrics = calculate_lge_sa_metrics(str(nii_file))
        scar_quality = None if metrics is None else metrics.get("Scar_Quality")
        results[nii_file.name.replace(".nii.gz", "")] = scar_quality

    return results


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
DATA_DIR = os.path.join(OUTPUT_DIR, "task2_lge", "SAX")
# DATA_DIR = "/home/nhattm/CMR-MULTI/input/CMR-MULTI/LGE_MULTI/SAX_VAL/anno/"


result = process_nii_folder(DATA_DIR)
# save to json file
output_path = os.path.join(OUTPUT_DIR, "task2_lge", "mass_predictions.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(result, f, indent=4)
