import os
import json
import nibabel as nib
import numpy as np
import logging
from scipy.stats import mode
from natsort import natsorted

# --- Constants for Calculation ---
LV_BLOOD_POOL_ID = 2
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.environ.get("INPUT_DIR", os.path.join(PROJECT_ROOT, "input"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))


def create_3d_blocks(data, num_blocks):
    """Simplified 3D block creation splitting CINE sequence into phases."""
    total_slices = data.shape[2]
    if total_slices < num_blocks:
        return None

    slices_per_block = total_slices // num_blocks
    if slices_per_block == 0:
        return None

    blocks = []
    for block_idx in range(num_blocks):
        slice_indices = [block_idx + i * num_blocks for i in range(slices_per_block)]

        if len(slice_indices) > 10:
            effective_indices = slice_indices[1:-1]
        else:
            effective_indices = slice_indices

        blocks.append(data[:, :, effective_indices])

    return blocks


def calculate_cine_sa_metrics_internal(cine_sa_mask_path, slice_num):
    """Calculates LV volumes and EF from a SAX CINE mask."""
    try:
        niigz = nib.load(cine_sa_mask_path)
        pred_data = niigz.get_fdata()
        spacing = niigz.header.get_zooms()

        blocks = create_3d_blocks(pred_data, slice_num)
        if not blocks:
            return None

        # Determine target number of slices with LV via mode
        valid_z_lengths = []
        for block in blocks:
            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            count = np.sum(has_lv)
            if count > 0:
                valid_z_lengths.append(count)

        if not valid_z_lengths:
            return None

        target_z_length = mode(valid_z_lengths, keepdims=False).mode

        block_volumes = []
        for i, block in enumerate(blocks):
            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z_indices = np.where(has_lv)[0]

            if len(valid_z_indices) != target_z_length:
                continue

            # Volume calculation with spacing correction
            lv_voxels = np.sum(block[..., valid_z_indices] == LV_BLOOD_POOL_ID)
            lv_vol = lv_voxels * spacing[0] * spacing[1] * spacing[2] / 1000.0
            block_volumes.append(lv_vol)

        if not block_volumes:
            return None

        block_volumes.sort()
        es_lv_vol = block_volumes[0]  # Min volume
        ed_lv_vol = block_volumes[-1]  # Max volume

        lv_ef = ((ed_lv_vol - es_lv_vol) / ed_lv_vol * 100) if ed_lv_vol > 0 else 0

        return {"LV_EF": lv_ef}
    except Exception as e:
        logging.error(f"Error processing {cine_sa_mask_path}: {e}")
        return None


def read_id_slice_sax():
    mapping_candidates = [
        os.path.join(INPUT_DIR, "CMR-MULTI", "CINE_MULTI", "id_slice_info_valid.json"),
        os.path.join(INPUT_DIR, "CMR-MULTI", "CINE_MULTI", "sax_slice_info_test.json"),
        os.path.join(INPUT_DIR, "CMR-MULTI", "CINE_MULTI", "sax_slice_info_valid.json"),
    ]
    mapping_path = next((path for path in mapping_candidates if os.path.exists(path)), None)
    if mapping_path is None:
        print(f"Warning: no SAX slice info JSON found under {os.path.join(INPUT_DIR, 'CMR-MULTI', 'CINE_MULTI')}.")
        return {}
    with open(mapping_path, "r") as f:
        id_slice = json.load(f)
    return id_slice


def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


if __name__ == "__main__":
    # --- Configuration ---

    DATA_DIR = os.path.join(OUTPUT_DIR, "task1_cine", "SAX")
    # Load slice info from the specified JSON
    slice_info_sax = read_id_slice_sax()

    results = {}
    errors = []

    print(f"Starting calculation for files in {DATA_DIR}...")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist.")
    else:
        for filename in natsorted(os.listdir(DATA_DIR)):
            if not filename.endswith(".nii.gz"):
                continue
            # Extract case ID from filename (assuming it's the part before .nii.gz)
            case_id = filename.replace(".nii.gz", "")[-3:]
            print(f"Processing case ID: {case_id} with file {filename}...")
            # The mask path is the file itself in the specific DATA_DIR
            mask_path = os.path.join(DATA_DIR, filename)

            # Find matching slice_num from JSON
            try:
                # Based on user description, case_id is the key in sax_remaining_files_ordered.json
                slice_num = slice_info_sax[str(case_id)]
            except KeyError:
                errors.append({"id": case_id, "error": f"ID {case_id} not found in sax_remaining_files_ordered.json"})
                continue

            # Calculate EF
            metrics = calculate_cine_sa_metrics_internal(mask_path, slice_num)
            patient_name = (
                filename.replace(".nii.gz", "")
            )  # Ensure patient name ends with a 3-digit number
            if metrics:
                results[patient_name] = metrics["LV_EF"]
                print(f"Processed {case_id}: EF = {metrics['LV_EF']:.2f}%")
            else:
                errors.append({"id": case_id, "error": "Calculation returned None"})

    # Save outputs
    output_data = convert_to_serializable(results)
    os.makedirs(os.path.join(OUTPUT_DIR, "task1_cine"), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "task1_cine", "ef_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # with open("./output/task1_cine/lv_ef_errors.json", "w", encoding="utf-8") as f:
    #     json.dump(errors, f, ensure_ascii=False, indent=4)

    print(f"\nFinished. Successfully processed {len(results)} cases.")
    print(f"Results saved to ef_predictions.json")
