import numpy as np
from skimage.morphology import remove_small_objects
import torch
import nibabel as nib
from skimage.transform import resize
from skimage.morphology import (
    label,
    remove_small_objects,
    remove_small_holes,
    closing,
    ball,
)


# def min_max_normalize(volume):
#     volume = (volume - np.min(volume)) / (np.max(volume)-np.min(volume)) * 255.0
#     return volume.astype(np.uint8)
def min_max_normalize(image, cmr_type="cine"):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    """
    if "cine" in cmr_type.lower():
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [0.05, 99.5])
    else:
        image = np.abs(image)
        # non_zeros = image > 0
        low, high = np.percentile(image, [0.01, 99.9])
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image


def remove_small_elements(segmentation_mask, min_size_remove=3):
    # Convert segmentation mask values greater than 0 to 1
    pred_mask = segmentation_mask > 0
    # Remove small objects (connected components) from the binary image
    mask = remove_small_objects(pred_mask, max_size=min_size_remove)
    # Multiply original segmentation mask with the mask to remove small objects
    clean_segmentation = segmentation_mask * mask
    return clean_segmentation


def make_volume(ndarray, voxel_spacing):
    volume = np.prod(voxel_spacing) * (ndarray.sum())
    return volume


def crop_resize_image(image, new_dim=256):
    """
    Process a 3D numpy image by removing non-zero background, cropping to square,
    resizing, and saving crop_index as slice objects for restoration.

    Parameters:
    image (np.ndarray): Input image of shape (x, y, z)
    new_dim (int): Desired dimension for the output square image (new_dim, new_dim, z)
    Returns:
    tuple: (processed_image, restore_info)
        - processed_image: Processed image of shape (new_dim, new_dim, z)
        - restore_info: Dict containing original_shape, crop_index, original_dim, new_dim
    """
    # Step 1: Remove non-zero background using np.nonzero
    nz = np.nonzero(image)

    # Get min and max indices
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)

    # Create crop index for non-zero region
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))

    # Crop to non-zero region
    cropped = image[crop_index]

    # Step 2: Cut to min dimension of x, y to make square
    crop_h, crop_w = cropped.shape[:2]
    min_dim = min(crop_h, crop_w)

    # Calculate center crop
    start_h = (crop_h - min_dim) // 2
    start_w = (crop_w - min_dim) // 2

    square = cropped[start_h : start_h + min_dim, start_w : start_w + min_dim, :]

    # Calculate origin indices after square crop as slices
    orig_min_row = min_indices[0] + start_h
    orig_max_row = orig_min_row + min_dim
    orig_min_col = min_indices[1] + start_w
    orig_max_col = orig_min_col + min_dim
    orig_min_z = min_indices[2]
    orig_max_z = max_indices[2] + 1

    crop_index = (slice(orig_min_row, orig_max_row), slice(orig_min_col, orig_max_col), slice(orig_min_z, orig_max_z))

    # Step 3: Resize to new_dim
    current_dim = square.shape[0]

    if current_dim != new_dim:
        resized = resize(
            square,
            (new_dim, new_dim, square.shape[2]),
            anti_aliasing=True,
            preserve_range=True,
        )
        # Ensure output dtype matches input
        resized = resized.astype(square.dtype)
    else:
        resized = square.copy()

    # Step 4: Save crop_index for restoration
    restore_info = {
        "original_shape": image.shape,
        "crop_index": crop_index,
        "original_dim": current_dim,
        "new_dim": new_dim,
    }

    return resized, restore_info


def crop_resize_mask(mask, restore_info):
    """
    Process a 3D numpy segmentation mask using restore_info from image processing,
    cropping to the same square region and resizing to the same dimension.

    Parameters:
    mask (np.ndarray): Input segmentation mask of shape (x, y, z), same shape as original image
    restore_info (dict): Restoration information from process_image, containing
                        original_shape, crop_index, original_dim, new_dim

    Returns:
    tuple: (processed_mask, restore_info)
        - processed_mask: Processed mask of shape (new_dim, new_dim, z)
        - restore_info: Same restore_info for consistency in restoration
    """
    # Validate mask shape
    if mask.shape != restore_info["original_shape"]:
        raise ValueError("Mask shape must match original image shape")

    # Step 1: Crop to the square region using crop_index
    crop_index = restore_info["crop_index"]
    square = mask[crop_index]

    # Step 2: Resize to new_dim using skimage
    current_dim = square.shape[0]
    new_dim = restore_info["new_dim"]

    if current_dim != new_dim:
        resized = resize(square, output_shape=(new_dim, new_dim, square.shape[2]), order=0, anti_aliasing=False)
        # Ensure output dtype matches input
        resized = resized.astype(np.uint8)
    else:
        resized = square.copy()

    return resized


def restore_mask(processed_mask, restore_info):
    """
    Restore a processed 3D segmentation mask back to its original shape.

    Improvements:
    - Uses nearest-neighbor interpolation (order=0)
    - Disables anti-aliasing (to preserve discrete labels)
    - Ensures labels are integers after restoration
    - Handles edge cases (padding, rounding) more robustly
    """

    # Step 1: Resize back to the original square dimension (nearest-neighbor)
    current_dim = processed_mask.shape[0]
    original_dim = restore_info["original_dim"]

    if current_dim != original_dim:
        resized = resize(
            processed_mask,
            output_shape=(original_dim, original_dim, processed_mask.shape[2]),
            order=0,  # nearest neighbor → preserves labels
            anti_aliasing=False,  # turn off to avoid soft edges
            preserve_range=True,  # keep label values as-is
        )
    else:
        resized = processed_mask.copy()

    # Step 2: Initialize empty mask of the original shape
    original_shape = restore_info["original_shape"]
    restored = np.zeros(original_shape, dtype=np.int16)

    # Step 3: Paste the restored square into its original crop position
    crop_index = restore_info["crop_index"]
    restored[crop_index] = resized.astype(np.int16)  # round & ensure int labels

    return restored


def preprocess_data_nii(image_path, dim_resize=256, cmr_type="cine"):
    data = {}
    image, data["affine"], data["header"] = load_nii(image_path)
    image = min_max_normalize(image, cmr_type=cmr_type)

    resized_image, restore_info = crop_resize_image(image, dim_resize)
    data["restore_info"] = restore_info
    batch_images = []
    for i in range(resized_image.shape[-1]):
        slice_inputs = resized_image[..., i : i + 1]
        slices_image = torch.from_numpy(slice_inputs.transpose(-1, 0, 1))  # shape (1, dim_resize, dim_resize)
        batch_images.append(slices_image)

    batch_images = torch.stack(batch_images).float()  # shape (n,1, dim_resize, dim_resize)
    data["image"] = batch_images
    return data


def predict_patches(images, model, view, num_classes=4, batch_size=4, device="cuda", fp16=False):
    """return the patches"""
    probability_output = torch.zeros(
        (images.size(0), num_classes, images.size(2), images.size(3)),
        device=device,
    )

    batch_start = 0
    batch_end = batch_size
    while batch_start < images.size(0):
        image = images[batch_start:batch_end]
        with torch.inference_mode():
            image = image.to(device)
            if fp16:
                image = image.half()
            y_pred = model(image, view=view)
            probability_output[batch_start:batch_end] = y_pred
        batch_start += batch_size
        batch_end += batch_size
    return probability_output.cpu().numpy()


def predict_data_model(data, model, num_classes=4, batch_size=8, device="cuda", min_size_remove=500, fp16=False):
    probability_output = predict_patches(
        data["image"], model, num_classes=num_classes, batch_size=batch_size, device=device, fp16=fp16
    )  # shape (n, num_classes, dim_resize, dim_resize)
    seg = np.argmax(probability_output, axis=1).transpose(1, 2, 0)  # shape (dim_resize, dim_resize, n)
    seg = remove_small_elements(seg, min_size_remove=min_size_remove)
    invert_seg = restore_mask(seg, data["restore_info"])
    return invert_seg


# shape (n, num_classes, dim_resize, dim_resize)


def load_nii(img_path):
    """Function to load a 'nii' or 'nii.gz' file."""
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


from skimage.morphology import (
    label,
    remove_small_objects,
    remove_small_holes,
    closing,
    ball,
)


def postprocess_multiclass_volume(
    pred_volume,
    class_values=None,
    min_size=50,
    hole_area=50,
    smooth_radius=1,
    keep_largest=False,
):
    """
    Post-process a 3D multi-class segmentation volume.

    Parameters
    ----------
    pred_volume : np.ndarray
        3D predicted mask with shape [H, W, Z] or [Z, H, W].
        Each voxel has integer class label.

    class_values : list[int] or None
        Classes to process. If None, process all labels > 0.

    min_size : int
        Remove connected components smaller than this.

    hole_area : int
        Fill holes smaller than this.

    smooth_radius : int
        Radius for 3D morphological closing.

    keep_largest : bool
        If True, keep only the largest connected component for each class.

    Returns
    -------
    processed_volume : np.ndarray
        Post-processed segmentation volume.
    """

    pred_volume = np.asarray(pred_volume)

    if pred_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {pred_volume.shape}")

    if class_values is None:
        class_values = [c for c in np.unique(pred_volume) if c > 0]

    processed_volume = np.zeros_like(pred_volume)

    for class_value in class_values:
        mask = pred_volume == class_value

        if mask.sum() == 0:
            continue

        # Remove tiny components
        mask = remove_small_objects(mask, max_size=min_size)

        if mask.sum() == 0:
            continue

        # Keep only largest connected component
        if keep_largest:
            labeled = label(mask)
            component_ids, counts = np.unique(labeled, return_counts=True)

            # remove background label 0
            component_ids = component_ids[component_ids != 0]
            counts = counts[1:]

            if len(component_ids) == 0:
                continue

            largest_component = component_ids[np.argmax(counts)]
            mask = labeled == largest_component

        # Fill small holes
        mask = remove_small_holes(mask, max_size=hole_area)

        # Smooth in 3D
        if smooth_radius > 0:
            mask = closing(mask, ball(smooth_radius))

        processed_volume[mask] = class_value

    return processed_volume


# new_pred = postprocess_multiclass_volume(pred, min_size=500, hole_area=50, smooth_radius=1, keep_largest=True)

# print(f"Dice Score: {dice_score(new_pred, mask, class_id=1)}")
# print(f"Dice Score: {dice_score(new_pred, mask, class_id=2)}")
# print(f"Dice Score: {dice_score(new_pred, mask, class_id=3)}")
