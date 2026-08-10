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
import scipy.ndimage as ndi
import SimpleITK as sitk


def anisodiff3(
    stack,
    niter=1,
    kappa=50,
    gamma=0.1,
    step=(1.0, 1.0, 1.0),
    option=1,
):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.
    """

    # initialize output array
    stack = stack.astype("float32")
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-((deltaD / kappa) ** 2.0)) / step[0]
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[1]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[2]
        elif option == 2:
            gD = 1.0 / (1.0 + (deltaD / kappa) ** 2.0) / step[0]
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[1]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

    return stackout


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0), option=1, ploton=False):

    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype("float32")
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[0]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[1]
        elif option == 2:
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[0]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


def n4_bias_correction_3d(vol_np, mask_np=None):
    """
    vol_np: 3D numpy array, shape [D, H, W]
    """

    vol_np = vol_np.astype(np.float32)

    img_sitk = sitk.GetImageFromArray(vol_np)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)

    if mask_np is None:
        mask_sitk = sitk.OtsuThreshold(img_sitk, 0, 1, 200)
    else:
        mask_sitk = sitk.GetImageFromArray(mask_np.astype(np.uint8))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])

    corrected_sitk = corrector.Execute(img_sitk, mask_sitk)

    return sitk.GetArrayFromImage(corrected_sitk).astype(np.float32)


def preprocess_cmr(
    volume: np.ndarray,
) -> np.ndarray:
    enhanced = anisodiff3(volume, niter=1, kappa=50, gamma=0.015, step=(1.0, 1.0, 1.0), option=1)
    enhanced = n4_bias_correction_3d(enhanced)
    enhanced = np.clip(enhanced, np.percentile(enhanced, 0.5), np.percentile(enhanced, 99.5))
    enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced))
    return enhanced


import scipy.ndimage as ndi


def augment_scar_only_elastic(
    image,
    mask,
    scar_label=3,
    myo_label=2,
    alpha=25.0,
    sigma=8.0,
    roi_dilation=15,
    myo_dilation=2,
    boundary_sigma=1.5,
):
    """
    Elastic augmentation for myocardial scar only.

    image: 2D image, shape [H, W]
    mask:  2D mask, shape [H, W]
           1 = LV cavity
           2 = LV myocardium
           3 = scar
           4 = RV cavity

    Returns:
        aug_image, aug_mask
    """

    image = image.astype(np.float32)
    mask = mask.copy()

    scar = mask == scar_label
    myo = mask == myo_label

    if scar.sum() == 0:
        return image.copy(), mask.copy()

    h, w = image.shape

    # --------------------------------------------------
    # 1. Create random smooth deformation field
    # --------------------------------------------------
    dx = np.random.randn(h, w).astype(np.float32)
    dy = np.random.randn(h, w).astype(np.float32)

    dx = ndi.gaussian_filter(dx, sigma=sigma)
    dy = ndi.gaussian_filter(dy, sigma=sigma)

    dx *= alpha
    dy *= alpha

    # --------------------------------------------------
    # 2. Restrict deformation to scar neighborhood
    # --------------------------------------------------
    roi = ndi.binary_dilation(scar, iterations=roi_dilation)

    dx *= roi
    dy *= roi

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    coords = np.array([yy + dy, xx + dx])

    # --------------------------------------------------
    # 3. Warp scar mask
    # --------------------------------------------------
    scar_def = (
        ndi.map_coordinates(
            scar.astype(np.float32),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
        > 0.5
    )

    # --------------------------------------------------
    # 4. Keep deformed scar anatomically valid
    #    Scar should remain inside/near myocardium.
    # --------------------------------------------------
    myo_region = ndi.binary_dilation(myo | scar, iterations=myo_dilation)
    scar_def = scar_def & myo_region

    if scar_def.sum() == 0:
        return image.copy(), mask.copy()
    # --------------------------------------------------
    # 5. Warp full image
    # --------------------------------------------------
    image_def = ndi.map_coordinates(image, coords, order=1, mode="nearest")

    # --------------------------------------------------
    # 6. Create new mask
    # --------------------------------------------------
    aug_mask = mask.copy()
    aug_mask[aug_mask == scar_label] = myo_label
    aug_mask[scar_def] = scar_label

    # --------------------------------------------------
    # 7. Smooth image transition near new scar
    # --------------------------------------------------
    boundary = ndi.gaussian_filter(scar_def.astype(np.float32), sigma=boundary_sigma)
    boundary = np.clip(boundary, 0.0, 1.0)

    blend_region = ndi.binary_dilation(scar | scar_def, iterations=roi_dilation)

    aug_image = image.copy()
    aug_image[blend_region] = image_def[blend_region] * boundary[blend_region] + image[blend_region] * (
        1.0 - boundary[blend_region]
    )

    return aug_image, aug_mask


def min_max_normalize(image, cmr_type="cine"):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    """
    if "cine" in cmr_type.lower():
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [0.05, 99.5])
        image = np.clip(image, low, high)
        image = (image - low) / (high - low)
    else:
        # image = np.abs(image)
        # low, high = np.percentile(image, [0.01, 99.9])
        # image = np.clip(image, low, high)
        # image = (image - low) / (high - low)
        ############### vs2 ###############################
        image = preprocess_cmr(image)

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


from scipy.ndimage import label, binary_dilation, generate_binary_structure


def postprocess_scar_between_lv_myo_3d(
    pred,
    prob=None,
    lv_label=1,
    myo_label=2,
    scar_label=3,
    min_ratio=0.03,
    max_ratio=0.4,
    min_scar_voxels=50,
    min_component_voxels=5,
    myo_dilation_iter=5,
    lv_dilation_iter=5,
    max_reduce_fraction=0.1,
    require_between_lv_myo=False,
):
    """
    Conservative 3D scar post-processing for CMR.

    Anatomical assumption:
        scar should be between / close to both LV cavity and LV myocardium.

    Removed scar is converted to myocardium, not background.

    Labels:
        1 = LV cavity
        2 = LV myocardium
        3 = myocardial scar
        4 = RV cavity
    """

    out = pred.copy()

    lv = out == lv_label
    myo = out == myo_label
    scar = out == scar_label

    scar_voxels = int(scar.sum())
    myo_total = myo | scar
    myo_total_voxels = int(myo_total.sum())

    if scar_voxels == 0:
        return out

    if myo_total_voxels == 0:
        out[scar] = myo_label
        return out

    scar_ratio = scar_voxels / myo_total_voxels

    # --------------------------------------------------
    # Case 1: tiny false-positive scar
    # --------------------------------------------------
    if scar_voxels < min_scar_voxels or scar_ratio < min_ratio:
        out[scar] = myo_label
        return out

    # --------------------------------------------------
    # Step 1: anatomical validity
    # Scar should be near both LV cavity and myocardium
    # --------------------------------------------------
    lv_region = binary_dilation(lv, iterations=lv_dilation_iter)
    myo_region = binary_dilation(myo, iterations=myo_dilation_iter)

    if require_between_lv_myo:
        valid_scar = scar & lv_region & myo_region
    else:
        valid_scar = scar & (lv_region | myo_region)

    # Safety: if this removes too much, keep original scar
    # This protects true scar patients.
    if valid_scar.sum() < 0.5 * scar_voxels:
        valid_scar = scar

    # --------------------------------------------------
    # Step 2: remove tiny 3D scar components
    # --------------------------------------------------
    structure = generate_binary_structure(3, 2)
    cc, num = label(valid_scar, structure=structure)

    cleaned_scar = np.zeros_like(valid_scar, dtype=bool)

    for i in range(1, num + 1):
        comp = cc == i
        comp_size = int(comp.sum())

        if comp_size >= min_component_voxels:
            cleaned_scar |= comp

    # Safety: if component filtering removes too much, keep original scar
    if cleaned_scar.sum() < 0.5 * scar_voxels:
        cleaned_scar = scar

    cleaned_ratio = cleaned_scar.sum() / myo_total_voxels

    # --------------------------------------------------
    # Step 3: only reduce if scar is extremely too much
    # --------------------------------------------------
    if cleaned_ratio > max_ratio and prob is not None:
        target_keep_voxels = int(max_ratio * myo_total_voxels)

        # Do not remove more than max_reduce_fraction of current scar
        min_keep_voxels = int((1.0 - max_reduce_fraction) * cleaned_scar.sum())
        target_keep_voxels = max(target_keep_voxels, min_keep_voxels)

        # Extract scar probability
        if prob.shape[0] > scar_label and prob.shape[1:] == pred.shape:
            scar_prob = prob[scar_label]
        elif prob.shape[-1] > scar_label and prob.shape[:-1] == pred.shape:
            scar_prob = prob[..., scar_label]
        else:
            raise ValueError("prob shape must be [C, ...] or [..., C] matching pred")

        candidate_idx = np.argwhere(cleaned_scar)
        candidate_probs = scar_prob[cleaned_scar]

        order = np.argsort(candidate_probs)[::-1]
        keep_idx = candidate_idx[order[:target_keep_voxels]]

        reduced_scar = np.zeros_like(cleaned_scar, dtype=bool)
        reduced_scar[tuple(keep_idx.T)] = True
        cleaned_scar = reduced_scar

    # --------------------------------------------------
    # Final update:
    # removed scar -> myocardium
    # kept scar remains scar
    # --------------------------------------------------
    out[out == scar_label] = myo_label
    out[cleaned_scar] = scar_label

    return out


# new_pred = postprocess_multiclass_volume(pred, min_size=500, hole_area=50, smooth_radius=1, keep_largest=True)

# print(f"Dice Score: {dice_score(new_pred, mask, class_id=1)}")
# print(f"Dice Score: {dice_score(new_pred, mask, class_id=2)}")
# print(f"Dice Score: {dice_score(new_pred, mask, class_id=3)}")
