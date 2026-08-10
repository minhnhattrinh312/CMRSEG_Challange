# CMRSEG Challenge

Multi-view 2D cardiac MRI segmentation pipeline for CMR-MULTI tasks:
- Task 1 (CINE): segment structures and estimate LV ejection fraction (EF).
- Task 2 (LGE): segment structures and estimate scar mass.

The repository includes:
- data preprocessing from NIfTI to NPZ slices,
- 5-fold training for CINE and LGE models,
- ensemble inference,
- post-processing and challenge output formatting.


## Quick Setup (Local)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
	torch lightning hydra-core natsort nibabel numpy pandas \
	scikit-image scipy tqdm kornia wandb
```

If you do not want Weights & Biases logging, set `LOG_WANDB: False` in `config/train_config.yaml`.

## Data Layout

This repository expects these key folders:

- Training data (for preprocessing/training):
	- `data/CMR-MULTI/...`
- Preprocessed training NPZ + split CSV:
	- `data/CMR-MULTI_npz/...`
	- `data/csv_files/...`
- Inference input (challenge style):
	- `input/CMR-MULTI/...`

Folder names are important because scripts discover data by convention.

## Configuration

Main configs:
- `config/data_config.yaml`: per-task/view class counts, paths, and class weights.
- `config/train_config.yaml`: optimizer/training/inference settings.

Important keys in `train_config.yaml`:
- `PREDICT_FOLDER`: `"VAL"` or `"TST"` (controls which folders are inferred).
- `LOG_WANDB`: enable/disable wandb.
- `ACCELERATOR`, `DEVICES`, `PRECISION`: hardware behavior.

## Preprocessing (Training Data)

Convert NIfTI training data to NPZ slices and generate 5-fold CSV split files:

```bash
python src/data_preprocessing/CMR2npz.py
```

This writes:
- NPZ files under `data/CMR-MULTI_npz/<CMR_TYPE>/<VIEW>/`
- CSV metadata under `data/csv_files/*.csv`

## Training

Train CINE models (5 folds):

```bash
python src/train/train_CINE.py
```

Train LGE models (5 folds):

```bash
python src/train/train_LGE.py
```

Checkpoints are saved under:
- `saved_models/CINE_fold1 ... CINE_fold5`
- `saved_models/LGE_fold1 ... LGE_fold5`

## Inference and Submission Outputs

Run the full pipeline:

```bash
bash src/run_submission.sh
```


Generated outputs:
- Segmentation masks:
	- `output/task1_cine/{2CH,4CH,SAX}/*.nii.gz`
	- `output/task2_lge/{2CH,4CH,RAS,SAX}/*.nii.gz`
- Metrics:
	- `output/task1_cine/ef_predictions.json`
	- `output/task2_lge/mass_predictions.json`

## Docker Usage

Build image:

```bash
docker build -t cmrseg-challenge .
```

Run with mounted challenge input/output:

```bash
docker run --rm \
	-v "$(pwd)/input:/input" \
	-v "$(pwd)/output:/output" \
	cmrseg-challenge
```

The container entrypoint runs `src/run_submission.sh` and uses:
- `INPUT_DIR=/input`
- `OUTPUT_DIR=/output`

## Notes and Troubleshooting

- Ensure `saved_models/` contains the expected checkpoint files referenced in `src/predict.py`.
- If running on CPU only, inference/training will be much slower.
- If wandb is not configured and training fails, disable it by setting `LOG_WANDB: False`.
- Naming in `output/` is normalized by `src/rename4val.py`; do not skip this step for challenge-style outputs.

## License

See `LICENSE`.