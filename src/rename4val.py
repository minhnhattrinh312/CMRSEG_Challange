import os
from natsort import natsorted

output_dir = "./output"

for task in os.listdir(output_dir):
    task_path = os.path.join(output_dir, task)
    if os.path.isdir(task_path):
        for cmr_view in os.listdir(task_path):
            cmr_view_path = os.path.join(task_path, cmr_view)
            count = 1
            if os.path.isdir(cmr_view_path):
                for file in natsorted(os.listdir(cmr_view_path)):
                    if file.endswith(".nii.gz"):
                        old_path = os.path.join(cmr_view_path, file)
                        new_filename = f"{os.path.basename(file).split('.')[0][:-3]}{count:03d}.nii.gz"
                        new_path = os.path.join(cmr_view_path, new_filename)
                        os.rename(old_path, new_path)
                        count += 1

# read json file
import json

for task in os.listdir(output_dir):
    task_path = os.path.join(output_dir, task)
    for json_file in os.listdir(task_path):
        if json_file.endswith(".json"):
            json_path = os.path.join(task_path, json_file)
            with open(json_path, "r") as f:
                predictions = json.load(f)
                # rename keys to match new patient names
                new_predictions = {}
                count = 1
                for key, value in predictions.items():
                    new_key = f"{key[:-3]}{count:03d}"
                    new_predictions[new_key] = value
                    count += 1
            # save new json file with the same path
            with open(json_path, "w") as f:
                json.dump(new_predictions, f, indent=4)
