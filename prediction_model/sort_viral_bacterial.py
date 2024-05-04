# sorting viral and bacterial pneumonia

import os
import shutil

directories = ["train", "val", "test"]

for directory in directories:
    original_dataset_dir = "./x-ray_data/" + directory + "/PNEUMONIA"
    bacterial_dir = "./x-ray_data/" + directory + "/BACTERIAL"
    viral_dir = "./x-ray_data/" + directory + "/VIRAL"

    os.makedirs(bacterial_dir, exist_ok=True)
    os.makedirs(viral_dir, exist_ok=True)

    for root, dirs, files in os.walk(original_dataset_dir):
        for file in files:
            if file.endswith(".jpeg"):
                if "bacteria" in file:
                    shutil.move(
                        os.path.join(root, file), os.path.join(bacterial_dir, file)
                    )
                elif "virus" in file:
                    shutil.move(os.path.join(root, file), os.path.join(viral_dir, file))
                else:
                    continue

        # delete the /PNEUMONIA directory if empty
        if not os.listdir(original_dataset_dir):
            os.rmdir(original_dataset_dir)
