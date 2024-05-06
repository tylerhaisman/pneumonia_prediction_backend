# sorting viral and bacterial pneumonia

import os
import shutil


def sort_directories():
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
                        shutil.move(
                            os.path.join(root, file), os.path.join(viral_dir, file)
                        )
                    else:
                        continue

            # delete the /PNEUMONIA directory if empty
            if not os.listdir(original_dataset_dir):
                os.rmdir(original_dataset_dir)


def count_directories():
    directories = ["train"]

    for directory in directories:
        bacterial_dir = "./x-ray_data/" + directory + "/BACTERIAL"
        viral_dir = "./x-ray_data/" + directory + "/VIRAL"
        normal_dir = "./x-ray_data/" + directory + "/NORMAL"

        bacterial_counter = 0
        viral_counter = 0
        normal_counter = 0

        for root, dirs, files in os.walk(bacterial_dir):
            for file in files:
                bacterial_counter = bacterial_counter + 1

        for root, dirs, files in os.walk(viral_dir):
            for file in files:
                viral_counter = viral_counter + 1

        for root, dirs, files in os.walk(normal_dir):
            for file in files:
                normal_counter = normal_counter + 1

        print("bacterial_counter:", bacterial_counter)
        print("viral_counter:", viral_counter)
        print("normal_counter:", normal_counter)
