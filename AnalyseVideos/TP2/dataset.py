import torch
import os


def create_dataset(root_dir, train=True):
    if train:
        root_dir = os.path.join(root_dir, "train")
    else:
        root_dir = os.path.join(root_dir, "test")

    dataset = {}

    print(list(os.walk(root_dir)))
    for root, dirs, files in os.walk(root_dir):
        print(dirs)
        for dir in dirs:
            if dir.startswith("CroppedFrames"):
                dataset[label] = []
                frames_dir = os.path.join(root_dir, dir)
                for file in os.listdir(frames_dir):
                    if file.endswith(".png"):
                        dataset[label].add(os.path.join(frames_dir, file))
