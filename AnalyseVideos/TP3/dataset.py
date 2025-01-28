import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)

        label = match_label_to_index(label)

        return image, label

def match_label_to_index(label):
    labels = {
        "Bowl": 0,
        "CanOfCocaCola": 1,
        "Jam": 2,
        "MilkBottle": 3,
        "Mug": 4,
        "OilBottle": 5,
        "Rice": 6,
        "Sugar": 7,
        "VinegarBottle": 8
    }

    return labels[label]

def create_dataset(root_dir, train=True):
    if train:
        root_dir = os.path.join(root_dir, "train")
    else:
        root_dir = os.path.join(root_dir, "test")

    dataset = []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if os.path.isdir(class_path):
            for subfolder in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder)

                for root, dirs, files in os.walk(subfolder_path):
                    if "CroppedFrames" in root:
                        for file in files:
                            if file.endswith(".png"):
                                image_path = os.path.join(root, file)
                                dataset.append((image_path, class_name))

    return dataset