import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Dataset personnalisé pour PyTorch.

        Args:
            dataset (list): Liste de tuples (chemin_vers_image, label).
            transform (callable, optional): Transformation à appliquer aux images.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        # Charger l'image avec PIL
        image = Image.open(image_path)
        
        # Appliquer les transformations si disponibles
        if self.transform:
            image = self.transform(image)

        return image, label

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

root_directory = "/home/rbonhomme1/Documents/IA-Projects/AnalyseVideos/TP2/datasets/GITW_light"
train_dataset_list = create_dataset(root_directory, train=True)
test_dataset_list = create_dataset(root_directory, train=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = ImageDataset(train_dataset_list, transform=transform)
test_dataset = ImageDataset(test_dataset_list, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for batch in train_loader:
    images, labels = batch
    print(images.shape, labels)
    break
