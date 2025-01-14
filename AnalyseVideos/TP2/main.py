import os
import frame_extractor
import dataset
import models
import train
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim

def extract_png_frames(video_directory):
    total = 0
    for root, _, files in os.walk(video_directory):
        for file_name in files:
            if file_name.endswith(".mp4"):
                total += 1

    print(f"Found {total} videos in '{video_directory}'.")

    count = 1
    for root, _, files in os.walk(video_directory):
        for file_name in files:
            if file_name.endswith(".mp4"):
                video_file = os.path.join(root, file_name)
                print(f"Processing: {video_file} ({count}/{total})")
                frame_extractor.extract_frames(video_file)
                count += 1

if __name__ == "__main__":
    ### Uncomment the following line to extract frames from videos (this is a long process)
    # extract_png_frames(os.path.join("datasets", "GITW_light"))

    dataset_directory = os.path.join("datasets", "GITW_light")
    train_dataset_list = dataset.create_dataset(dataset_directory, train=True)
    test_dataset_list = dataset.create_dataset(dataset_directory, train=False)

    transform = transforms.Compose([
        transforms.Resize((566, 566)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = dataset.ImageDataset(train_dataset_list, transform=transform)
    test_dataset = dataset.ImageDataset(test_dataset_list, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.ResNetClassifier(num_classes=9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.ResNetClassifier(num_classes=9).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = train.evaluate(model, test_loader, criterion, device)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved as 'best_model.pth'!")

    print("Training complete!")