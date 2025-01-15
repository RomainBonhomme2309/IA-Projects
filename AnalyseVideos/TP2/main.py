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
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker

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

def plot_results(train_loss, val_loss, train_acc, val_acc, model_name):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label=r"Training Loss")
    plt.plot(val_loss, label=r"Validation Loss")
    plt.xlabel(r"Epochs")
    plt.ylabel(r"Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label=r"Training Accuracy")
    plt.plot(val_acc, label=r"Validation Accuracy")
    plt.xlabel(r"Epochs")
    plt.ylabel(r"Accuracy")
    plt.legend()

    plt.suptitle(rf"Training and Validation Results for {model_name}")

    plt.savefig(f"results_{model_name}.pdf")

    plt.show()

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

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = [(models.EfficientNetClassifier(num_classes=9).to(device), "EfficientNet")]

    criterion = nn.CrossEntropyLoss()
    num_epochs = 20

    for model, model_name in models_list:
        tracker = OfflineEmissionsTracker(country_iso_code="FRA", region="Nouvelle-Aquitaine", output_dir=f"carbon_logs_{model_name}")
        tracker.start()

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_accuracy = 0.0

        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_accuracy = train.train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = train.evaluate(model, test_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), f"best_model_{model_name}.pth")
                print(f"Model saved as 'best_model_{model_name}.pth'!")

        tracker.stop()

        plot_results(train_losses, val_losses, train_acc, val_acc, model_name)      

        print(f"Training complete for model {model_name}!")
