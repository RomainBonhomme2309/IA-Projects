import os
import frame_extractor
import dataset
import models
import train
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker
import gradcam

labels_list = [
    "Bowl",
    "CanOfCocaCola",
    "Jam",
    "MilkBottle",
    "Mug",
    "OilBottle",
    "Rice",
    "Sugar",
    "VinegarBottle",
]


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

    plt.savefig(os.path.join("results", f"results_{model_name}.pdf"))

    plt.show()


if __name__ == "__main__":
    ### Uncomment the following line to extract frames from videos (this is a long process)
    # extract_png_frames(os.path.join("datasets", "GITW_light"))

    dataset_directory = os.path.join("datasets", "GITW_light")
    train_dataset_list = dataset.create_dataset(dataset_directory, train=True)
    test_dataset_list = dataset.create_dataset(dataset_directory, train=False)

    train_transform = transforms.Compose(
        [
            # Data augmentation
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )

    train_dataset = dataset.ImageDataset(train_dataset_list, transform=train_transform)
    test_dataset = dataset.ImageDataset(test_dataset_list, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = [
        (models.ResNetClassifier(num_classes=9).to(device), "ResNet18"),
        (models.EfficientNetClassifier(num_classes=9).to(device), "EfficientNet"),
    ]

    criterion = nn.CrossEntropyLoss()
    num_epochs = 30

    for model, model_name in models_list:
        tracker = OfflineEmissionsTracker(
            country_iso_code="FRA",
            region="Nouvelle-Aquitaine",
            output_dir=f"carbon_logs_{model_name}",
        )
        tracker.start()

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_accuracy = 0.0

        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_accuracy = train.train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = train.evaluate(
                model, test_loader, criterion, device
            )
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

        print("Generating Grad-CAM visualizations...")

        if model_name == "ResNet18":
            target_layer = model.base_model.layer4[
                -1
            ]  # Last convolutional layer of ResNet
        elif model_name == "EfficientNet":
            target_layer = model.base_model.features[
                -1
            ]  # Last convolutional layer of EfficientNet
        grad_cam = gradcam.GradCAM(model, target_layer)

        images_per_class = {}

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            for i in range(len(labels)):
                class_idx = labels[i].item()
                if class_idx not in images_per_class:
                    images_per_class[class_idx] = inputs[i].unsqueeze(0)
                if len(images_per_class) == 9:
                    break
            if len(images_per_class) == 9:
                break

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()

        for idx, (class_idx, img_tensor) in enumerate(images_per_class.items()):
            cam = grad_cam.generate(img_tensor, class_idx=class_idx)
            original_img = img_tensor[0].cpu()
            cam_overlay = gradcam.overlay_cam_on_image(original_img, cam)

            axes[idx].imshow(cam_overlay)
            axes[idx].set_title(f"{labels_list[class_idx]}")
            axes[idx].axis("off")

        plt.savefig(os.path.join("results", f"gradcam_{model_name}.pdf"))
        plt.show()
