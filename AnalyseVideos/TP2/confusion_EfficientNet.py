import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import models
import os
import dataset


dataset_directory = os.path.join("datasets", "GITW_light")
test_dataset_list = dataset.create_dataset(dataset_directory, train=False)

labels_list = ["Bowl",
    "CanOfCocaCola",
    "Jam",
    "MilkBottle",
    "Mug",
    "OilBottle",
    "Rice",
    "Sugar",
    "VinegarBottle"]

transform = transforms.Compose([
    transforms.Resize((566, 566)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = dataset.ImageDataset(test_dataset_list, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.EfficientNetClassifier(num_classes=9).to(device)
model.load_state_dict(torch.load("best_model_EfficientNet.pth", map_location=device))
model.eval()

# Gather predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_list)

plt.figure(figsize=(12, 12))
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix EfficientNet")
plt.savefig("confusion_matrix_EfficientNet.pdf")
plt.show()