import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("core")

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


import numpy as np
import torch
import torch.nn.functional as F


def read_flo_file(filename):
    """Read a .flo optical flow file."""
    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: incorrect magic number ({magic}).")

        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # Read the flow data
        data = np.fromfile(f, np.float32, count=2 * width * height)
        data = data.reshape((height, width, 2))  # Shape: (H, W, 2)

    # Convert to tensor (shape: H, W, 2)
    flow_tensor = torch.tensor(data).permute(2, 0, 1)  # Shape: (2, H, W)

    # Redimensionner pour obtenir la taille (2, 440, 1024)
    target_height = 440
    target_width = 1024
    flow_rescaled = F.interpolate(
        flow_tensor.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Retourner le flux redimensionné sans la dimension batch
    return flow_rescaled.squeeze(0)  # Shape: (2, 440, 1024)


def viz(img, flo, flow_gt):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flow_gt = flow_gt.permute(1, 2, 0).cpu().numpy()

    # Map flow to RGB image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    img_flo_gt = np.concatenate([img_flo, flow_viz.flow_to_image(flow_gt)], axis=0)

    cv2.imshow("image", img_flo_gt[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def compute_metrics(flow_estimated, flow_gt):
    """
    Calculate aEPE, aAE, and MSE for two optical flow images.

    Args:
        flow_estimated: Estimated optical flow (tensor of shape (2, H, W)).
        flow_gt: Ground truth optical flow (tensor of shape (2, H, W)).

    Returns:
        aEPE, aAE, MSE (float): Calculated metric values.
    """
    # Vérification que les dimensions sont compatibles
    if flow_estimated.shape != flow_gt.shape:
        raise ValueError("flow_estimated and flow_gt must have the same shape")

    # Calcul de l'erreur de point de fin moyen (aEPE)
    epe = torch.norm(
        flow_estimated - flow_gt, dim=0
    )  # Norme Euclidienne sur la première dimension
    aEPE = epe.mean().item()  # Moyenne des erreurs

    # Calcul de l'erreur absolue moyenne (aAE)
    aAE = (
        torch.abs(flow_estimated - flow_gt).mean().item()
    )  # Moyenne des erreurs absolues

    # Calcul de l'erreur quadratique moyenne (MSE)
    mse = torch.mean(
        (flow_estimated - flow_gt) ** 2
    ).item()  # Moyenne des carrés des erreurs

    return aEPE, aAE, mse


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    aEPEs = []
    aAEs = []
    MSEs = []

    with torch.no_grad():
        images = sorted(
            glob.glob(os.path.join(args.path, "*.png"))
            + glob.glob(os.path.join(args.path, "*.jpg"))
        )
        flows = sorted(glob.glob(os.path.join(args.path_flo, "*.flo")))

        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            flow_data = read_flo_file(flows[idx])
            flow_gt = flow_data.float().to(DEVICE)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            if idx == len(images) // 2:
                viz(image1, flow_up, flow_gt)

            flow_up = flow_up.squeeze()

            aEPE, aAE, MSE = compute_metrics(flow_up, flow_gt)
            aEPEs.append(aEPE)
            aAEs.append(aAE)
            MSEs.append(MSE)

    print("aEPE:", np.mean(aEPEs))
    print("aAE:", np.mean(aAEs))
    print("MSE:", np.mean(MSEs))

    # Ploting
    plt.plot(aEPEs, label="aEPE")
    plt.plot(aAEs, label="aAE")
    plt.plot(MSEs, label="MSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--path_flo", help="path to .flo files")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficient correlation implementation",
    )
    args = parser.parse_args()

    demo(args)
