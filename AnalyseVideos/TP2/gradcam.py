import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))
        cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)
        cam = np.maximum(cam, 0)
        cam = cam[0]

        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        return cam


def overlay_cam_on_image(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap + np.float32(img.permute(1, 2, 0).cpu().numpy())
    superimposed_img = np.clip(superimposed_img / np.max(superimposed_img), 0, 1)
    return np.uint8(255 * superimposed_img)
