import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


class FullPad:
    """Pad image to exact target dimensions using corner-sampled background colour."""

    def __init__(self, target_width: int, target_height: int):
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        s = image.size()
        width = s[-1]
        height = s[-2]

        if width > self.target_width and height > self.target_height:
            return image

        horizontal_offset = int((self.target_width - width) / 2)
        vertical_offset = int((self.target_height - height) / 2)

        np_img = np.asarray(image)
        top_left = np_img[0][0][0]
        bottom_left = np_img[0][-1][0]
        top_right = np_img[0][0][-1]
        bottom_right = np_img[0][-1][-1]
        avg_bg = (top_left.item() + bottom_left.item() + top_right.item() + bottom_right.item()) / 4

        padding = (horizontal_offset, vertical_offset)
        return F.pad(image, padding, avg_bg, "constant")


class SquarePad:
    """Pad image to square using corner-sampled background colour."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        s = image.size()
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)

        top_left = image[0][0][0]
        bottom_left = image[0][-1][0]
        top_right = image[0][0][-1]
        bottom_right = image[0][-1][-1]
        avg_bg = (top_left.item() + bottom_left.item() + top_right.item() + bottom_right.item()) / 4

        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, avg_bg, "constant")


class ReflectPad(torch.nn.Module):
    """Pad image using reflection, then resize to target dimensions."""

    def __init__(self, target_width: int = 299, target_height: int = 299):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        s = image.size()
        width = s[-1]
        height = s[-2]

        if width > self.target_width and height > self.target_height:
            return image

        horizontal_offset = max(int((self.target_width - width) / 2), 0)
        vertical_offset = max(int((self.target_height - height) / 2), 0)

        numpy_image = image.numpy()
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        cv2_image = cv2.copyMakeBorder(
            cv2_image, vertical_offset, vertical_offset, horizontal_offset, horizontal_offset, cv2.BORDER_REFLECT
        )
        cv2_image = cv2.resize(cv2_image, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
        new_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        new_image = np.transpose(new_image, (2, 0, 1))
        return torch.from_numpy(new_image)
