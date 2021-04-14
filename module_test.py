import cv2
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.datasets import LoadDataUnlabeled


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


if __name__ == '__main__':
    img = Image.open('ipsc_yolo_17/images/val2020/17d_5_89.jpg')
    print(img.size)
    weak_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
    ])
    strong_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=1.0),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomApply([transforms.GaussianBlur(7, [0.1, 2.0])], p=1.0),
    ])
    # prepare image and figure
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # img.show()
    # plt.show()
    weak_img = weak_transform(img)
    # ax1.imshow(img)
    plt.figure(1)
    plt.imshow(weak_img)
    plt.show()

    strong_img = strong_transform(weak_img)
    plt.figure(2)
    plt.imshow(strong_img)
    plt.show()

    # tensor_transform =  transforms.ToTensor()
    img = np.array(img)
    img = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )
    # img_t = tensor_transform(img)
    print('test done!')