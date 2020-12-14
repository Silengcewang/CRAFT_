"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import torch
from skimage import io


def img_resize(image, imgH, imgW):
    h, w = image.shape[0], image.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(image, (resized_w, imgH), interpolation=cv2.INTER_LINEAR)
    padding_im = np.zeros((imgH, imgW, 3), dtype=np.float32)
    # padding_im[:, 0:resized_w] = resized_image
    padding_im[:, int((imgW-resized_w)/2):resized_w + int((imgW-resized_w)/2)] = resized_image
    return padding_im


def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:, :, :3]
    img = np.array(img)

    return img


def normalizeMeanVariance(in_img):
    # should be RGB order
    mean = (0.485, 0.456, 0.406)
    variance = (0.229, 0.224, 0.225)
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(imgs, target_h, target_w, set_batch=True):
    if set_batch:
        img_list, img_ratio, img_src = [], [], []
        for img in imgs:
            img_src.append(img)
            h, w = img.shape[:2]
            h_ratio, w_ratio = h / target_h, w / target_w
            # resized = img_resize(img, target_h, target_w)
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            mean = (0.485, 0.456, 0.406)
            variance = (0.229, 0.224, 0.225)
            img = resized.copy().astype(np.float32)
            img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
            img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

            # cv2.imshow('1', img)
            # cv2.waitKey(0)

            img_list.append(img)
            img_ratio.append((h_ratio, w_ratio))

        img_list = np.asarray(img_list, dtype=np.float32)
        img_list = np.transpose(img_list, (0, 3, 1, 2))
        return img_list, img_src, img_ratio
    else:
        for img in imgs:
            mag_ratio = 1.5
            square_size = 1280
            image_src = img.copy()
            height, width, channel = img.shape
            # magnify image size
            target_size = mag_ratio * max(height, width)
            # set original image size
            if target_size > square_size:
                target_size = square_size
            ratio = target_size / max(height, width)
            target_h, target_w = int(height * ratio), int(width * ratio)
            proc = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # make canvas and paste image
            target_h32, target_w32 = target_h, target_w
            if target_h % 32 != 0:
                target_h32 = target_h + (32 - target_h % 32)
            if target_w % 32 != 0:
                target_w32 = target_w + (32 - target_w % 32)
            resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
            resized[0:target_h, 0:target_w, :] = proc

            mean = (0.485, 0.456, 0.406)
            variance = (0.229, 0.224, 0.225)

            img = resized.copy().astype(np.float32)
            img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
            resized /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

            img_list = np.asarray([resized], dtype=np.float32)
            img_list = np.transpose(img_list, (0, 3, 1, 2))
            ratio_h = ratio_w = 1 / ratio
            return img_list, [image_src], [[ratio_h, ratio_w]]


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
