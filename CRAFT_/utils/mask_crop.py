import time
from PIL import Image
import math
from math import *
import numpy as np
import cv2
from PIL import Image, ImageOps

from collections import OrderedDict
from scipy.spatial import distance as dist


def SortPoint(points):
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]
    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]
    return sp


def order_coord(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = x_sorted[:2, :]
    rightMost = x_sorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, bl, br], dtype="float32")


def extendLength(sp, ymax, xmax, offset=0):
    width = np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2)
    height = np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2)

    z = np.sqrt((width ** 2) + height ** 2)
    y_scale = offset / z
    x_scale = np.sqrt((offset ** 2) - y_scale ** 2)
    scale = np.asarray([x_scale, y_scale])
    sp = np.asarray(sp)
    sp[[0, 2]] -= scale
    sp[[1, 3]] += scale
    sp[:, 0] = sp[:, 0].clip(min=0, max=xmax - 1)
    sp[:, 1] = sp[:, 1].clip(min=0, max=ymax - 1)
    return list(sp)


def imgWrapAffine(src, coord):

    points = np.zeros((4, 2), dtype="float32")
    points[0] = (coord[0][0], coord[0][1])
    points[1] = (coord[1][0], coord[1][1])
    points[2] = (coord[3][0], coord[3][1])
    points[3] = (coord[2][0], coord[2][1])

    # sp = SortPoint(points)
    sp = points
    width = int(np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))
    dstrect = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype="float32")
    sp = np.array(sp)
    transform = cv2.getPerspectiveTransform(sp, dstrect)

    warpedimg = cv2.warpPerspective(src, transform, (width, height))

    return warpedimg


def crop_mask(image, masks, idxx):
    result = OrderedDict()
    for i in range(masks.shape[0]):
        # opencv4 contours, _
        _, contours, _ = cv2.findContours((255 * masks[i, :, :]).astype(np.uint8).copy(), 1, 1)
        rect_list = []
        if len(contours) != 1:
            for j in range(len(contours)):
                rect = cv2.minAreaRect(contours[j])
                rect_list.append(rect[1][0])
            idx = rect_list.index(max(rect_list))
            rect = cv2.minAreaRect(contours[idx])
            _, _, ang = cv2.minAreaRect(contours[0])
        else:
            rect = cv2.minAreaRect(contours[0])
            _, _, ang = cv2.minAreaRect(contours[0])

        box = cv2.boxPoints(rect)

        box_init = np.int0(box)
        wrap_img = imgWrapAffine(image.copy(), box_init, idxx)

        H, W, _ = wrap_img.shape

 

        if int(H/W) > 2:
            wrap_img = np.rot90(wrap_img)

    return wrap_img, box
