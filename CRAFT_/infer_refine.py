# -*- coding: utf-8 -*-
import glob
import os
import time

import cv2
import numpy as np
import torch
from skimage import io

from models.model_adv import CRAFT_adv
from utils import file_utils, craft_utils, imgproc, mask_crop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
list(map(os.remove, glob.glob('./result/*.*')))

class TextDetection:
    def __init__(self, weight):
        self.net = CRAFT_adv()
        dict_a = {k.replace("module.", ""): v for k, v in torch.load(weight, map_location='cpu').items()}
        dict_b = {k.replace("module.", ""): v for k, v in
                  torch.load('weights/craft_refiner_CTW1500.pth', map_location='cpu').items()}
        dict_a.update(dict_b)
        self.net.load_state_dict(dict_a)
        self.net = self.net.to(device)
        self.net.eval()

        self.text_threshold = 0.5
        self.link_threshold = 0.7
        self.low_text_score = 0.4
        self.poly = False

    def detect(self, image, set_batch):
        img_resized, img_srcs, img_ratio = imgproc.resize_aspect_ratio(image, 512, 512, set_batch=set_batch)
        img_resized = torch.from_numpy(img_resized)
        with torch.no_grad():
            # torch.onnx.export(self.net, img_resized, "craft_dynamic.onnx", verbose=True, input_names=['input'], opset_version=12,
            #                   output_names=['output1', 'output2'],
            #                   dynamic_axes={"input": {0: "-1", 2: "-1", 3: "-1"}, "output1": {0: "-1"}, "output2": {0: "-1"}})
            y_refiner, score_text = self.net(img_resized)
        score_links = y_refiner[:, 0, :, :].cpu().data.numpy()
        score_texts = score_text[:, 0, :, :].cpu().data.numpy()

        polys_list = []
        for idx, score_link in enumerate(score_links):
            boxes, polys = craft_utils.getDetBoxes(score_texts[idx],
                                                   score_link,
                                                   self.text_threshold,
                                                   self.link_threshold,
                                                   self.low_text_score,
                                                   poly=self.poly)

            for k in range(len(polys)):
                if polys[k] is None:
                    polys[k] = boxes[k]
            polys_list.append(polys)

        point_list2 = []
        for idx, image_src in enumerate(img_srcs):
            img = np.array(image_src[:, :, ::-1])
            with open('./result/' + "gt_" + str(idx) + '.txt', 'w') as f:
                for i, box in enumerate(polys_list[idx]):
                    box[:, 0] *= img_ratio[idx][1] * 2
                    box[:, 1] *= img_ratio[idx][0] * 2
                    # warp_img = mask_crop.imgWrapAffine(img, np.int0(box))
                    # cv2.imwrite(''.join(['./result/warp_img_', str(i), '.jpg']), warp_img)
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    point_list2.append(','.join([str(p) for p in poly]) + ',1\n')
                    f.write(','.join([str(p) for p in poly]) + ',1\n')
                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(''.join(['./result/', str(time.time()), '.jpg']), img)


if __name__ == '__main__':
    TextDetect = TextDetection('weights/craft_mlt_25k.pth')
    set_batch = False

    s = time.time()
    imgs = [io.imread(x, pilmode='RGB') for x in file_utils.get_files('imgs')[0]]
    if set_batch:
        TextDetect.detect(imgs, set_batch)
    else:
        for img in imgs:
            TextDetect.detect([img], set_batch)
    print(time.time() - s)
