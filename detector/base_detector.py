import cv2
import numpy as np
import torch
import time

from mmdet.datasets.pipelines import Compose

from modify_mmdet_utils.image import show_image


class BaseDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.transform = Compose([
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
        ])
        self.transform_q = Compose([
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='my_DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes_q']),
        ])

    @torch.no_grad()
    def init(self, img, bbox):
        self.model.eval()

        item = self.pre_item(img, bbox)
        item = self.transform_q(item)
        img = item['img'].data.unsqueeze(0).to(self.device)
        bboxes = item['gt_bboxes_q'].data.to(self.device)
        self.model._process_query(img, [bboxes])

    @torch.no_grad()
    def update(self, img, **kwargs):
        self.model.eval()

        item = self.pre_item(img, None)
        item = self.transform(item)
        img = item['img'].data.unsqueeze(0).to(self.device)
        img_meta = item['img_metas'].data

        torch.cuda.synchronize()
        begin = time.time()
        results = self.model._process_gallary(img, [img_meta], rescale=True, **kwargs)
        torch.cuda.synchronize()
        dur_time = time.time() - begin
        return dur_time, results

    def forward_test(self, img_files, init_bbox, visualize=False, thr=0.1):
        frame_num = len(img_files)
        all_boxes = []

        total_time = 0
        for f, img_file in enumerate(img_files):
            img = cv2.imread(img_file)
            img = img[:, :, ::-1].copy()
            if f == 0:
                self.init(img, init_bbox)

            dur_time, all_box = self.update(img)
            total_time += dur_time
            all_box = all_box[all_box[:, -1] > thr]

            if visualize:
                show_image(img, all_box[:, :-1], thickness=1)
            all_boxes.append(all_box)

        return frame_num/total_time, all_boxes

    def pre_item(self, img, bbox):
        item = {
            'img': img,
            'ori_shape': img.shape,
            'filename': None,
            'ori_filename': None,
            'flip': False,
            'flip_direction': None,
            'img_fields': ['img'],
        }
        if bbox is not None:
            item.update({
                'gt_bboxes_q': bbox,
                'bbox_fields': ['gt_bboxes_q'],
            })
        return item
