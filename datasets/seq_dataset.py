import json
import os
import os.path as osp
import random
import shutil

import torch

import cv2
import numpy as np
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC
import lap


@DATASETS.register_module()
class SeqDataset(Dataset):

    def __init__(self, root_dir=None, subset='train', classes=('object',), training=True,
                 train_pipeline=None, test_pipeline=None, box_refine=False, upper_thr=0.6, update_w=0.5, match_thr=0.4,
                 seq_len=3, version=1, **kwargs):
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'
        assert classes == ('object',) or classes == ('object', 'bg_object')
        if root_dir is None:
            root_dir = osp.expanduser('./data/FSC147/FSC147_384_V2')

        self.root_dir = root_dir
        self.subset = subset
        self.CLASSES = classes
        self.training = training
        self.box_refine = box_refine
        self.upper_thr = upper_thr
        self.update_w = update_w
        self.match_thr = match_thr
        self.version = version

        self.annotations_list = []  # type:list[dict]
        self.cls_dict = {}
        self.id_2_index = {}
        self.load_annotations()

        self.flag = np.zeros(len(self.annotations_list), dtype=np.uint8)
        for i, each in enumerate(self.annotations_list):
            if each['w'] / each['h'] > 1:
                self.flag[i] = 1

        self.train_transforms = Compose(train_pipeline)
        self.test_transforms = Compose(test_pipeline)

        self.save_boxes = False
        self.seq_len = seq_len

    def __getitem__(self, index):
        data = self.annotations_list[index]
        img_x_path, img_x_name, gt_bbox_x, pseudo_data, cls = \
            data["img_path"], data["img_name"], data['gt_bbox'], data['pseudo_data'], data['cls']

        # prepare template
        img_z, img_z_name, bboxes_z, select_neg_query = self.load_query(cls, index)
        item_z = self.pre_item(img_z, bboxes_z, bboxes_z, img_z_name)
        if self.training:
            item_z = self.train_transforms(item_z)
        else:
            item_z['flip'] = False
            item_z['flip_direction'] = None
            item_z = self.test_transforms(item_z)

        # prepare seq
        pseudo_bboxes = np.array(pseudo_data['bboxes'], dtype=np.float32)
        pseudo_confs = np.array(pseudo_data['confs'], dtype=np.float32)

        img_x = cv2.imread(img_x_path)
        img_x1 = img_x[:, :, ::-1].copy()

        gt_bbox_x = np.array(gt_bbox_x, dtype=np.float32)

        shuffle_idx = np.array(range(len(pseudo_bboxes)))
        np.random.shuffle(shuffle_idx)

        pseudo_bboxes = pseudo_bboxes[shuffle_idx]
        gt_bboxes_x = np.array(np.vstack([gt_bbox_x, pseudo_bboxes]), dtype=np.float32)[:200, ...]

        pseudo_confs = pseudo_confs[shuffle_idx]
        if self.box_refine:
            confs = np.array([[c, c, c, c] for c in pseudo_confs], dtype=np.float32)
            gt_bboxes_confs = np.array(np.vstack([np.ones_like(gt_bbox_x), confs]), dtype=np.float32)[:200, ...]
        else:
            gt_bboxes_confs = np.ones_like(gt_bboxes_x)

        if self.training:
            max_h, max_w = 0, 0
            img_x_list = []
            img_meta_x_list = []
            gt_bboxes_x_list = []
            sample_bboxes_x_list = []

            for i in range(self.seq_len):
                item_x = self.pre_item(img_x1.copy(), gt_bboxes_x.copy(), gt_bbox_x.copy(), img_x_name)
                item_x = self.train_transforms(item_x)
                img_x_list.append(item_x['img'])
                img_meta_x_list.append(item_x['img_metas'])
                h, w = item_x['img_metas'].data['pad_shape'][:2]
                max_h = max(max_h, h)
                max_w = max(max_w, w)
                gt_bboxes_x_list.append(item_x['gt_bboxes'])
                sample_bboxes_x_list.append(item_x['gt_bboxes_q'])
            seq_imgs = torch.zeros((self.seq_len, 3, max_h, max_w))

            for i in range(self.seq_len):
                h, w = img_meta_x_list[i].data['pad_shape'][:2]
                seq_imgs[i, ..., :h, :w] = img_x_list[i].data

            item = {
                'img_x_list': seq_imgs,
                'img_meta_x_list': img_meta_x_list,
                'gt_bboxes_x_list': gt_bboxes_x_list,
                'sample_bboxes_x_list': sample_bboxes_x_list,
                'img_z': item_z['img'],
                'img_meta_z': item_z['img_metas'],
                'sample_bboxes_z': item_z['gt_bboxes_q'],
                'select_neg_query': select_neg_query,
                'gt_bboxes_confs': gt_bboxes_confs,
            }
            _tmp = item['gt_bboxes_x_list'][0].data
        else:
            item_x = self.pre_item(img_x1, gt_bboxes_x, gt_bbox_x, img_x_name)
            item_z = self.pre_item(img_z, bboxes_z, bboxes_z, img_z_name)

            item_x['flip'] = False
            item_x['flip_direction'] = None
            item_x = self.test_transforms(item_x)
            item_z['flip'] = False
            item_z['flip_direction'] = None
            item_z = self.test_transforms(item_z)

            item = {
                'img_x_list': item_x['img'],
                'img_meta_x_list': item_x['img_metas'],
                'img_meta_x': item_x['img_metas'],
                'gt_bboxes_x_list': item_x['gt_bboxes'],
                'sample_bboxes_x_list': item_x['gt_bboxes_q'],
                'img_z': item_z['img'],
                'img_meta_z': item_z['img_metas'],
                'sample_bboxes_z': item_z['gt_bboxes_q'],
                'select_neg_query': select_neg_query,
                'gt_bboxes_confs': gt_bboxes_confs,
            }
            _tmp = item['gt_bboxes_x_list'].data
        if len(self.CLASSES) == 2:
            gt_labels = _tmp.new_zeros(len(_tmp)).long() if not select_neg_query else _tmp.new_ones(len(_tmp)).long()
        else:
            gt_labels = _tmp.new_zeros(len(_tmp)).long()
        item['gt_labels'] = DC(gt_labels)
        return item

    def load_query(self, cls, index):
        classes = [c for c in self.cls_dict.keys()]
        classes.remove(cls)

        p = np.random.random()
        select_neg_query = p > 0.8 and len(self.CLASSES) >= 2 and self.training

        if select_neg_query:
            cls = random.choice(classes)
            index = random.choice(self.cls_dict[cls])
        data = self.annotations_list[index]

        img_path = data["img_path"]
        img_name = data["img_name"]

        img = cv2.imread(img_path)
        img = img[:, :, ::-1].copy()

        sample_boxes = np.array(data['gt_bbox'], dtype=np.float32)
        np.random.shuffle(sample_boxes)

        return img, img_name, sample_boxes, select_neg_query

    def load_annotations(self):
        split_file = os.path.join(self.root_dir, 'Train_Test_Val_FSC_147.json')
        with open(split_file) as f:
            split_data = json.load(f)
        index = 0

        with open(os.path.join(self.root_dir, 'filtered_images.json')) as f:
            valid_image_names = json.load(f)['filtered_images']

        if self.version == 1:
            valid_data = split_data[self.subset]
        elif self.version == 2:
            valid_data = [each for each in split_data[self.subset] if each in valid_image_names]
        else:
            valid_data = valid_image_names

        for img_name in valid_data:
            img_path = os.path.join(self.root_dir, 'images_384_VarV2', img_name)
            anno_path = os.path.join(self.root_dir, 'pseudo_init', img_name.replace('jpg', 'json'))
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            gt_bbox = anno_data['gt_bbox']

            init_confs = [self.upper_thr] * len(anno_data["confs"])
            pseudo_data = {
                'bboxes': anno_data["pseudo_bbox"],
                'confs': init_confs
            }
            cls = anno_data['cls']
            # if cls in self.ignore_classes or img_name in self.ignore_images:
            #     continue
            h = anno_data['height']
            w = anno_data['width']

            annotation = {
                "img_path": img_path,
                "img_name": img_name,
                "gt_bbox": gt_bbox,
                "pseudo_data": pseudo_data,
                "cls": cls,
                "w": w,
                "h": h
            }
            self.annotations_list.append(annotation)

            if cls in self.cls_dict:
                self.cls_dict[cls].append(index)
            else:
                self.cls_dict[cls] = [index]

            self.id_2_index[img_name] = index
            index += 1

    def revise_data(self, new_boxes, img_name, work_dir, epoch):
        new_boxes = new_boxes.copy().astype('float64')
        count = 0

        new_boxes = new_boxes[new_boxes[:, 4] > self.upper_thr]
        out_path = os.path.join(work_dir, 'pseudo_labels', f'pseudo_{epoch}', img_name.replace('jpg', 'json'))
        anno_data = self.annotations_list[self.id_2_index[img_name]]

        if not new_boxes.any():
            if self.save_boxes:
                with open(out_path, 'w') as f:
                    json.dump(anno_data, f, indent=2)
            return count

        pseudo_bboxes = np.array(anno_data['pseudo_data']['bboxes'].copy())
        pseudo_confs = np.array(anno_data['pseudo_data']['confs'].copy())
        overlaps = bbox_overlaps(new_boxes[:, :4], pseudo_bboxes)

        match, unmatched_gt, unmatched_det = linear_assignment(1 - overlaps, 1 - self.match_thr)

        for i, j in match:
            pseudo_box = pseudo_bboxes[j]
            new_box = new_boxes[i]
            iou = overlaps[i][j]
            if pseudo_confs[j] > new_box[4]: continue
            anno_data['pseudo_data']['bboxes'][j] = [self.update_w * a + (1 - self.update_w) * b for a, b in
                                                     zip(pseudo_box, new_box)]
            anno_data['pseudo_data']['confs'][j] = new_box[-1]
            count += 1
        if self.save_boxes:
            with open(out_path, 'w') as f:
                json.dump(anno_data, f, indent=2)
        return count

    def updating_mode(self, epoch, work_dir):
        self.training = False
        path = os.path.join(work_dir, 'pseudo_labels', f'pseudo_{epoch}')
        if os.path.exists(path):
            shutil.rmtree(path)
        if (epoch + 1) % 2 == 0:
            os.makedirs(path)
            self.save_boxes = True
        else:
            self.save_boxes = False

    def training_mode(self):
        self.training = True

    def pre_item(self, img, gt_bboxes, gt_bboxes_q, filename):
        return {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_bboxes_q': gt_bboxes_q,
            'img_fields': ['img'],
            'bbox_fields': ['gt_bboxes', 'gt_bboxes_q'],
            'filename': filename,
            'ori_filename': filename,
            'ori_shape': img.shape
        }

    def __len__(self):
        return len(self.annotations_list)


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
