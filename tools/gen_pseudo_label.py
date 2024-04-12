import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from tqdm import tqdm


def generate_bbox_v1(gt_bbox, points, H, W, id):
    gt_bbox = np.array(gt_bbox)
    w = np.mean(gt_bbox[:, 2] - gt_bbox[:, 0])
    h = np.mean(gt_bbox[:, 3] - gt_bbox[:, 1])
    max_w = np.max(gt_bbox[:, 2] - gt_bbox[:, 0])
    max_h = np.max(gt_bbox[:, 3] - gt_bbox[:, 1])

    scale_limit = 1 * max_h if max_w > max_h else 1 * max_w
    box_scale = w / h

    scales = []
    annotation = {}
    points = np.array([[int(a), int(b)] for a, b in points])
    tree = scipy.spatial.KDTree(points.copy(), leafsize=1024)
    k = min(len(points), 3)

    assert k > 2

    distances, distances_idx = tree.query(points, k=len(points))

    distances_mean = np.mean(distances[:, 1:k], axis=1)

    scale = []
    for s_p in distances_mean:
        s_p = np.clip(s_p, 1, scale_limit)
        scale.append(s_p)
    scale = np.array(scale)

    scales.extend(list(scale))

    boxes_with_scale = np.zeros((len(points), 4), dtype=np.float32)
    scale_w = scale / 2.
    scale_h = scale / 2.
    if box_scale > 1:
        scale_w = scale_w * box_scale
    else:
        scale_h = scale_h / box_scale
    scale_w = np.clip(scale_w, 0, max_w/2)
    scale_h = np.clip(scale_h, 0, max_h/2)
    boxes_with_scale[:, 0], boxes_with_scale[:, 2] = points[:, 0] - scale_w, points[:, 0] + scale_w  # x1, x2
    boxes_with_scale[:, 1], boxes_with_scale[:, 3] = points[:, 1] - scale_h, points[:, 1] + scale_h  # y1, y2

    boxes_with_scale[:, 0:4:2] = np.clip(boxes_with_scale[:, 0:4:2], 1, W - 1)
    boxes_with_scale[:, 1:4:2] = np.clip(boxes_with_scale[:, 1:4:2], 1, H - 1)

    match_index = box_filter(gt_bbox, boxes_with_scale)
    all_index = list(range(len(boxes_with_scale)))

    for each in match_index:
        if each in all_index:
            all_index.remove(each)
        else:
            print('warning', match_index, id)

    boxes_with_scale = boxes_with_scale[all_index]

    annotation['pseudo_bbox'] = boxes_with_scale.tolist()
    annotation['confs'] = 0.6 * np.ones((boxes_with_scale.shape[0]))
    annotation['confs'] = annotation['confs'].tolist()
    return annotation


def box_filter(gt_boxes, pseudo_boxes):
    gt_boxes = np.array(gt_boxes)
    pseudo_boxes = np.array(pseudo_boxes)

    gt_center = gt_boxes[:, :2] + gt_boxes[:, 2:]
    pseudo_center = pseudo_boxes[:, :2] + pseudo_boxes[:, 2:]

    N = pseudo_center.shape[0]
    K = gt_center.shape[0]

    points = pseudo_center.reshape(N, 1, 2).repeat(K, 1)
    query_points = gt_center.reshape(1, K, 2).repeat(N, 0)

    distance = np.sqrt((points[:, :, 0] - query_points[:, :, 0]) ** 2 + (points[:, :, 1] - query_points[:, :, 1]) ** 2)
    match = np.argmin(distance, axis=0)
    return match


def main():
    root = './data/FSC147/FSC147_384_V2'
    image_root = os.path.join(root, 'images_384_VarV2')
    anno_file = os.path.join(root, 'annotation_FSC147_384.json')
    class_file = os.path.join(root, 'ImageClasses_FSC147.txt')
    split_file = os.path.join(root, 'Train_Test_Val_FSC_147.json')
    out_dir = os.path.join(root, 'pseudo_init')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(split_file) as f:
        split_data = json.load(f)

    cls_dict = {}
    with open(class_file) as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            image_name, cls = data[0], data[1:]
            cls_name = '_'.join(cls)
            cls_dict[image_name] = cls_name

    image_list = os.listdir(image_root)
    # np.random.shuffle(image_list)
    for image_name in tqdm(image_list):

        annotation = {}
        image_path = os.path.join(image_root, image_name)
        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anno_data = annotations[image_name]
        # H, W = anno_data['H'], anno_data['W']
        box_examples_coordinates = anno_data['box_examples_coordinates']
        points = anno_data['points']

        annotation["image_name"] = image_name
        annotation["gt_bbox"] = [[c[0][0], c[0][1], c[2][0], c[2][1]] for c in box_examples_coordinates]
        annotation["points"] = points
        annotation["cls"] = cls_dict[image_name]
        annotation["height"] = H
        annotation["width"] = W
        H, W = image.shape[:2]

        pseudo_label = generate_bbox_v1(annotation["gt_bbox"], points, H, W, image_name)

        annotation.update(pseudo_label)

        with open(os.path.join(out_dir, image_name.replace('jpg', 'json')), 'w') as f:
            json.dump(annotation, f, indent=2)

        # for bbox_points in box_examples_coordinates:
        #     cv2.rectangle(image, bbox_points[0], bbox_points[2], (0, 255, 0), 2)
        # for point in annotation['points']:
        #     point = [int(c) for c in point]
        #     cv2.circle(image, point, 1, (0, 255, 0), 2)
        
        # for bbox in annotation['pseudo_bbox']:
        #     bbox = [int(c) for c in bbox]
        #     x1, y1, x2, y2 = bbox
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        
        # plt.imshow(image)
        # plt.title(image_name)
        # plt.show()
        # plt.close()


if __name__ == '__main__':
    main()
