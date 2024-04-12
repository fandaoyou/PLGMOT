import argparse
import os
import shutil

from detector.base_detector import BaseDetector
from models import *
from modify_mmdet_utils import *
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track demo')
    parser.add_argument("--exp_id", type=str, default='ours+br+loss_ablation1_0.4')
    parser.add_argument('--config', help='train config file path', default='configs/ours.py')
    parser.add_argument('--checkpoint', default='work_dirs/ours+br+loss_ablation1_0.4/latest.pth')
    parser.add_argument('--data_root', default='./data/GMOT40')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    model = model.to(device)

    data_root = args.data_root

    seqs = [each.split('.')[0] for each in os.listdir(os.path.join(data_root, 'template_box_by_seq_for_global_track'))]
    seqs.sort()

    tracker = BaseDetector(model, device)



    for i in range(len(seqs)):
        seq = seqs[i]
        # seq = 'bird-0'
        img_files = os.listdir(os.path.join(data_root, f'GenericMOT_JPEG_Sequence/{seq}/img1'))
        img_files.sort()
        img_files = [os.path.join(data_root, f'GenericMOT_JPEG_Sequence/{seq}/img1', each) for each in img_files]
        _, _, x, y, w, h, _ = np.loadtxt(os.path.join(data_root, f'template_box_by_seq_for_global_track/{seq}.txt'),
                                         delimiter=',')
        init_bbox = np.array([[x, y, w + x, h + y]], dtype=np.float32)

        output_seq_dir = os.path.join('./work_dirs', args.exp_id, 'det', seq, 'det')
        if os.path.exists(output_seq_dir):
            shutil.rmtree(output_seq_dir)
        os.makedirs(output_seq_dir)
        det_file = os.path.join(output_seq_dir, 'det.txt')

        print(f"{i}/{len(seqs)} Processing {seq}...")
        with open(det_file, 'a') as f:
            FPS, all_boxes = tracker.forward_test(img_files, init_bbox, visualize=False, thr=0.1)
            print('FPS:', FPS)
            for index, boxes in enumerate(all_boxes):
                for box in boxes:
                    x1, y1, x2, y2, conf = box
                    line = f"{index},-1,{x1},{y1},{x2},{y2},{conf},-1,-1,-1\n"
                    f.write(line)