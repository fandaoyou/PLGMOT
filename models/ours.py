import numpy as np
import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import bbox2roi, bbox2result
from torch import nn
import torch.nn.functional as F
from . import BaseModel
from .modules.transformer import FeatureReconstructTransformer
from .modules.update_net import UpdateNet


@DETECTORS.register_module()
class PLGMOT(BaseModel):
    def __init__(self, backbone, rpn_head, roi_head, train_cfg,
                 test_cfg, neck=None, pretrained=None, init_cfg=None, up_thr=0.6):
        super(PLGMOT, self).__init__(
            backbone=backbone, neck=neck, rpn_head=rpn_head, roi_head=roi_head,
            train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg
        )
        self.feat_encoder = FeatureReconstructTransformer(in_c=256, embed_dim=256, depth=4, num_heads=8)
        self.update_net = UpdateNet(in_c=256, out_c=256)
        self.up_thr = up_thr

    def forward_train(self, img_z, img_x_list, img_meta_z, img_meta_x_list,
                      sample_bboxes_z, sample_bboxes_x_list, gt_bboxes_x_list,
                      gt_labels, gt_bboxes_confs, gt_bboxes_ignore=None, **kwargs):
        losses = {}
        total = 0.
        seq_size, batch_size = len(img_meta_x_list), len(img_meta_x_list[0])

        assert batch_size == 1

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)

        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x_list[0])
        sample_bboxes_z = sample_bboxes_z[0][:1]
        rois_z = bbox2roi([sample_bboxes_z])
        bbox_feats_z = self.roi_head.bbox_roi_extractor(z[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)
        bbox_feats_z = bbox_feats_z.mean(dim=0, keepdim=True)

        init_feat = bbox_feats_z.clone().detach()
        cur_feat = bbox_feats_z.clone()

        for i in range(seq_size):
            losses_i = {}

            x_i = [u[i:i + 1] for u in x]

            gt_bboxes_i = gt_bboxes_x_list[i]
            img_meta_x_i = img_meta_x_list[i]
            gt_bboxes_confs_i = gt_bboxes_confs[0]

            # rpn forward
            rpn_feats = self.rpn_modulator(x_i, cur_feat)
            rpn_outs = self.rpn_head(rpn_feats)
            rpn_loss_inputs = rpn_outs + (gt_bboxes_i, img_meta_x_i)
            rpn_losses_i = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas=img_meta_x_i, cfg=proposal_cfg)
            losses_i.update(rpn_losses_i)

            # roi forward
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None]
            assign_result_i = self.roi_head.bbox_assigner.assign(
                proposal_list[0], gt_bboxes_i[0], gt_bboxes_ignore[0], gt_labels[0]
            )
            sampling_result = self.roi_head.bbox_sampler.sample(
                assign_result_i, proposal_list[0], gt_bboxes_i[0], gt_labels[0],
                feats=[lvl_feat[0][None] for lvl_feat in rpn_feats]
            )
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            sampling_results = [sampling_result]

            rois_x_i = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats_x_i = self.roi_head.bbox_roi_extractor(
                x_i[:self.roi_head.bbox_roi_extractor.num_inputs], rois_x_i)

            bbox_feats = self.rcnn_modulator(cur_feat, bbox_feats_x_i)

            cls_score, bbox_pred = self.roi_head.bbox_head(bbox_feats)

            labels, label_weights, bbox_targets, bbox_weights = self.roi_head.bbox_head.get_targets(
                sampling_results,
                gt_bboxes_i,
                gt_labels,
                self.roi_head.train_cfg
            )

            bbox_weights[:len(pos_assigned_gt_inds)] = gt_bboxes_confs_i[pos_assigned_gt_inds]

            loss_bbox = self.roi_head.bbox_head.loss(cls_score, bbox_pred, rois_x_i, labels, label_weights, bbox_targets, bbox_weights)

            if i != 0:
                losses_i.update(loss_bbox)
                total += 1.

            # feat reconstruct
            with torch.no_grad():
                props_bboxes = self.get_proposal_bboxes(img_meta_x_i[0], rois_x_i, cls_score, bbox_pred,
                                                        sample_bboxes_x_list[i][0])
                props_feats = self.get_proposal_feats(x_i, [props_bboxes]).unsqueeze(dim=0)  # [dets, 256, 7, 7]
            rec_feats = self.feat_encoder(props_feats)  # [1, 256, 7, 7]
            cur_feat = self.update_net(rec_feats, init_feat, cur_feat)

            # update losses
            for k, v in losses_i.items():
                if k in losses:
                    if isinstance(v, (tuple, list)):
                        for u in range(len(v)):
                            losses[k][u] += v[u]
                    else:
                        losses[k] += v
                else:
                    losses[k] = v

        for k, v in losses.items():
            if isinstance(v, (tuple, list)):
                for u in range(len(v)):
                    losses[k][u] /= total
            else:
                losses[k] /= total
        return losses

    def forward_test(self, img_z, img_x_list, img_meta_z, img_meta_x_list, sample_bboxes_z, **kwargs):

        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x_list)

        results = []
        batch_size = len(img_meta_x_list)
        for i in range(batch_size):
            # RPN forward
            z_i = [u[i:i + 1] for u in z]
            x_i = [u[i:i + 1] for u in x]
            sample_bboxes_z_i = sample_bboxes_z[i]
            img_meta_xi = img_meta_x_list[i:i + 1]

            rois_z = bbox2roi([sample_bboxes_z_i])
            bbox_feats_z = self.roi_head.bbox_roi_extractor(z_i[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)
            bbox_feats_z = bbox_feats_z.mean(dim=0, keepdim=True)

            # rpn forward
            rpn_feats = self.rpn_modulator(x_i, bbox_feats_z)
            proposal_list = self.rpn_head.simple_test_rpn(rpn_feats, img_meta_xi)

            # rcnn forward
            det_bboxes, det_labels = self.simple_test_bboxes(
                bbox_feats_z, x_i, img_meta_xi, proposal_list, self.test_cfg.rcnn, **kwargs
            )

            bbox_results = bbox2result(det_bboxes, det_labels, self.roi_head.bbox_head.num_classes)

            bboxes = bbox_results[0]
            results.append(bboxes)

        return results

    def get_proposal_feats(self, feats, gt_bboxes):
        rois = bbox2roi(gt_bboxes)
        proposal_feats = self.roi_head.bbox_roi_extractor(feats[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        return proposal_feats

    def get_proposal_bboxes(self, img_meta, rois, cls_score, bbox_pred, sample_bboxes):
        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        det_bboxes, det_labels = self.roi_head.bbox_head.get_bboxes(
            rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=self.test_cfg.rcnn
        )
        bbox_results = det_bboxes[det_labels == 0, :]
        pred_bboxes = bbox_results[bbox_results[:, -1] > self.up_thr, :-1][:16]
        props_bboxes = sample_bboxes
        if pred_bboxes.any():
            props_bboxes = torch.vstack([props_bboxes, pred_bboxes])
        return props_bboxes

    def _process_query(self, img_z, sample_bboxes_z_i):
        z = self.extract_feat(img_z)
        rois_z = bbox2roi(sample_bboxes_z_i)
        bbox_feats_z = self.roi_head.bbox_roi_extractor(z[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)
        bbox_feats_z = bbox_feats_z.mean(dim=0, keepdim=True)
        self.cur_feat = bbox_feats_z
        self.init_feat = bbox_feats_z

    def _process_gallary(self, img_x, img_meta_x, **kwargs):
        x = self.extract_feat(img_x)

        # RPN forward
        rpn_feats = self.rpn_modulator(x, self.cur_feat)
        proposal_list = self.rpn_head.simple_test_rpn(rpn_feats, img_meta_x)

        # RCNN forward
        det_bboxes, det_labels = self.simple_test_bboxes(
            self.cur_feat, x, img_meta_x, proposal_list, self.test_cfg.rcnn, **kwargs
        )

        # update query
        bbox_results = [det_bboxes[det_labels == i, :] for i in range(self.roi_head.bbox_head.num_classes)]

        props_bboxes = bbox_results[0]
        props_bboxes = props_bboxes[props_bboxes[:, -1] > self.up_thr][:16]
        if props_bboxes.any():
            props_feats = self.get_proposal_feats(x, [props_bboxes]).unsqueeze(dim=0)
            rec_feats = self.feat_encoder(props_feats)
            self.cur_feat = self.update_net(rec_feats, self.init_feat, self.cur_feat)

        out = bbox_results[0].detach().cpu().numpy()
        return out

    def forward(self, img_z, img_x_list, img_meta_z, img_meta_x_list, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img_z, img_x_list, img_meta_z, img_meta_x_list, **kwargs)
        else:
            return self.forward_test(img_z, img_x_list, img_meta_z, img_meta_x_list, **kwargs)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()

        # backbone
        z = self.extract_feat(img)
        x = self.extract_feat(img)
        box = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32).cuda()
        rois_z = bbox2roi([box])
        bbox_feats_z = self.roi_head.bbox_roi_extractor(z[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)
        rpn_feats = self.rpn_modulator(x, bbox_feats_z)

        # rpn
        rpn_feats = self.rpn_modulator(x, bbox_feats_z)
        rpn_outs = self.rpn_head(rpn_feats)

        proposals = torch.randn(1000, 4).to(img.device)

        # roi_head

        rois_x_i = bbox2roi([proposals])
        bbox_feats_x_i = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois_x_i)

        bbox_feats = self.rcnn_modulator(bbox_feats_z, bbox_feats_x_i)

        cls_score, bbox_pred = self.roi_head.bbox_head(bbox_feats)

        rec_feats = self.feat_encoder(bbox_feats_x_i[:16].unsqueeze(0))  # [1, 256, 7, 7]
        cur_feat = self.update_net(rec_feats, rec_feats.clone(), rec_feats.clone())
        return outs