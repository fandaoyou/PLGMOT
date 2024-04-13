import numpy as np
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, TwoStageDetector
from mmdet.core import bbox2roi, bbox2result

from .modules.modulators import RPN_Modulator, RCNN_Modulator


@DETECTORS.register_module()
class BaseModel(TwoStageDetector):
    def __init__(self, backbone, rpn_head, roi_head, train_cfg,
                 test_cfg, neck=None, pretrained=None, init_cfg=None):
        super(BaseModel, self).__init__(
            backbone=backbone, neck=neck, rpn_head=rpn_head, roi_head=roi_head,
            train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg
        )
        self.rpn_modulator = RPN_Modulator()
        self.rcnn_modulator = RCNN_Modulator()

    def forward_train(self, img_z, img_x, img_meta_z, img_meta_x,
                      sample_bboxes_z, sample_bboxes_x, gt_bboxes_x,
                      gt_labels, gt_bboxes_confs, gt_bboxes_ignore=None, **kwargs):
        losses = {}
        total = 0.
        batch_size = len(img_meta_x)

        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)

        for i in range(batch_size):
            losses_i = {}

            z_i = [u[i:i + 1] for u in z]
            x_i = [u[i:i + 1] for u in x]
            gt_bboxes_i = gt_bboxes_x[i:i + 1]
            sample_bboxes_z_i = sample_bboxes_z[i]
            sample_bboxes_x_i = sample_bboxes_x[i]
            gt_labels_i = gt_labels[i:i + 1]
            img_meta_xi = img_meta_x[i:i + 1]
            gt_bboxes_confs_i = gt_bboxes_confs[i]

            sample_bboxes_z_i = sample_bboxes_z_i[:1]
            rois_z = bbox2roi([sample_bboxes_z_i])
            bbox_feats_z = self.roi_head.bbox_roi_extractor(z_i[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)
            bbox_feats_z = bbox_feats_z.mean(dim=0, keepdim=True)

            # rpn forward
            rpn_feats = self.rpn_modulator(x_i, bbox_feats_z)
            rpn_outs = self.rpn_head(rpn_feats)
            rpn_loss_inputs = rpn_outs + (gt_bboxes_i, img_meta_xi)
            rpn_losses_i = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas=img_meta_xi, cfg=proposal_cfg)
            losses_i.update(rpn_losses_i)

            # roi forward
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None]
            assign_result = self.roi_head.bbox_assigner.assign(
                proposal_list[0], gt_bboxes_i[0], gt_bboxes_ignore[0], gt_labels_i[0]
            )
            sampling_result = self.roi_head.bbox_sampler.sample(
                assign_result, proposal_list[0], gt_bboxes_i[0], gt_labels_i[0],
                feats=[lvl_feat[0][None] for lvl_feat in rpn_feats]
            )
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            sampling_results = [sampling_result]

            rois_x = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats_x = self.roi_head.bbox_roi_extractor(
                x_i[:self.roi_head.bbox_roi_extractor.num_inputs], rois_x)

            bbox_feats = self.rcnn_modulator(bbox_feats_z, bbox_feats_x)

            cls_score, bbox_pred = self.roi_head.bbox_head(bbox_feats)

            labels, label_weights, bbox_targets, bbox_weights = self.roi_head.bbox_head.get_targets(
                sampling_results,
                gt_bboxes_i,
                gt_labels_i,
                self.roi_head.train_cfg
            )

            bbox_weights[:len(pos_assigned_gt_inds)] = gt_bboxes_confs_i[pos_assigned_gt_inds]

            loss_bbox = self.roi_head.bbox_head.loss(cls_score, bbox_pred, rois_x, labels, label_weights, bbox_targets, bbox_weights)

            losses_i.update(loss_bbox)

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
            total += 1.

        for k, v in losses.items():
            if isinstance(v, (tuple, list)):
                for u in range(len(v)):
                    losses[k][u] /= total
            else:
                losses[k] /= total
        return losses

    def forward_test(self, img_z, img_x, img_meta_z, img_meta_x, sample_bboxes_z, **kwargs):

        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        results = []
        batch_size = len(img_meta_x)
        for i in range(batch_size):
            # RPN forward
            z_i = [u[i:i + 1] for u in z]
            x_i = [u[i:i + 1] for u in x]
            sample_bboxes_z_i = sample_bboxes_z[i]
            img_meta_xi = img_meta_x[i:i + 1]

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

    def simple_test_bboxes(self, bbox_feats_z, x, img_meta_xi, proposals, rcnn_test_cfg, rescale=False, **kwargs):

        rois_x = bbox2roi(proposals)
        bbox_feats_x = self.roi_head.bbox_roi_extractor(x[:self.roi_head.bbox_roi_extractor.num_inputs], rois_x)

        roi_feats = self.rcnn_modulator(bbox_feats_z, bbox_feats_x)

        cls_score, bbox_pred = self.roi_head.bbox_head(roi_feats)

        img_shape = img_meta_xi[0]['img_shape']
        scale_factor = img_meta_xi[0]['scale_factor']

        det_bboxes, det_labels = self.roi_head.bbox_head.get_bboxes(
            rois_x, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg
        )

        return det_bboxes, det_labels

    def _process_query(self, img_z, sample_bboxes_z_i):
        z = self.extract_feat(img_z)
        rois_z = bbox2roi(sample_bboxes_z_i)
        bbox_feats_z = self.roi_head.bbox_roi_extractor(z[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)
        bbox_feats_z = bbox_feats_z.mean(dim=0, keepdim=True)
        self._query = bbox_feats_z

    def _process_gallary(self, img_x, img_meta_x, **kwargs):
        x = self.extract_feat(img_x)

        # RPN forward
        rpn_feats = self.rpn_modulator(x, self._query)
        proposal_list = self.rpn_head.simple_test_rpn(rpn_feats, img_meta_x)

        # RCNN forward
        det_bboxes, det_labels = self.simple_test_bboxes(
            self._query, x, img_meta_x, proposal_list, self.test_cfg.rcnn, **kwargs
        )

        bbox_results = bbox2result(det_bboxes, det_labels, self.roi_head.bbox_head.num_classes)
        out = bbox_results[0]

        return out

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=1)
        return outputs

    @auto_fp16(apply_to=('img_x', 'img_z'))
    def forward(self, img_z, img_x, img_meta_z, img_meta_x, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(img_z, img_x, img_meta_z, img_meta_x, **kwargs)
