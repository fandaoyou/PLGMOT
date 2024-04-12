import time
import warnings
from typing import List, Optional, Tuple

import torch
from mmcv.runner import RUNNERS, EpochBasedRunner, get_host_info
from torch.utils.data import DataLoader

import mmcv


@RUNNERS.register_module()
class my_Runner(EpochBasedRunner):

    @torch.no_grad()
    def update(self, data_loader):
        count = 0
        self.model.eval()
        dataset = data_loader.dataset
        dataset.updating_mode(self.epoch, self.work_dir)
        for i, data in enumerate(data_loader):
            bboxes = self.model(return_loss=False, rescale=True, **data)
            img_meta_x = data['img_meta_x'].data[0]

            for i, img_meta in enumerate(img_meta_x):
                image_name = img_meta['filename']
                count += dataset.revise_data(bboxes[i], image_name, self.work_dir, self.epoch)
        self.logger.info(f'update {count} bounding boxes in {self.epoch} Epoch')
        dataset.training_mode()

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)

        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an epoch')
                    if mode == 'val':
                        raise ValueError(
                            f'runner has no method named val to run an epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[0], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')