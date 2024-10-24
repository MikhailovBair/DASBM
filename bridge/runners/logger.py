from pytorch_lightning.loggers import CSVLogger as _CSVLogger, WandbLogger as _WandbLogger
import wandb
import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def log_metrics(self, metric_dict, step=None):
        pass

    def log_hyperparams(self, params):
        pass

    def log_image(self, key, images, **kwargs):
        pass


class CSVLogger(_CSVLogger):
    def log_image(self, key, images, **kwargs):
        pass


class WandbLogger(_WandbLogger):
    LOGGER_JOIN_CHAR = '/'

    def log_metrics(self, metrics, step=None, fb=None):
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        super().log_metrics(metrics, step=step)

    def log_image(self, key, images, **kwargs):
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        step = kwargs.pop("step", None)
        fb = kwargs.pop("fb", None)
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        if n == 1:
            metrics = {key: wandb.Image(images[0], **kwarg_list[0])}
        else:
            metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step=step, fb=fb)


class BaseWriter(object):
    def __init__(self, opt):
        self.rank = 0
    def add_scalar(self, step, key, val):
        pass  # do nothing
    def add_image(self, step, key, image):
        pass  # do nothing
    def close(self): pass


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter, self).__init__(opt)
        if self.rank == 0:
            print(f"opt = {opt}")
            run_dir = os.path.join(opt.path_to_save_info, opt.data.dataset, opt.exp_name)

            os.makedirs(run_dir, exist_ok=True)
            print(f"run dir for tensorboard = {run_dir}")
            self.writer = SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0:
            self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = torch.tensor(image).permute((2, 0, 1))
            print(f"logging image with size {image.shape} to tensorboard with key = {key}, max = {image.max()}")
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0:
            self.writer.close()
