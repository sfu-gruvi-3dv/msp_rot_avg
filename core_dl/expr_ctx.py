import os
from functools import wraps


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ExprCtx(metaclass=Singleton):

    tmp_dir = None

    ckpt_dir = None

    log_dir = None

    gpu_workspace_dev = None

    def set_tmp_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.tmp_dir = dir_path

    def set_ckpt_dir(self, ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.ckpt_dir = ckpt_dir

    def set_log_dir(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

    def set_workspace_gpu_dev(self, dev):
        self.gpu_workspace_dev = dev
