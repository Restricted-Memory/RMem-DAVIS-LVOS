import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PRE_LVOS'

        self.init_dir()

        self.DATASETS = ['lvos']

        pretrain_stage = 'PRE_YTB_DAV'
        pretrain_ckpt = 'save_step_100000.pth'
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'result',
                                           self.EXP_NAME, pretrain_stage,
                                           'ema_ckpt', pretrain_ckpt)

        self.TRAIN_SAVE_STEP = 1000
        self.TRAIN_TOTAL_STEPS = 1000
        self.DATA_SEQ_LEN = 15
        self.TRAIN_LONG_TERM_MEM_GAP = 4
