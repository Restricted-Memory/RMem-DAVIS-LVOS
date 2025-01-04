from .default_deaot import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'R50_DeAOTL'

        self.MODEL_ENCODER = 'resnet50'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'  # https://download.pytorch.org/models/resnet50-0676ba61.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x

        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5
        self.TEST_MAX_LONG_TERM_MEM_GAP = 30
        self.FORMER_MEM_LEN = 1
        self.LATTER_MEM_LEN = 7
        self.RESTRIC_MEMORY = True
        self.RESTRIC_MEMORY_USING_ATTEN_WEIGHT = True
        # temporal positional embedding
        self.USE_TEMPORAL_POS_EMB = True
