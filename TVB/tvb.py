from models import *
import numpy as np
from tvb_utils import *
SUPPORTED_MODELS = ['1CNN', '3CNN', '3DNN', 'CNNLSTM', 'Seq-CNNLSTM']

class TVB_handler():
    def __init__(self,
                pretrained_model_name,
                custom_model=False,
                mode='predict',
                get_f1=True,
                get_acc=True,
                window_size=300,
                stride=100,
                normalisation='minmax'
                ):
        if mode != 'predict':
            raise NotImplementedError
        if pretrained_model_name not in SUPPORTED_MODELS:
            raise Exception(f'{pretrained_model_name} should be from {SUPPORTED_MODELS}')

        self.model = self.get_pretrained_model(pretrained_model_name)
        self.window_size = window_size
        self.stride = stride
        self.normalisation = normalisation

    def get_pretrained_model(self, modelname):
        if modelname == 'CNNLSTM':
            return CNNLSTM()
        return None, None

    def predict(self, data, custom_metric=None, display_model=False, display_model_info=False):
        if display_model:
            print(self.model)
        if display_model_info:
            a = self.model.info()
            print(a)
        data = read_data(data)
        dataloader = make_dataloaders(data, self.window_size, self.stride, self.normalisation)



a = TVB_handler('CNNLSTM')
data = np.random.uniform(size=(6, 800))
# data = list(data)
a.predict(data, display_model_info=True, display_model=True)
