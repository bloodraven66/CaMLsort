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
                normalisation='minmax',
                tensordataset=True,
                batch_size=4,
                collate_fn=None,
                num_workers=None,
                shuffle=False,
                device='cuda',
                progressbar=True,
                ):
        raiseException(mode, '==', 'predict')
        if pretrained_model_name not in SUPPORTED_MODELS:
            raise Exception(f'{pretrained_model_name} should be from {SUPPORTED_MODELS}')

        self.model = self.get_pretrained_model(pretrained_model_name)
        self.window_size = window_size
        self.stride = stride
        self.normalisation = normalisation
        self.tensordataset = tensordataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = 2
        self.shuffle = shuffle
        self.device = device
        self.progressbar = progressbar

    def get_pretrained_model(self, modelname):
        models = {'CNNLSTM':CNNLSTM(),
                '1CNN':CNN1() ,
                '3CNN':CNN3(),
                '3DNN':DNN3(),
                'Seq-CNNLSTM':Seq_CNNLSTM()}
        return models[modelname]

    def predict(self, data, custom_metric=None, display_model=False, display_model_info=False):
        if display_model:
            print(self.model)
        if display_model_info:
            a = self.model.info()
            print(a)
        data = read_data(data)
        data = make_numpyloader(data, self.window_size, self.stride, self.normalisation)
        results = pred_from_numpy(data, self.model, self.device, self.window_size, self.tensordataset, self.batch_size,
                        self.collate_fn, self.num_workers, self.shuffle, self.progressbar)
        return results

a = TVB_handler('CNNLSTM')
data = np.random.uniform(size=(5, 900))
# data = list(data)
out = a.predict(data, display_model_info=True, display_model=True)
print(out)
