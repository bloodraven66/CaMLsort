from TVB.models import *
from TVB.data_utils import *
import numpy as np
import os
import copy
from TVB.train import *
from TVB.tvb_utils import *
from huggingface_hub import hf_hub_download
from TVB.logger import logger

SUPPORTED_MODELS = [
                    '1CNN',
                    '3CNN',
                    '3DNN',
                    'CNNLSTM',
                    'Seq_CNNLSTM',
                    'Seq_CNNLSTM_1sec'
                    ]

class TVB_handler():
    def __init__(self,
                pretrained_model_name=None,
                config='TVB/configs/default.yaml'
                ):

        if not os.path.exists(config):
            hf_hub_download(
                            repo_id=self.repo_id,
                            filename='default.yaml',
                            cache_dir=self.model_cache_dir,
                            force_filename='default.yaml'
                            )
            config = os.path.join(self.model_cache_dir, 'default.yaml')
            logger.info(f'Using config in {config}')
            
        args = read_yaml(config)
        if pretrained_model_name is not None: args.pretrained_model_name = pretrained_model_name
        self.__dict__.update(args)
        if not args.use_logger:
            logger.propagate = False
        if args.pretrained_model_name not in SUPPORTED_MODELS:
            raise Exception(f'{args.pretrained_model_name} should be from {SUPPORTED_MODELS}')
        logger.info(f"Using {args.pretrained_model_name} model")
        self.model = self.get_pretrained_model(self.pretrained_model_name)
        self.args = args
        self.args.config_name = config
        if args.load_weights:
            self.download_and_load_weights()


    def check_device_status(self):
        if self.device != 'cpu':
            assert torch.cuda.is_available(), 'GPU not accessible, specify device as CPU'
        logger.info(f'Running on device- {self.device}')

    def get_pretrained_model(self, modelname):
        models = {
                'CNNLSTM':CNNLSTM,
                '1CNN':CNN1 ,
                '3CNN':CNN3,
                '3DNN':DNN3,
                'Seq_CNNLSTM':Seq_CNNLSTM,
                'Seq_CNNLSTM_1sec': Seq_CNNLSTM_1sec,
                }
        params = models[modelname].info()
        kernel_stride_info = models[modelname].kernel_stride_info()
        if kernel_stride_info["kernel"] != self.window_size:
            logger.info(f'Changing window size according to model training configuration to {kernel_stride_info["kernel"]}')
            self.window_size = kernel_stride_info["kernel"]
        if kernel_stride_info["stride"] != self.stride:
            logger.info(f'Changing shift size according to model training configuration to {kernel_stride_info["stride"]}')
            self.stride = kernel_stride_info["stride"]
        if self.pretrained_model_name in ['Seq_CNNLSTM', 'Seq_CNNLSTM_1sec']:
            self.batch_size = 1
            logger.info(f'Batch size > 1 for {self.pretrained_model_name} not supported yet, changing it to 1.')
        return models[modelname](**params)

    def download_and_load_weights(self):
        model_names = {
                    'CNNLSTM':'1cnn_fold2.pth',
                    '1CNN':'final_actual_1cnn_fold2.pth' ,
                    '3CNN':'final_actual_3cnn_fold2.pth',
                    '3DNN':'final_actual_3dnn_fold2.pth',
                    'Seq_CNNLSTM_1sec':'seq_cnn_lstm_1sec_fold2_300.pth',
                    'Seq_CNNLSTM':'seq_cnn_lstm_fold2_300.pth',
                    }
        if os.path.exists(os.path.join(self.model_cache_dir, model_names[self.pretrained_model_name])):
            logger.info(f'pretrained model weights for {self.pretrained_model_name} found at {self.model_cache_dir}')
        else:
            logger.info(f'pretrained model weights for {self.pretrained_model_name} not found at {self.model_cache_dir}, Downloading..')
            hf_hub_download(
                            repo_id=self.repo_id,
                            filename=model_names[self.pretrained_model_name],
                            cache_dir=self.model_cache_dir,
                            force_filename=model_names[self.pretrained_model_name]
                            )
            logger.info(f'Weights stored at {self.model_cache_dir}')
        self.model.load_state_dict(torch.load(os.path.join(self.model_cache_dir, model_names[self.pretrained_model_name]), map_location=self.device))
        logger.info(f'Loaded weights into model')

    def model_docs(self, 
                display_model=False,
                display_model_info=False,
                ):
        if display_model:
            logger.info(self.model)
        if display_model_info:
            model_info = self.model.info()
            logger.info(f'model parameters:{model_info}')
    
    def data_handler(self, 
                    data, 
                    label, 
                    filename, 
                    exp_name, 
                    use_sample_data,
                    sampling_rate
                    ):
        if data is None and use_sample_data is False:
            raise Exception("Provide data or enable 'use_sample_data'")

        if use_sample_data:
            data = download_sample_dataset(self.sample_data_path, self.repo_id)
        
        data = read_data(data, label, filename, exp_name)
        data = check_sampling_rate(data, sampling_rate)
        data = make_numpyloader(data, self.window_size, self.stride, self.normalisation)
        return data, copy.deepcopy(data)

    def train_data_handler(self, 
                            train_data, 
                            train_label, 
                            validation_data,
                            validation_label,
                            sampling_rate
                    ):

        assert sampling_rate == 30, f're sampling not supported'
        trainloader = make_dataloaders(train_data, train_label, self.window_size, self.stride, self.normalisation, batch_size=self.batch_size, shuffle=self.shuffle)
        valloader = make_dataloaders(validation_data, validation_label, self.window_size, self.stride, self.normalisation, batch_size=self.batch_size, shuffle=self.shuffle)
        logger.info('Created dataloaders')
        return trainloader, valloader



    def predict(self,
                data=None,
                label=None,
                filename=None,
                exp_name="exp",
                use_sample_data=False,
                custom_metric=None,
                display_model=False,
                display_model_info=False,
                sampling_rate=30,
                ):
        self.model_docs(display_model, display_model_info)
        data, raw_data = self.data_handler(data, label, filename, exp_name, use_sample_data, sampling_rate)
         
        results = pred_from_dict(data, self.model, self.device, self.window_size, self.tensordataset, self.batch_size,
                        self.num_workers, self.shuffle, self.progressbar)
        
        return results
    
    def train(self,
            train_data=None,
            train_label=None,
            validation_data=None,
            validation_label=None,
            filename=None,
            exp_name="exp",
            sampling_rate=30,
            ):
        
        loaders = self.train_data_handler(train_data, train_label, validation_data, validation_label, sampling_rate)
        if 'random_segment_training' in self.args and self.pretrained_model_name == 'Seq_CNNLSTM_1sec':
            logger.info('Creating random length continuous windows for training')
        train_model(self.model, loaders, device=self.device, training_args=self.args)

    # def prep_data(self, da)
# data = np.random.randn(9, 9000)
# import scipy.io
# data = scipy.io.loadmat('/home/sathvik/Downloads/5.mat')['imaging'].squeeze()[:, 1]
# data = np.array([data.tolist() , data.tolist()])
# print(data.shape)
# # exit()
# tvb_handler = TVB_handler('Seq_CNNLSTM_1sec', config='TVB/configs/default.yaml')
# tvb_handler.train(data, np.array([1, 0]), data, np.array([1, 0]))

# tvb_handler.predict(use_sample_data=True
# res = tvb_handler.predict(data)
# res = res['exp']['0']
# print(res)
# import matplotlib.pyplot as plt
# plt.plot(res['posterior0'])
# plt.plot(res['posterior1'])
# plt.savefig('p0.png')