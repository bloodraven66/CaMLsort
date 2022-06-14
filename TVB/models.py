import torch.nn as nn
import torch

class CNNLSTM(nn.Module):
    def __init__(self,
                num_feats,
                kernel_size,
                stride_size ,
                hidden_size,
                num_layers
                ):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(1,
                            num_feats,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(batch_first=True,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            input_size=num_feats
                            )
        self.feat = num_feats
        self.linear = nn.Linear(hidden_size*60,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.conv(c_in)
        c_out = self.relu(c_out)
        c_out = self.lstm(c_out.permute(0, 2, 1))[0]
        c_out = c_out.reshape(c_out.shape[0], -1)
        c_out = self.linear(c_out)
        return c_out

    @staticmethod
    def info():
        return {
            "num_feats" : 12,
            "hidden_size" : 32,
            "kernel_size" : 5,
            "stride_size" : 5,
            "num_layers" : 1,
        }

    @staticmethod
    def kernel_stride_info():
        return {
            "kernel" : 300,
            "stride" : 30,
        }

    def results(self, out):
        return extract_from_presoftmax_logits(out)

class Seq_CNNLSTM(nn.Module):
    def __init__(self,
                kernel_size,
                stride_size,
                num_feats,
                num_layers,
                hidden_size,
                last_conv_kernel_size,
                last_conv_filter_size,
                ):
        super(Seq_CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(1,
                            num_feats,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=num_feats,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.conv3 = nn.Conv1d(hidden_size, 1, kernel_size=10,stride=10)

    def forward(self, x):
        x = self.relu(self.conv(x.permute(0, 2, 1)))
        x = x.permute(1, 0, 2)
        x = torch.flatten(x, start_dim=1, end_dim=-1).unsqueeze(0).permute(0, 2, 1)
        x = self.relu(self.lstm(x)[0]).permute(0, 2, 1)
        x = self.conv3(x)
        return x

    @staticmethod
    def info():
        return {
            "kernel_size" : 30,
            "stride_size" : 30,
            "num_feats" : 4,
            "num_layers" : 1,
            "hidden_size" : 24,
            "last_conv_kernel_size": 10,
            "last_conv_filter_size" : 10,
        }

    @staticmethod
    def kernel_stride_info():
        return {
            "kernel" : 300,
            "stride" : 300,
        }

    def results(self, out):
        return extract_from_presigmoid_logits(out)


class Seq_CNNLSTM_1sec(nn.Module):
    def __init__(self,
                kernel_size,
                stride_size,
                num_feats,
                num_layers,
                hidden_size,
                ):
        super(Seq_CNNLSTM_1sec, self).__init__()
        self.conv = nn.Conv1d(1,
                            num_feats,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=num_feats,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=True
                            )
        self.conv3 = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        x = self.relu(self.conv(x.permute(0, 2, 1)))
        x = x.permute(1, 0, 2)
        x = torch.flatten(x, start_dim=1, end_dim=- 1).unsqueeze(0).permute(0, 2, 1)
        x = self.lstm(x)[0]
        x = self.conv3(x)
        return x

    @staticmethod
    def info():
        return {
            "kernel_size" : 30,
            "stride_size" : 30,
            "num_feats" : 4,
            "num_layers" : 1,
            "hidden_size" : 24,
        }

    @staticmethod
    def kernel_stride_info():
        return {
            "kernel" : 300,
            "stride" : 300,
        }

    def results(self, out):
        return extract_from_presigmoid_logits(out)

class CNN3(nn.Module):
    def __init__(self,
                num_feats,
                kernel_size,
                stride_size,
                pool_size,
                reduce_dim
                ):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv1d(1,
                            num_feats//4,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.conv2 = nn.Conv1d(num_feats//4,
                            num_feats//2,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.conv3 = nn.Conv1d(num_feats//2,
                            num_feats,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        self.feat = num_feats
        self.z = reduce_dim
        self.linear = nn.Linear(num_feats*reduce_dim,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        out = self.conv1(c_in)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        c_out = self.pool(out)
        c_out = c_out.reshape(c_out.shape[0], self.feat*self.z)
        c_out = self.linear(c_out)
        return c_out

    def results(self, out):
        return extract_from_presoftmax_logits(out)

    @staticmethod
    def info():
        return {
            "kernel_size" : 20,
            "stride_size" : 2,
            "num_feats" : 12,
            "pool_size" : 3,
            "reduce_dim" : 7,
        }

    @staticmethod
    def kernel_stride_info():
        return {
            "kernel" : 300,
            "stride" : 30,
        }

class CNN1(nn.Module):
    def __init__(self,
                num_feats,
                kernel_size,
                stride_size,
                pool_size,
                reduce_dim
                ):
        super(CNN1, self).__init__()
        self.conv = nn.Conv1d(1,
                            num_feats,
                            kernel_size=kernel_size,
                            stride=stride_size
                            )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        self.feat = num_feats
        self.z = reduce_dim
        self.linear = nn.Linear(num_feats*reduce_dim,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        out = self.conv(c_in)
        out = self.relu(out)
        c_out = self.pool(out)
        c_out = c_out.reshape(c_out.shape[0], self.feat*self.z)
        c_out = self.linear(c_out)
        return c_out


    @staticmethod
    def info():
        return {
            "kernel_size" : 20,
            "stride_size" : 1,
            "num_feats" : 12,
            "pool_size" : 3,
            "reduce_dim" : 93,
        }
    def results(self, out):
        return extract_from_presoftmax_logits(out)

    @staticmethod
    def kernel_stride_info():
        return {
            "kernel" : 300,
            "stride" : 30,
        }


class DNN3(nn.Module):
    def __init__(self,
                num_feats,
                pool_size,
                reduce_dim
                ):
        super(DNN3, self).__init__()
        self.linear1 = nn.Linear(1,num_feats//2)
        self.pool = nn.MaxPool1d(pool_size)
        self.linear2 = nn.Linear(num_feats//2, num_feats)
        self.feat = num_feats
        self.relu = nn.ReLU()
        self.z = reduce_dim
        self.linear4 = nn.Linear(num_feats*reduce_dim,2)

    def forward(self, x):

        out = self.linear1(x)
        out = self.pool(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.pool(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        out = self.relu(out)
        out = out.reshape(out.shape[0], self.feat*self.z)
        out = self.linear4(out)
        return out

    @staticmethod
    def info():
        return {
            "num_feats" : 64,
            "pool_size" : 3,
            "reduce_dim" : 33,
        }

    def results(self, out):
        return extract_from_presoftmax_logits(out)

    @staticmethod
    def kernel_stride_info():
        return {
            "kernel" : 300,
            "stride" : 30,
        }

def extract_from_presoftmax_logits(out):
    softmax_output = nn.Softmax(dim=1)(out).detach().cpu()
    class_ = torch.argmax(softmax_output, dim=1).numpy()
    return {'predicted_class':class_,
            'posterior0':softmax_output[:, 0].tolist(),
            'posterior1':softmax_output[:, 1].tolist()}

def extract_from_presigmoid_logits(out):
    class_ = (torch.sigmoid(out)>0.5).float().tolist()
    post0 = (1-torch.sigmoid(out)).detach().cpu().numpy()
    post1 = (torch.sigmoid(out)).detach().cpu().numpy()
    return {'predicted_class':class_,
            'posterior0':post0.tolist(),
            'posterior1':post1.tolist()}
