import torch.nn as nn
import torch

class CNNLSTM(nn.Module):
    def __init__(self, num_feat=12, kernel_size=5, stride=5,):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(1, num_feat, kernel_size=kernel_size,stride=stride)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(batch_first=True, num_layers=1, hidden_size=32, input_size=12)
        self.feat = num_feat
        self.linear = nn.Linear(32*60,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.conv(c_in)
        c_out = self.relu(c_out)
        c_out = self.lstm(c_out.permute(0, 2, 1))[0]
        c_out = c_out.reshape(c_out.shape[0], -1)
        c_out = self.linear(c_out)
        return c_out

    def info(self):
        return {
            "num_conv_layers" : 1,
            "in_features" : 1,
            "out_features" : 12,
            "kernel_size" : 5,
            "stride_size" : 5,
            "num_lstm_layers" : 1,
            "lstm_hidden_size" : 32
        }

    def results(self, out):
        softmax_output = nn.Softmax(dim=1)(out).detach().cpu()
        class_ = torch.argmax(softmax_output, dim=1).numpy()
        return {'class':class_,
                'post0':softmax_output[:, 0].tolist(),
                'post1':softmax_output[:, 1].tolist()}

class Seq_CNNLSTM(nn.Module):
    def __init__(self, num_feat=12, kernel_size=5, stride=5,):
        super(Seq_CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(1, num_feat, kernel_size=kernel_size,stride=stride)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(batch_first=True, num_layers=1, hidden_size=32, input_size=12)
        self.feat = num_feat
        self.linear = nn.Linear(32*60,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.conv(c_in)
        c_out = self.relu(c_out)
        c_out = self.lstm(c_out.permute(0, 2, 1))[0]
        c_out = c_out.reshape(c_out.shape[0], -1)
        c_out = self.linear(c_out)
        return c_out

    def info(self):
        return {
            "num_conv_layers" : 1,
            "in_features" : 1,
            "out_features" : 12,
            "kernel_size" : 5,
            "stride_size" : 5,
            "num_lstm_layers" : 1,
            "lstm_hidden_size" : 32
        }

    def results(self, out):
        softmax_output = nn.Softmax(dim=1)(out).detach().cpu()
        class_ = torch.argmax(softmax_output, dim=1).numpy()
        return {'class':class_,
                'post0':softmax_output[:, 0].tolist(),
                'post1':softmax_output[:, 1].tolist()}
class CNN1(nn.Module):
    def __init__(self, num_feat=12, kernel_size=5, stride=5,):
        super(CNN1, self).__init__()
        self.conv = nn.Conv1d(1, num_feat, kernel_size=kernel_size,stride=stride)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(batch_first=True, num_layers=1, hidden_size=32, input_size=12)
        self.feat = num_feat
        self.linear = nn.Linear(32*60,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.conv(c_in)
        c_out = self.relu(c_out)
        c_out = self.lstm(c_out.permute(0, 2, 1))[0]
        c_out = c_out.reshape(c_out.shape[0], -1)
        c_out = self.linear(c_out)
        return c_out

    def info(self):
        return {
            "num_conv_layers" : 1,
            "in_features" : 1,
            "out_features" : 12,
            "kernel_size" : 5,
            "stride_size" : 5,
            "num_lstm_layers" : 1,
            "lstm_hidden_size" : 32
        }

    def results(self, out):
        softmax_output = nn.Softmax(dim=1)(out).detach().cpu()
        class_ = torch.argmax(softmax_output, dim=1).numpy()
        return {'class':class_,
                'post0':softmax_output[:, 0].tolist(),
                'post1':softmax_output[:, 1].tolist()}
class CNN3(nn.Module):
    def __init__(self, num_feat=12, kernel_size=5, stride=5,):
        super(CNN3, self).__init__()
        self.conv = nn.Conv1d(1, num_feat, kernel_size=kernel_size,stride=stride)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(batch_first=True, num_layers=1, hidden_size=32, input_size=12)
        self.feat = num_feat
        self.linear = nn.Linear(32*60,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.conv(c_in)
        c_out = self.relu(c_out)
        c_out = self.lstm(c_out.permute(0, 2, 1))[0]
        c_out = c_out.reshape(c_out.shape[0], -1)
        c_out = self.linear(c_out)
        return c_out

    def info(self):
        return {
            "num_conv_layers" : 1,
            "in_features" : 1,
            "out_features" : 12,
            "kernel_size" : 5,
            "stride_size" : 5,
            "num_lstm_layers" : 1,
            "lstm_hidden_size" : 32
        }

    def results(self, out):
        softmax_output = nn.Softmax(dim=1)(out).detach().cpu()
        class_ = torch.argmax(softmax_output, dim=1).numpy()
        return {'class':class_,
                'post0':softmax_output[:, 0].tolist(),
                'post1':softmax_output[:, 1].tolist()}
class DNN3(nn.Module):
    def __init__(self, num_feat=12, kernel_size=5, stride=5,):
        super(DNN3, self).__init__()
        self.conv = nn.Conv1d(1, num_feat, kernel_size=kernel_size,stride=stride)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(batch_first=True, num_layers=1, hidden_size=32, input_size=12)
        self.feat = num_feat
        self.linear = nn.Linear(32*60,2)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.conv(c_in)
        c_out = self.relu(c_out)
        c_out = self.lstm(c_out.permute(0, 2, 1))[0]
        c_out = c_out.reshape(c_out.shape[0], -1)
        c_out = self.linear(c_out)
        return c_out

    def info(self):
        return {
            "num_conv_layers" : 1,
            "in_features" : 1,
            "out_features" : 12,
            "kernel_size" : 5,
            "stride_size" : 5,
            "num_lstm_layers" : 1,
            "lstm_hidden_size" : 32
        }

    def results(self, out):
        softmax_output = nn.Softmax(dim=1)(out).detach().cpu()
        class_ = torch.argmax(softmax_output, dim=1).numpy()
        return {'class':class_,
                'post0':softmax_output[:, 0].tolist(),
                'post1':softmax_output[:, 1].tolist()}

    # def results(self, out):
    #     score = torch.sigmoid(out).detach().cpu()
    #     pred = (score>0.5).float().tolist()
    #     print(pred)
