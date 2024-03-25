import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


device = "cpu"


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, height, width):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * self.height * self.width, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # print(batch_size)
        energy = self.projection(encoder_outputs)
        # print(energy.shape)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # print(weights.shape)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvGRU cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        """
        super(ConvGRUCell, self).__init__()
        # self.height, self.width = input_size
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.input_dim = input_dim

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, bs, h, w):
        if device != "cpu":
            return torch.zeros(bs, self.hidden_dim, h, w).cuda()
        else:
            return torch.zeros(bs, self.hidden_dim, h, w)

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, num_classes,
                 batch_first=False, bias=True, return_all_layers=False, attention=None):
        """
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.attention = attention

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)
        self.attention_layer = SelfAttention(hidden_dim[-1], self.height, self.width)
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim[-1] * self.height * self.width, self.hidden_dim[-1]),
            nn.BatchNorm1d(self.hidden_dim[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dim[-1], self.num_classes))

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        x = layer_output_list[0][:, -1, :, :, :]  # (1, 64, 6, 6)
        final_hidden_state = last_state_list[0][0]
        if self.attention:
            x = x.view(x.size(0), -1).unsqueeze(1)
            # print(x.shape)
            embedding, attn_weights = self.attention_layer(x)
            output = self.output_layer(embedding)
            # x = self.output_layer(x)
            return output
        else:
            x = x.view(x.size(0), -1)
            return self.output_layer(x)

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    height = width = 6
    channels = 256
    hidden_dim = [32, 64]
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 2  # number of stacked hidden layer
    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    num_classes=2,
                    batch_size=2,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False,
                    attention=True).to(device)

    batch_size = 2
    time_steps = 3
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width).to(device)  # (b,t,c,h,w)
    output = model(input_tensor)
    print(output.shape)
