from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DoubleConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels is None:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels

        self.convLSTM = ConvLSTM(input_dim=self.in_channels,
                                 hidden_dim=self.mid_channels,
                                 kernel_size=(5, 5),
                                 num_layers=1,
                                 batch_first=True,
                                 bias=False,
                                 return_all_layers=False)
        self.conv = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x: ",x.shape)
        b, c, h, w = x.size()
        x = torch.reshape(x, (b // 7, 7, c, h, w))
        out1, state1 = self.convLSTM(x)
        out1 = torch.tensor([item.cpu().detach().numpy() for item in out1]).cuda(3)
        out1 = out1.squeeze(0)
        # print("out1:",out1.shape)
        b, t, c, h, w = out1.size()
        input2 = torch.reshape(out1, (b * t, c, h, w))
        input2 = self.relu1(input2)
        out2 = self.conv(input2)
        out2 = self.BN(out2)
        out2 = self.relu2(out2)
        # _, nc, nh, nw = out2.size()
        # out2 = torch.reshape(out2, (b, t, nc, nh, nw))
        return out2


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class DownLSTM(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DownLSTM, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConvLSTM(in_channels, out_channels)
        )


class UpLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpLSTM, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvLSTM(in_channels, out_channels, in_channels // 2)
            # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvLSTM(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class LastConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(LastConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}


class UNetLSTM(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNetLSTM, self).__init__()
        self.in_channels = in_channels // 7
        self.num_classes = num_classes
        self.bilinear = bilinear
        # 16,32,64,128
        self.in_conv = DoubleConv(self.in_channels, base_c)
        self.down1 = DownLSTM(base_c, base_c * 2)
        self.down2 = DownLSTM(base_c * 2, base_c * 4)
        self.down3 = DownLSTM(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownLSTM(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.conv = nn.Conv2d(base_c * 7, base_c, kernel_size=3, padding=1, bias=False)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.size()
        x = torch.reshape(x, (b, 7, c // 7, h, w))
        x = torch.reshape(x, (b * 7, c // 7, h, w))
        x1 = self.in_conv(x)
        # _, c, h, w = x1.size()
        # x1 = torch.reshape(x1, (b // 7, 7, c, h, w))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x = (b*t, c, h, w)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        b, c, h, w = x.size()
        x = torch.reshape(x, (b // 7, 7, c, h, w))
        x = torch.reshape(x, (b // 7, c * 7, h, w))
        x = self.conv(x)
        logits = self.out_conv(x)

        return {"out": logits}


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1, batch_first=True):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=batch_first)  # utilize the LSTM model in torch.nn
        # self.linear1 = nn.Linear(hidden_size, output_size) # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # print(x.shape)
        # s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # x = x.view(s * b, h)
        # x = self.linear1(x)
        # x = x.view(s, b, -1)
        return x


class U7net(nn.Module):
    def __init__(self,
                 in_channels: int = 7,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(U7net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.unet1 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        self.unet2 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        self.unet3 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        self.unet4 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        self.unet5 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        self.unet6 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        self.unet7 = UNet(in_channels // 7, num_classes, bilinear, base_c)
        # self.lstm = nn.LSTM(input_size=400*400, hidden_size=400*400, num_layers=1, batch_first=True)
        self.out_conv = OutConv(num_classes * 7, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # print(x.shape)
        x1, x2, x3, x4, x5, x6, x7 = x.split(3, 1)
        # print(x1.shape)
        y1 = self.unet1(x1)
        y2 = self.unet2(x2)
        y3 = self.unet3(x3)
        y4 = self.unet4(x4)
        y5 = self.unet5(x5)
        y6 = self.unet6(x6)
        y7 = self.unet7(x7)
        # print(x1["out"].shape)
        y = torch.cat((y1["out"], y2["out"], y3["out"], y4["out"], y5["out"], y6["out"], y7["out"]), dim=1)
        # b, c, w, h = y.shape[0], y.shape[1], y.shape[2], y.shape[3]
        # print(y.shape)
        # fy = y.flatten(2)
        # pre = self.lstm(fy)
        # pre = pre.reshape([b, c, w, h])
        logits = self.out_conv(y)
        return {"out": logits}


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # print("input_tensor.shape",input_tensor.shape)
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
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