import math
import random

import numpy as np

# torch imports
import torch
import torch.nn.functional as F
from torch import nn
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

DEFAULT_SKIP_SCALE = 0.1


class TrainableGeLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(channels, 1))  # per-channel

    def forward(self, x):
        # x shape: [batch, channels, ...]
        return self.beta * x * 0.5 * (1 + torch.erf((self.beta * x) / math.sqrt(2)))


class Crop(nn.Module):
    # crop layer is responsible for trimming the tensor from the right when creating
    # a causal convolution operation. Source: Gridin 2022

    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, x):
        if self.crop_size == 0:
            return x
        return x[:, :, : -self.crop_size]


class TemporalCausalLayer(nn.Module):
    # Source: adapted from Gridin 2022

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, outGelu=None):
        super().__init__()
        if outGelu is None:
            outGelu = True

        padding = (kernel_size - 1) * dilation
        conv_params = {"kernel_size": kernel_size, "stride": stride, "padding": padding, "dilation": dilation}

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1, self.conv2, self.crop2, self.relu2, self.dropout2)

        self.bias = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1)) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
        self.outGelu = outGelu

    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)

        if self.outGelu:
            return self.relu(y + b)

        return y + b


class TemporalConvolutionNetwork(nn.Module):
    # Source: Gridin 2022
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {"kernel_size": kernel_size, "stride": 1, "dropout": dropout}
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param["dilation"] = dilation
            tcl_param["outGelu"] = True
            tcl = TemporalCausalLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def tcn_receptive_field(kernel_size, n_layers, convs_per_block=2):
    return 1 + convs_per_block * (kernel_size - 1) * sum(2**i for i in range(n_layers))


class TCNFull(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, skipSamples):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.nLayers = len(num_channels)
        self.model_name = "TCNFull"
        self.skipSamples = skipSamples
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.tcn(x))

    def receptive_field(self):
        return tcn_receptive_field(self.kernel_size, len(self.num_channels))


class TCNFullSkip(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, skipSamples):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.nLayers = len(num_channels)
        self.model_name = "TCNFullSkip"
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        self.skipSamples = skipSamples
        self.log_scale = nn.Parameter(torch.tensor(math.log(DEFAULT_SKIP_SCALE), dtype=torch.float32))

    def forward(self, x):
        T = x.shape[-1]

        scale = torch.exp(self.log_scale)

        x_skip = F.pad(x, (self.skipSamples, 0), mode="constant", value=0)
        x_skip = x_skip[:, [0], :T]  # crop back to original length

        return self.head(self.tcn(x)) * scale + x_skip

    def receptive_field(self):
        return tcn_receptive_field(self.kernel_size, self.nLayers)




class OriginalTCNSequencePredictor(nn.Module):
    """Apply the original sliding-window TCN to a full waveform sequence."""

    def __init__(self, model, window_size=75, predict_point=65, chunk_size=32768):
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.predict_point = predict_point
        self.chunk_size = chunk_size
        self.model_name = "OriginalTCNSequencePredictor"
        self.lookaheadSamples = window_size - predict_point
        self.skipSamples = 0

    def forward(self, x):
        x = x[:, :2, :]
        batch, _, n_samples = x.shape
        out = x.new_zeros((batch, 1, n_samples))

        if n_samples < self.window_size:
            return out

        windows = x.unfold(dimension=-1, size=self.window_size, step=1)
        windows = windows.permute(0, 2, 1, 3).reshape(-1, x.shape[1], self.window_size)

        preds = []
        for start in range(0, windows.shape[0], self.chunk_size):
            preds.append(self.model(windows[start : start + self.chunk_size]))
        preds = torch.cat(preds, dim=0).reshape(batch, -1)

        start_ind = self.predict_point
        stop_ind = start_ind + preds.shape[1]
        out[:, 0, start_ind:stop_ind] = preds

        return out



#### NOT MINE ############
# Credits Johnatan
#J. B.Martin, H. E.Alderson, J. C.Gore, M. D.Does, and K. D.Harkins,
# “Modeling the MRI gradient system with a temporal convolutional network:
# Improved reconstruction by prediction of readout gradient errors,”
# Magnetic Resonance in Medicine95, no. 1 (2026): 286-298, https://doi.org/10.1002/mrm.70044.

def create_dataset(
    dataset_x,
    dataset_y,
    window_size,
    predict_point,
    pct_data_to_keep=1,
    subtract_baseline=False,
    predict_single_timepoint=False,
    return_traj_y=True,
    verbose=False,
):
    """Transform a time series into a prediction dataset. Output type float.

    Args:
        dataset_x: A numpy array of time series, first dimension is the time steps
        window_size: Size of window for prediction
        predict_point: point in window to predict. 0 first point in window
        window_spacing: Don't necessarily need continuous windows. Separate start point by n samples
        pct_data_to_keep: Don't necessarily need continuous windows. But also don't want
            regular window spacing, as this can sync up with dynamics. Randomly disperse
            samples instead
        subtract_baseline: if true, start timeseries target y at 0 for each window
        return_traj_y: if true, y is the INTEGRAL of the observed gradient up to the midpoint of the window
    """

    # check to make sure that predict_point is valid
    if predict_point > window_size:
        raise ValueError("predict_point must be less than window_size")
    if predict_point < 0:
        raise ValueError("predict_point must be greater than 0")

    X, y = [], []
    # What to do if window size is greater than the length of the dataset?? pad with 0's in that dim

    if window_size > len(dataset_x) - 1:
        while window_size > len(dataset_x) - 1:
            dataset_x = np.concatenate((dataset_x, np.zeros((1, 3))), axis=0)
            dataset_y = np.concatenate((dataset_y, np.zeros((1, 1))), axis=0)

    target_cumsum = np.cumsum(dataset_y)

    # get the indices that will be kept
    data_length = len(dataset_x) - window_size
    if pct_data_to_keep == 1:
        # use all data in dataset
        samples_to_keep = range(0, data_length, 1)
    else:
        # use a randomly selected subset
        samples_to_keep = random.sample(range(data_length), int(pct_data_to_keep * data_length))
        samples_to_keep.sort()
    if verbose:
        print(samples_to_keep)
    for i in samples_to_keep:
        feature = dataset_x[i : i + window_size, :]

        # predict a specified point in the window
        if predict_single_timepoint:
            if return_traj_y:  # noqa: SIM108
                # NOTE: predicting the GRADIENT INTEGRAL AT WINDOW MIDPOINT
                # looking at the FIRST DIFFERENCE of the cumsum
                target = target_cumsum[i + predict_point + 1] - target_cumsum[i + predict_point]
            else:
                target = dataset_y[i + predict_point]
        else:
            print("error: predict multiple timepoints not implemented")

        X.append(feature)
        y.append(target)

    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y))
    X = X.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)

    return X, y


class TCN(nn.Module):
    # Same as Temporal ConvolutionNetwork class, but with a linear output layer to collect for prediction
    # Source: Gridin 2022

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.model_name = "TCN"
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        return self.linear(y[:, :, -1])

