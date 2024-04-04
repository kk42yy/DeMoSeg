import torch
from torch import nn
from training.network.ConvBlock import ConvDropoutNormNonlin

class CSA(nn.Module):
    def __init__(self, inc=512, kernel_size=3, ratio=0.25, sort_small_first=False):
        super(CSA, self).__init__()
        pass

    
class CSADropoutNormNonlin(nn.Module):
    def __init__(self, channels,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 sort_small_first=False):
        super(CSADropoutNormNonlin, self).__init__()
        pass

    
class SplitConvCSA(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        pass